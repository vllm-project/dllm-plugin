"""Virtual batch decomposition for block-style attention.

Following vLLM's chunked_local_attention pattern, transforms attention metadata
to create virtual batches for prefix and block attention chunks.

Supports both V1 CommonAttentionMetadata and V2 FlashAttentionMetadata.

Reference: vllm/model_executor/layers/attention/chunked_local_attention.py
"""

from __future__ import annotations

import numpy as np
import torch

# vLLM imports (centralized in vllm_compat for version handling)
from dllm_plugin.vllm_compat import CommonAttentionMetadata


def _is_v2_metadata(attn_metadata) -> bool:
    """Check if metadata is V2 FlashAttentionMetadata."""
    # V2 has 'use_cascade', V1 does not
    return hasattr(attn_metadata, "use_cascade")


def make_block_attention_virtual_batches(
    attn_metadata: CommonAttentionMetadata,
    num_prefix_tokens_per_request: list[int],
    block_size: int,
    kv_cache_block_size: int = 16,
) -> tuple[CommonAttentionMetadata | None, CommonAttentionMetadata]:
    """Transform metadata for block-style dual-chunk attention (multi-request).

    Adapts vLLM's virtual batches pattern to handle heterogeneous prefix lengths
    across multiple requests in a unified batch.

    Creates two virtual batches:
    1. Prefix chunk: Q=current_block (block_size tokens), KV=prefix
       (heterogeneous lengths)
    2. Block chunk: Q=current_block (block_size tokens), KV=current_block
       (uniform block_size)

    Each virtual batch gets its own:
    - seq_lens: Length of KV for that chunk
      (heterogeneous for prefix, uniform for block)
    - block_table: KV cache pages accessible to that chunk
    - query_start_loc: Position offsets in the query tensor

    Args:
        attn_metadata: Original CommonAttentionMetadata from vLLM
        num_prefix_tokens_per_request: Per-request prefix lengths
            (list[int] matching num_reqs)
        block_size: Size of current generation block (typically 32)
        kv_cache_block_size: KV cache block size (default 16, should be
            queried from cache_config in future)

    Returns:
        (prefix_metadata, block_metadata): Unified virtual batch metadata
            prefix_metadata is None if all requests have num_prefix_tokens == 0

    Reference:
        vllm/v1/attention/backends/utils.py:make_local_attention_virtual_batches()
    """
    # V2 compatibility: FlashAttentionMetadata has different structure
    if _is_v2_metadata(attn_metadata):
        return _make_block_attention_virtual_batches_v2(
            attn_metadata,
            num_prefix_tokens_per_request,
            block_size,
            kv_cache_block_size,
        )

    # V1 path (CommonAttentionMetadata)
    # Get number of requests from seq_lens
    num_reqs = (
        len(attn_metadata.seq_lens)
        if hasattr(attn_metadata, "seq_lens")
        else attn_metadata.batch_size
    )
    device = (
        attn_metadata.seq_lens.device
        if hasattr(attn_metadata, "seq_lens")
        else attn_metadata.query_start_loc.device
    )
    total_query_tokens = attn_metadata.num_actual_tokens

    # Validate input
    if len(num_prefix_tokens_per_request) != num_reqs:
        raise ValueError(
            f"Expected {num_reqs} prefix lengths, "
            f"got {len(num_prefix_tokens_per_request)}"
        )

    # Convert to numpy for indexing calculations (vLLM pattern)
    num_prefix_tokens_np = np.array(num_prefix_tokens_per_request, dtype=np.int32)
    max_prefix_tokens = int(num_prefix_tokens_np.max())

    # Edge case: First block (all requests have no prefix)
    if max_prefix_tokens == 0:
        # Only block self-attention, no prefix chunk
        # Calculate blocks needed for block chunk
        num_block_blocks = (block_size + kv_cache_block_size - 1) // kv_cache_block_size
        block_block_table = attn_metadata.block_table_tensor[:, :num_block_blocks]

        block_metadata = CommonAttentionMetadata(
            query_start_loc=attn_metadata.query_start_loc,
            query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
            seq_lens=torch.full(
                (num_reqs,), block_size, dtype=torch.int32, device=device
            ),
            num_reqs=num_reqs,
            num_actual_tokens=total_query_tokens,
            max_query_len=block_size,
            max_seq_len=block_size,
            block_table_tensor=block_block_table,
            slot_mapping=attn_metadata.slot_mapping,
            causal=False,  # Non-causal (bidirectional within block)
        )
        return None, block_metadata

    # Compute per-request prefix blocks (heterogeneous)
    num_prefix_blocks_per_req = (
        num_prefix_tokens_np + kv_cache_block_size - 1
    ) // kv_cache_block_size
    max_prefix_blocks = int(num_prefix_blocks_per_req.max())

    # --- Virtual Batch 1: Prefix chunk (heterogeneous KV lengths) ---
    # Query: current block (block_size tokens)
    # KV: prefix (heterogeneous num_prefix_tokens per request)

    # Build block table for prefix chunk
    # Each request gets its own prefix block range from original block table
    prefix_block_table_list = []
    for req_idx in range(num_reqs):
        n_blocks = int(num_prefix_blocks_per_req[req_idx])
        if n_blocks > 0:
            req_blocks = attn_metadata.block_table_tensor[req_idx, :n_blocks]
        else:
            # Edge case: this request has no prefix (first block)
            req_blocks = torch.empty(0, dtype=torch.int32, device=device)

        # Pad to max_prefix_blocks for rectangular tensor
        # Use -1 sentinel (vLLM convention) - 0 is a valid page ID
        if len(req_blocks) < max_prefix_blocks:
            padding = torch.full(
                (max_prefix_blocks - len(req_blocks),),
                fill_value=-1,
                dtype=torch.int32,
                device=device,
            )
            req_blocks = torch.cat([req_blocks, padding])

        prefix_block_table_list.append(req_blocks)

    # Stack into unified virtual batch [num_reqs, max_prefix_blocks]
    prefix_block_table = torch.stack(prefix_block_table_list, dim=0)

    # Build seq_lens for prefix chunk (heterogeneous!)
    prefix_seq_lens = torch.tensor(
        num_prefix_tokens_np,
        dtype=torch.int32,
        device=device,
    )

    prefix_metadata = CommonAttentionMetadata(
        query_start_loc=attn_metadata.query_start_loc,
        query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
        seq_lens=prefix_seq_lens,  # Heterogeneous!
        num_reqs=num_reqs,
        num_actual_tokens=total_query_tokens,
        max_query_len=block_size,
        max_seq_len=max_prefix_tokens,
        block_table_tensor=prefix_block_table,
        slot_mapping=attn_metadata.slot_mapping,
        causal=False,  # Non-causal (all queries attend to all prefix keys)
    )

    # --- Virtual Batch 2: Block chunk (uniform block_size) ---
    # Query: current block (block_size tokens)
    # KV: current block (block_size tokens, from CURRENT forward pass, not cache)
    #
    # IMPORTANT: Block chunk uses current forward pass K/V tensors (passed explicitly
    # to attn() via key=key, value=value), NOT cached KV. The block_table here is
    # used by vLLM to WRITE the current block KV to cache (via slot_mapping),
    # not to READ from cache. This is bidirectional attention within the current block.
    #
    # CRITICAL: Extract block pages per-request based on ACTUAL prefix length,
    # not max_prefix_blocks. In heterogeneous batches, each request's current
    # block starts at a different position in its block table.
    # (vLLM paged cache: block_table maps logical block positions to physical pages)

    num_block_blocks = (block_size + kv_cache_block_size - 1) // kv_cache_block_size
    block_block_table_list = []

    for req_idx in range(num_reqs):
        # Each request's block chunk pages start after ITS OWN prefix blocks
        n_prefix_blocks = int(num_prefix_blocks_per_req[req_idx])
        block_start_idx = n_prefix_blocks
        block_end_idx = block_start_idx + num_block_blocks

        # P1-2: Validate block table bounds before slicing
        block_table_cols = attn_metadata.block_table_tensor.shape[1]
        if block_end_idx > block_table_cols:
            raise ValueError(
                f"Request {req_idx} block chunk requires pages "
                f"[{block_start_idx}:{block_end_idx}] but block_table only has "
                f"{block_table_cols} columns (prefix_blocks={n_prefix_blocks}, "
                f"block_blocks={num_block_blocks}, block_size={block_size}, "
                f"kv_cache_block_size={kv_cache_block_size})"
            )

        # Extract physical page IDs for this request's current block
        req_block_pages = attn_metadata.block_table_tensor[
            req_idx, block_start_idx:block_end_idx
        ]
        block_block_table_list.append(req_block_pages)

    # Stack into unified virtual batch [num_reqs, num_block_blocks]
    block_block_table = torch.stack(block_block_table_list, dim=0)

    block_metadata = CommonAttentionMetadata(
        query_start_loc=attn_metadata.query_start_loc,
        query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
        seq_lens=torch.full((num_reqs,), block_size, dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=total_query_tokens,
        max_query_len=block_size,
        max_seq_len=block_size,
        block_table_tensor=block_block_table,
        slot_mapping=attn_metadata.slot_mapping,
        causal=False,  # Non-causal (bidirectional within block)
    )

    return prefix_metadata, block_metadata


def _make_block_attention_virtual_batches_v2(
    attn_metadata,  # FlashAttentionMetadata
    num_prefix_tokens_per_request: list[int],
    block_size: int,
    kv_cache_block_size: int = 16,
):
    """V2 version for FlashAttentionMetadata.

    V2 differences from V1:
    - num_reqs is implicit: len(query_start_loc) - 1
    - block_table instead of block_table
    - Returns FlashAttentionMetadata instead of CommonAttentionMetadata
    """
    device = attn_metadata.query_start_loc.device
    # V2: num_reqs is implicit in query_start_loc shape
    num_reqs = len(attn_metadata.query_start_loc) - 1
    total_query_tokens = attn_metadata.num_actual_tokens

    # Validate input
    if len(num_prefix_tokens_per_request) != num_reqs:
        raise ValueError(
            f"Expected {num_reqs} prefix lengths, "
            f"got {len(num_prefix_tokens_per_request)}"
        )

    # Convert to numpy for indexing calculations
    num_prefix_tokens_np = np.array(num_prefix_tokens_per_request, dtype=np.int32)
    max_prefix_tokens = int(num_prefix_tokens_np.max())

    # Edge case: First block (all requests have no prefix)
    if max_prefix_tokens == 0:
        # Only block self-attention, no prefix chunk
        # For first block with no prefix, we can just use the original metadata
        # (it already has the correct KV range for the current block)
        # No need for dual-chunk decomposition
        return None, attn_metadata

    # Compute per-request prefix blocks (heterogeneous)
    num_prefix_blocks_per_req = (
        num_prefix_tokens_np + kv_cache_block_size - 1
    ) // kv_cache_block_size
    max_prefix_blocks = int(num_prefix_blocks_per_req.max())

    # --- Virtual Batch 1: Prefix chunk (heterogeneous KV lengths) ---
    prefix_block_table_list = []
    for req_idx in range(num_reqs):
        n_blocks = int(num_prefix_blocks_per_req[req_idx])
        if n_blocks > 0:
            req_blocks = attn_metadata.block_table_tensor[req_idx, :n_blocks]
        else:
            req_blocks = torch.empty(0, dtype=torch.int32, device=device)

        # Pad to max_prefix_blocks
        if len(req_blocks) < max_prefix_blocks:
            padding = torch.full(
                (max_prefix_blocks - len(req_blocks),),
                fill_value=-1,
                dtype=torch.int32,
                device=device,
            )
            req_blocks = torch.cat([req_blocks, padding])

        prefix_block_table_list.append(req_blocks)

    prefix_block_table = torch.stack(prefix_block_table_list, dim=0)
    prefix_seq_lens = torch.tensor(
        num_prefix_tokens_np,
        dtype=torch.int32,
        device=device,
    )

    prefix_metadata = type(attn_metadata)(
        num_actual_tokens=total_query_tokens,
        max_query_len=block_size,
        query_start_loc=attn_metadata.query_start_loc,
        max_seq_len=max_prefix_tokens,
        seq_lens=prefix_seq_lens,  # Heterogeneous!
        block_table_tensor=prefix_block_table,
        slot_mapping=attn_metadata.slot_mapping,
        use_cascade=attn_metadata.use_cascade,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=0,
        dcp_context_kv_lens=None,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        causal=False,  # Non-causal prefix attention
    )

    # --- Virtual Batch 2: Block chunk (uniform block_size) ---
    num_block_blocks = (block_size + kv_cache_block_size - 1) // kv_cache_block_size
    block_block_table_list = []

    for req_idx in range(num_reqs):
        n_prefix_blocks = int(num_prefix_blocks_per_req[req_idx])
        block_start_idx = n_prefix_blocks
        block_end_idx = block_start_idx + num_block_blocks

        # Validate block table bounds
        block_table_cols = attn_metadata.block_table_tensor.shape[1]
        if block_end_idx > block_table_cols:
            raise ValueError(
                f"Request {req_idx} block chunk requires pages "
                f"[{block_start_idx}:{block_end_idx}] but block_table only has "
                f"{block_table_cols} columns"
            )

        req_block_pages = attn_metadata.block_table_tensor[
            req_idx, block_start_idx:block_end_idx
        ]
        block_block_table_list.append(req_block_pages)

    block_block_table = torch.stack(block_block_table_list, dim=0)

    block_metadata = type(attn_metadata)(
        num_actual_tokens=total_query_tokens,
        max_query_len=block_size,
        query_start_loc=attn_metadata.query_start_loc,
        max_seq_len=block_size,
        seq_lens=torch.full((num_reqs,), block_size, dtype=torch.int32, device=device),
        block_table_tensor=block_block_table,
        slot_mapping=attn_metadata.slot_mapping,
        use_cascade=attn_metadata.use_cascade,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=0,
        dcp_context_kv_lens=None,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        causal=False,  # Bidirectional within block
    )

    return prefix_metadata, block_metadata
