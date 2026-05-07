"""Virtual batch decomposition for block-style attention.

Following vLLM's chunked_local_attention pattern, transforms CommonAttentionMetadata
to create virtual batches for prefix and block attention chunks.

Reference: vllm/model_executor/layers/attention/chunked_local_attention.py
"""

from __future__ import annotations

import torch

# vLLM imports (centralized in vllm_compat for version handling)
from dllm_plugin.vllm_compat import CommonAttentionMetadata


def make_block_attention_virtual_batches(
    attn_metadata: CommonAttentionMetadata,
    num_prefix_tokens: int,
    block_size: int,
    kv_cache_block_size: int = 16,
) -> tuple[CommonAttentionMetadata | None, CommonAttentionMetadata]:
    """Transform metadata for block-style dual-chunk attention.

    Creates two virtual batches per request:
    1. Prefix chunk: Q=current_block (block_size tokens), KV=prefix (num_prefix_tokens)
    2. Block chunk: Q=current_block (block_size tokens), KV=current_block (block_size)

    Each virtual batch gets its own:
    - seq_lens: Length of KV for that chunk
    - block_table: KV cache pages accessible to that chunk
    - query_start_loc: Position offsets in the query tensor

    Args:
        attn_metadata: Original CommonAttentionMetadata from vLLM
        num_prefix_tokens: Number of committed tokens (prefix length)
        block_size: Size of current generation block (typically 32)
        kv_cache_block_size: KV cache block size (default 16, should be
            queried from cache_config in future)

    Returns:
        (prefix_metadata, block_metadata): Transformed metadata for each chunk
            prefix_metadata is None if num_prefix_tokens == 0

    Raises:
        NotImplementedError: If num_reqs > 1 (multi-request batching not yet supported)
    """
    device = attn_metadata.query_start_loc.device
    num_reqs = attn_metadata.num_reqs
    total_query_tokens = attn_metadata.num_actual_tokens

    # MVP limitation: Only single-request batches supported
    # Multi-request batching with heterogeneous prefix lengths requires
    # per-request virtual batch transformation (deferred to Phase 7.1)
    if num_reqs > 1:
        raise NotImplementedError(
            "LLaDA2.0 virtual batch attention does not support multi-request "
            "batching in this release (MVP Phase 7). Use max_num_seqs=1 or "
            "wait for Phase 7.1 update. See docs/OPERATOR_LLaDA2.md for details."
        )

    # Edge case: First block (no prefix)
    if num_prefix_tokens == 0:
        # Only block self-attention, no prefix chunk
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
            block_table_tensor=attn_metadata.block_table_tensor,
            slot_mapping=attn_metadata.slot_mapping,
            causal=False,  # Non-causal (bidirectional within block)
        )
        return None, block_metadata

    # Calculate how many KV cache pages we need for prefix and block
    # Assuming block_table has shape [num_reqs, max_num_blocks_per_seq]
    # We need to slice it to get only the pages for prefix vs block

    # Calculate blocks needed for prefix using configured KV cache block size
    num_prefix_blocks = (
        num_prefix_tokens + kv_cache_block_size - 1
    ) // kv_cache_block_size

    # Slice block_table for each chunk
    prefix_block_table = attn_metadata.block_table_tensor[:, :num_prefix_blocks]
    block_start_idx = num_prefix_blocks
    num_block_blocks = (block_size + kv_cache_block_size - 1) // kv_cache_block_size
    block_end_idx = block_start_idx + num_block_blocks
    block_block_table = attn_metadata.block_table_tensor[
        :, block_start_idx:block_end_idx
    ]

    # --- Virtual Batch 1: Prefix chunk ---
    # Query: current block (block_size tokens)
    # KV: prefix (num_prefix_tokens)

    prefix_metadata = CommonAttentionMetadata(
        query_start_loc=attn_metadata.query_start_loc,
        query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
        seq_lens=torch.full(
            (num_reqs,), num_prefix_tokens, dtype=torch.int32, device=device
        ),
        num_reqs=num_reqs,
        num_actual_tokens=total_query_tokens,
        max_query_len=block_size,
        max_seq_len=num_prefix_tokens,
        block_table_tensor=prefix_block_table,
        slot_mapping=attn_metadata.slot_mapping,  # May need adjustment
        causal=False,  # Non-causal (all queries attend to all prefix keys)
    )

    # --- Virtual Batch 2: Block chunk ---
    # Query: current block (block_size tokens)
    # KV: current block (block_size tokens)

    block_metadata = CommonAttentionMetadata(
        query_start_loc=attn_metadata.query_start_loc,
        query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
        seq_lens=torch.full((num_reqs,), block_size, dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=total_query_tokens,
        max_query_len=block_size,
        max_seq_len=block_size,
        block_table_tensor=block_block_table,
        slot_mapping=attn_metadata.slot_mapping,  # May need adjustment
        causal=False,  # Non-causal (bidirectional within block)
    )

    return prefix_metadata, block_metadata
