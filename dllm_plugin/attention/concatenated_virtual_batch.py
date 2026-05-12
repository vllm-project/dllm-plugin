"""Concatenated virtual batch for block-causal attention.

This module implements the FIX for the softmax normalization bug in dual-chunk
attention. Instead of TWO separate attention calls with TWO separate softmax
normalizations (which causes weights to sum to 2.0), we create ONE virtual
batch combining prefix + block KV, resulting in ONE softmax normalization
(weights sum to 1.0).

Mathematical correctness:
- Dual-chunk (WRONG): sum(prefix_weights) + sum(block_weights) = 1.0 + 1.0 = 2.0 ❌
- Concatenated (CORRECT): sum(combined_weights) = 1.0 ✅

See: ROOT_CAUSE_SOFTMAX_NORMALIZATION.md for full analysis.
"""

import numpy as np
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata


def create_concatenated_virtual_batch(
    attn_metadata: CommonAttentionMetadata,
    num_prefix_tokens_per_request: list[int],
    block_size: int,
    kv_cache_block_size: int,
) -> CommonAttentionMetadata:
    """Create ONE virtual batch combining prefix + block KV.

    This fixes the softmax normalization bug by creating a single virtual batch
    that spans BOTH the prefix (cached KV) and current block (just-cached KV),
    resulting in a single softmax normalization over ALL keys.

    How it works:
    1. Current block KV is written to cache BEFORE attention
       (via unified_kv_cache_update)
    2. FlashAttention reads BOTH prefix KV (already cached) + block KV
       (just cached) using the concatenated block_table
    3. Single attention call → single softmax → mathematically correct!

    Args:
        attn_metadata: Original CommonAttentionMetadata from vLLM
        num_prefix_tokens_per_request: Per-request prefix lengths (list[int])
        block_size: Size of current generation block (typically 32)
        kv_cache_block_size: KV cache block size (typically 16)

    Returns:
        CommonAttentionMetadata with:
        - seq_lens = num_prefix_tokens + block_size per request
        - block_table = [prefix_pages | current_block_pages] concatenated
        - Single attention call → single softmax → correct normalization ✅

    Example:
        Request 0: 64 prefix tokens + 32 block tokens = 96 total
        Request 1: 32 prefix tokens + 32 block tokens = 64 total

        With kv_cache_block_size=16:
        - Request 0: 4 prefix pages + 2 block pages = 6 pages total
        - Request 1: 2 prefix pages + 2 block pages = 4 pages total

        block_table = [
            [p0, p1, p2, p3, p4, p5],  # Req 0: prefix(4) + block(2)
            [p0, p1, p2, p3, -1, -1],  # Req 1: prefix(2) + block(2) + padding
        ]

        FlashAttention will read all pages via block_table and apply
        a SINGLE softmax over all 96 (or 64) keys per request.
    """
    num_reqs = attn_metadata.num_reqs
    device = attn_metadata.query_start_loc.device

    # Validate input
    if len(num_prefix_tokens_per_request) != num_reqs:
        raise ValueError(
            f"Expected {num_reqs} prefix lengths, "
            f"got {len(num_prefix_tokens_per_request)}"
        )

    # Convert to numpy for indexing (vLLM pattern)
    num_prefix_tokens_np = np.array(num_prefix_tokens_per_request, dtype=np.int32)

    # Compute blocks needed for prefix and current block
    num_prefix_blocks_per_req = (
        num_prefix_tokens_np + kv_cache_block_size - 1
    ) // kv_cache_block_size
    num_block_blocks = (block_size + kv_cache_block_size - 1) // kv_cache_block_size

    # Total blocks per request = prefix_blocks + block_blocks
    num_total_blocks_per_req = num_prefix_blocks_per_req + num_block_blocks
    max_total_blocks = int(num_total_blocks_per_req.max())

    # --- Build concatenated block table: [prefix_pages | block_pages] ---
    concatenated_block_table_list = []

    for req_idx in range(num_reqs):
        n_prefix_blocks = int(num_prefix_blocks_per_req[req_idx])

        # Get prefix pages (already in cache from previous blocks)
        if n_prefix_blocks > 0:
            prefix_pages = attn_metadata.block_table_tensor[req_idx, :n_prefix_blocks]
        else:
            # First block: no prefix
            prefix_pages = torch.empty(0, dtype=torch.int32, device=device)

        # Get current block pages (about to be written to cache)
        block_start_idx = n_prefix_blocks
        block_end_idx = block_start_idx + num_block_blocks

        # Validate block table bounds
        block_table_cols = attn_metadata.block_table_tensor.shape[1]
        if block_end_idx > block_table_cols:
            raise ValueError(
                f"Request {req_idx} requires pages [{block_start_idx}:{block_end_idx}] "
                f"but block_table only has {block_table_cols} columns "
                f"(prefix_blocks={n_prefix_blocks}, block_blocks={num_block_blocks})"
            )

        block_pages = attn_metadata.block_table_tensor[
            req_idx, block_start_idx:block_end_idx
        ]

        # Concatenate: [prefix_pages | block_pages]
        if n_prefix_blocks > 0:
            req_pages = torch.cat([prefix_pages, block_pages])
        else:
            req_pages = block_pages

        # Pad to max_total_blocks for rectangular tensor
        # Use -1 sentinel (vLLM convention for unused pages)
        if len(req_pages) < max_total_blocks:
            padding = torch.full(
                (max_total_blocks - len(req_pages),),
                fill_value=-1,
                dtype=torch.int32,
                device=device,
            )
            req_pages = torch.cat([req_pages, padding])

        concatenated_block_table_list.append(req_pages)

    # Stack into unified block table [num_reqs, max_total_blocks]
    concatenated_block_table = torch.stack(concatenated_block_table_list, dim=0)

    # --- Build combined seq_lens: prefix + block per request ---
    combined_seq_lens = torch.tensor(
        num_prefix_tokens_np + block_size,
        dtype=torch.int32,
        device=device,
    )

    max_combined_seq_len = int(combined_seq_lens.max())

    # --- Create unified CommonAttentionMetadata ---
    return CommonAttentionMetadata(
        query_start_loc=attn_metadata.query_start_loc,
        query_start_loc_cpu=attn_metadata.query_start_loc_cpu,
        seq_lens=combined_seq_lens,               # prefix + block (heterogeneous)
        num_reqs=num_reqs,
        num_actual_tokens=attn_metadata.num_actual_tokens,
        max_query_len=block_size,                 # Query is current block
        max_seq_len=max_combined_seq_len,         # max(prefix + block)
        block_table_tensor=concatenated_block_table,
        slot_mapping=attn_metadata.slot_mapping,
        causal=False,  # Bidirectional within block, block-causal across
        seq_lens_cpu_upper_bound=attn_metadata.seq_lens_cpu_upper_bound
        if hasattr(attn_metadata, "seq_lens_cpu_upper_bound")
        else None,
        _seq_lens_cpu=combined_seq_lens.cpu()
        if hasattr(attn_metadata, "_seq_lens_cpu")
        else None,
    )
