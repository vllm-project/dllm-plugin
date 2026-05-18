"""Concatenated virtual batch for block-causal attention.

This module implements the FIX for the softmax normalization bug in dual-chunk
attention. Instead of TWO separate attention calls with TWO separate softmax
normalizations (which causes weights to sum to 2.0), we create ONE virtual
batch combining prefix + block KV, resulting in ONE softmax normalization
(weights sum to 1.0).

Mathematical correctness:
- Dual-chunk (WRONG): sum(prefix_weights) + sum(block_weights) = 1.0 + 1.0 = 2.0
- Concatenated (CORRECT): sum(combined_weights) = 1.0

This function operates on CommonAttentionMetadata (backend-agnostic) and is
called inside the builder's build() method, BEFORE the backend-specific
transformation (FlashInfer or FlashAttention). The builder then transforms
the modified CommonAttentionMetadata into the correct backend format.
"""

from dataclasses import replace

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

    Modifies CommonAttentionMetadata so that each request's block_table and
    seq_lens span both the committed prefix pages and the current block pages.
    The backend builder (FlashInfer/FlashAttention) then transforms this into
    the correct backend-specific format.

    Args:
        attn_metadata: Original CommonAttentionMetadata from vLLM
        num_prefix_tokens_per_request: Per-request prefix lengths (list[int])
        block_size: Size of current generation block (typically 32)
        kv_cache_block_size: KV cache block size (typically 16)

    Returns:
        New CommonAttentionMetadata with concatenated block_table_tensor and
        updated seq_lens reflecting prefix + block per request.
    """
    num_reqs = attn_metadata.num_reqs
    device = attn_metadata.block_table_tensor.device

    if len(num_prefix_tokens_per_request) != num_reqs:
        raise ValueError(
            f"Expected {num_reqs} prefix lengths, "
            f"got {len(num_prefix_tokens_per_request)}"
        )

    num_prefix_tokens_np = np.array(num_prefix_tokens_per_request, dtype=np.int32)
    num_prefix_blocks_np = (
        num_prefix_tokens_np + kv_cache_block_size - 1
    ) // kv_cache_block_size
    num_block_blocks = (block_size + kv_cache_block_size - 1) // kv_cache_block_size

    num_total_blocks_np = num_prefix_blocks_np + num_block_blocks
    max_total_blocks = int(num_total_blocks_np.max())

    block_table = attn_metadata.block_table_tensor
    bt_np = block_table.cpu().numpy()
    num_bt_cols = bt_np.shape[1]

    # Vectorized construction: build output block table in numpy
    out_np = np.full((num_reqs, max_total_blocks), -1, dtype=bt_np.dtype)
    for req_idx in range(num_reqs):
        n_prefix = int(num_prefix_blocks_np[req_idx])
        block_start = n_prefix
        block_end = min(block_start + num_block_blocks, num_bt_cols)

        if n_prefix > 0:
            out_np[req_idx, :n_prefix] = bt_np[req_idx, :n_prefix]
        if block_end > block_start:
            n_copy = block_end - block_start
            out_np[req_idx, n_prefix : n_prefix + n_copy] = bt_np[
                req_idx, block_start:block_end
            ]

    concatenated_block_table = torch.from_numpy(out_np).to(device)

    combined_seq_lens = torch.tensor(
        num_prefix_tokens_np + block_size,
        dtype=attn_metadata.seq_lens.dtype,
        device=device,
    )
    max_combined_seq_len = int(combined_seq_lens.max())

    return replace(
        attn_metadata,
        seq_lens=combined_seq_lens,
        max_seq_len=max_combined_seq_len,
        block_table_tensor=concatenated_block_table,
        causal=False,
        _seq_lens_cpu=None,
    )
