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

    All operations are GPU-resident — no CPU transfers.

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
    block_table = attn_metadata.block_table_tensor
    device = block_table.device

    if len(num_prefix_tokens_per_request) != num_reqs:
        raise ValueError(
            f"Expected {num_reqs} prefix lengths, "
            f"got {len(num_prefix_tokens_per_request)}"
        )

    prefix_tokens = torch.tensor(
        num_prefix_tokens_per_request, dtype=torch.int32, device=device
    )
    num_prefix_blocks = (prefix_tokens + kv_cache_block_size - 1) // kv_cache_block_size
    num_block_blocks = (block_size + kv_cache_block_size - 1) // kv_cache_block_size
    num_total_blocks = num_prefix_blocks + num_block_blocks
    max_total_blocks = int(num_total_blocks.max().item())

    # Both prefix and block pages are contiguous from column 0 in the
    # original table, so the concatenation is bt[r, 0:n_total].
    out = torch.full(
        (num_reqs, max_total_blocks), -1, dtype=block_table.dtype, device=device
    )
    col_idx = torch.arange(max_total_blocks, device=device).unsqueeze(0)
    valid = col_idx < num_total_blocks.unsqueeze(1)
    # Clamp indices to block_table column range
    src_idx = col_idx.clamp(max=block_table.shape[1] - 1)
    out[valid] = block_table.gather(1, src_idx.expand(num_reqs, -1))[valid]

    combined_seq_lens = prefix_tokens.to(attn_metadata.seq_lens.dtype) + block_size
    max_combined_seq_len = int(combined_seq_lens.max().item())

    return replace(
        attn_metadata,
        seq_lens=combined_seq_lens,
        max_seq_len=max_combined_seq_len,
        block_table_tensor=out,
        causal=False,
        _seq_lens_cpu=combined_seq_lens.cpu(),
    )
