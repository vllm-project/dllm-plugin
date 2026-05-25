# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for block diffusion remasking.

Two-kernel design per expert guidance:
  Kernel 1: online softmax + argmax per (batch, position)
  Kernel 2: cross-position max reduction + threshold + commit per batch

Drop-in replacement for batched_remask(). Numerically equivalent.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"VOCAB_BLOCK": 1024}, num_warps=4),
        triton.Config({"VOCAB_BLOCK": 2048}, num_warps=8),
        triton.Config({"VOCAB_BLOCK": 4096}, num_warps=8),
        triton.Config({"VOCAB_BLOCK": 8192}, num_warps=16),
    ],
    key=["vocab_size"],
)
@triton.jit
def _argmax_confidence_kernel(
    logits_ptr,
    draft_ptr,
    argmax_out_ptr,
    confidence_out_ptr,
    mask_token_id: tl.constexpr,
    vocab_size: tl.constexpr,
    logits_batch_stride,
    logits_pos_stride,
    draft_batch_stride,
    out_batch_stride,
    VOCAB_BLOCK: tl.constexpr,
):
    """Kernel 1: online softmax + argmax for one (batch, position).

    Each program handles one position in one batch element.
    Iterates over vocab in VOCAB_BLOCK chunks, tracking argmax and
    computing softmax probability of the argmax token via online
    softmax (single pass, no intermediate probability tensor).
    """
    batch_idx = tl.program_id(0)
    pos_idx = tl.program_id(1)

    draft_offset = batch_idx * draft_batch_stride + pos_idx
    current_token = tl.load(draft_ptr + draft_offset)

    out_offset = batch_idx * out_batch_stride + pos_idx
    neg_inf = -1e30

    if current_token != mask_token_id:
        tl.store(argmax_out_ptr + out_offset, current_token)
        tl.store(confidence_out_ptr + out_offset, neg_inf)
        return

    logits_base = batch_idx * logits_batch_stride + pos_idx * logits_pos_stride

    best_val = neg_inf
    best_idx: tl.int64 = 0
    running_max = neg_inf
    sum_exp = 0.0

    for v_start in range(0, vocab_size, VOCAB_BLOCK):
        v_offsets = v_start + tl.arange(0, VOCAB_BLOCK)
        v_mask = v_offsets < vocab_size
        logits_block = tl.load(
            logits_ptr + logits_base + v_offsets,
            mask=v_mask,
            other=neg_inf,
        ).to(tl.float32)

        # rm_mask: prevent selecting mask_token_id as prediction
        is_mask_tok = v_offsets == mask_token_id
        logits_block = tl.where(is_mask_tok, neg_inf, logits_block)

        block_max = tl.max(logits_block)
        if block_max > best_val:
            best_val = block_max
            best_idx = v_start + tl.argmax(logits_block, axis=0)

        new_max = tl.maximum(running_max, block_max)
        sum_exp = sum_exp * tl.exp(running_max - new_max)
        sum_exp += tl.sum(tl.exp(logits_block - new_max))
        running_max = new_max

    confidence = tl.exp(best_val - running_max) / sum_exp

    tl.store(argmax_out_ptr + out_offset, best_idx)
    tl.store(confidence_out_ptr + out_offset, confidence)


@triton.jit
def _threshold_commit_kernel(
    draft_ptr,
    argmax_ptr,
    confidence_ptr,
    out_draft_ptr,
    all_done_ptr,
    num_transferred_ptr,
    mask_token_id: tl.constexpr,
    threshold: tl.constexpr,
    block_size: tl.constexpr,
    draft_batch_stride,
    out_batch_stride,
):
    """Kernel 2: cross-position max reduction + threshold + commit.

    One program per batch element. Processes all block_size positions.
    """
    batch_idx = tl.program_id(0)

    pos_offsets = tl.arange(0, block_size)
    conf_offset = batch_idx * out_batch_stride + pos_offsets

    confidences = tl.load(confidence_ptr + conf_offset)

    max_confidence = tl.max(confidences)
    actual_threshold = max_confidence - 1e-5
    actual_threshold = tl.minimum(actual_threshold, threshold)
    actual_threshold = tl.maximum(actual_threshold, -1000.0)

    draft_offset = batch_idx * draft_batch_stride + pos_offsets
    current_tokens = tl.load(draft_ptr + draft_offset)
    argmax_tokens = tl.load(argmax_ptr + conf_offset)

    is_masked = current_tokens == mask_token_id
    above_threshold = confidences >= actual_threshold
    transfer = is_masked & above_threshold

    new_tokens = tl.where(transfer, argmax_tokens, current_tokens)
    tl.store(out_draft_ptr + draft_offset, new_tokens)

    all_resolved = tl.sum((new_tokens != mask_token_id).to(tl.int32)) == block_size
    num_transferred = tl.sum(transfer.to(tl.int32))

    tl.store(all_done_ptr + batch_idx, all_resolved.to(tl.int32))
    tl.store(num_transferred_ptr + batch_idx, num_transferred)


def batched_remask_triton(
    logits: torch.Tensor,
    input_draft: torch.Tensor,
    mask_token_id: int,
    threshold: float,
    temperature: float = 0.0,
    use_float64: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Triton remasking — drop-in replacement for batched_remask().

    Two-kernel design:
      Kernel 1: online softmax + argmax per (batch, position)
      Kernel 2: cross-position threshold + commit per batch

    Falls back to PyTorch for temperature > 0 (Gumbel noise requires
    float64 stochastic ops that don't benefit from Triton fusion).
    """
    if temperature > 0:
        from dllm_plugin.sampling.diffusion_sampler import batched_remask

        return batched_remask(
            logits,
            input_draft,
            mask_token_id,
            threshold,
            temperature,
            use_float64,
            out,
        )

    batch_size, block_size, vocab_size = logits.shape
    device = logits.device

    logits_f = logits.float().contiguous()
    argmax_out = torch.empty(batch_size, block_size, dtype=torch.int64, device=device)
    confidence_out = torch.empty(
        batch_size, block_size, dtype=torch.float32, device=device
    )

    _argmax_confidence_kernel[(batch_size, block_size)](
        logits_f,
        input_draft,
        argmax_out,
        confidence_out,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        logits_batch_stride=logits_f.stride(0),
        logits_pos_stride=logits_f.stride(1),
        draft_batch_stride=input_draft.stride(0),
        out_batch_stride=argmax_out.stride(0),
    )

    if out is not None:
        out_draft = out
        out_draft.copy_(input_draft)
    else:
        out_draft = input_draft.clone()
    all_done = torch.zeros(batch_size, dtype=torch.int32, device=device)
    num_transferred = torch.zeros(batch_size, dtype=torch.int32, device=device)

    _threshold_commit_kernel[(batch_size,)](
        input_draft,
        argmax_out,
        confidence_out,
        out_draft,
        all_done,
        num_transferred,
        mask_token_id=mask_token_id,
        threshold=threshold,
        block_size=block_size,
        draft_batch_stride=input_draft.stride(0),
        out_batch_stride=argmax_out.stride(0),
    )

    return out_draft, all_done.bool(), num_transferred
