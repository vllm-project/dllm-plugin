# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched GPU diffusion sampler for LLaDA2 block denoising.

Called from LLaDA2ModelState.custom_sample() with batched PyTorch
operations on GPU. No CPU-GPU sync during the hot path.

Each step:
1. Compute softmax probabilities from logits (batched)
2. Threshold-based commit: positions with max_prob > threshold commit
3. Argmax for committed positions, mask_token_id for non-committed
4. Check convergence: if all positions committed, block is done
5. Return (canvas, num_sampled) where num_sampled=block_size if done, 0 if not
"""

from __future__ import annotations

import torch


def batched_remask(
    logits: torch.Tensor,
    input_draft: torch.Tensor,
    mask_token_id: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched remasking on GPU.

    Args:
        logits: [batch, block_size, vocab_size] — model output logits
        input_draft: [batch, block_size] — current canvas token IDs
        mask_token_id: token ID for masked positions
        threshold: softmax probability threshold for committing

    Returns:
        canvas: [batch, block_size] — updated canvas (committed + masked)
        all_done: [batch] — True if all positions resolved
        num_transferred: [batch] — number of newly committed positions
    """
    probs = torch.softmax(logits.float(), dim=-1)
    max_probs, argmax_ids = probs.max(dim=-1)

    # Only masked positions can be updated
    is_masked = input_draft == mask_token_id
    confident = max_probs > threshold

    # Transfer: masked AND confident
    transfer = is_masked & confident

    # Update canvas: transfer argmax for confident masked positions
    canvas = input_draft.clone()
    canvas[transfer] = argmax_ids[transfer]

    # If no positions transferred via threshold, transfer top-1 masked
    # by confidence (minimum 1 per request to guarantee progress)
    no_transfer = ~transfer.any(dim=-1)  # [batch]
    if no_transfer.any():
        for b in no_transfer.nonzero(as_tuple=True)[0]:
            masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
            if len(masked_pos) > 0:
                best = masked_pos[max_probs[b, masked_pos].argmax()]
                canvas[b, best] = argmax_ids[b, best]

    # Check convergence
    all_done = (canvas != mask_token_id).all(dim=-1)  # [batch]
    num_transferred = (canvas != input_draft).sum(dim=-1)  # [batch]

    return canvas, all_done, num_transferred
