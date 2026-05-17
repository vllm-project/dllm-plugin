# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion sampler for block denoising models.

Replaces the stock vLLM Sampler for diffusion models. Called through
the normal self.sampler(logits, input_batch) path after custom_sampler()
returns (DiffusionSampler, None) at model load time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

logger = logging.getLogger(__name__)


def batched_remask(
    logits: torch.Tensor,
    input_draft: torch.Tensor,
    mask_token_id: int,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched remasking on GPU.

    Args:
        logits: [batch, block_size, vocab_size] — model output logits
        input_draft: [batch, block_size] — current draft token IDs
        mask_token_id: token ID for masked positions
        threshold: softmax probability threshold for committing

    Returns:
        draft: [batch, block_size] — updated draft (committed + masked)
        all_done: [batch] — True if all positions resolved
        num_transferred: [batch] — number of newly committed positions
    """
    probs = torch.softmax(logits.float(), dim=-1)
    x0 = torch.argmax(logits.float(), dim=-1)
    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

    is_masked = input_draft == mask_token_id
    # Skip positions where the model predicts mask_id (matches dInfer rm_mask)
    is_masked = is_masked & (x0 != mask_token_id)

    x0 = torch.where(is_masked, x0, input_draft)
    neg_inf = torch.tensor(-torch.inf, device=logits.device)
    confidence = torch.where(is_masked, x0_p, neg_inf)

    # Adaptive threshold: commit all positions above threshold, or if none
    # exceed it, commit all positions within 1e-5 of the max (matches dInfer)
    actual_threshold = (
        (torch.max(confidence, dim=-1)[0] - 1e-5).clamp(-1000, threshold).unsqueeze(-1)
    )
    transfer = confidence >= actual_threshold

    draft = input_draft.clone()
    draft[transfer] = x0[transfer]

    all_done = (draft != mask_token_id).all(dim=-1)
    num_transferred = (draft != input_draft).sum(dim=-1)

    return draft, all_done, num_transferred


class DiffusionSampler:
    """Block diffusion sampler replacing vLLM's stock Sampler.

    Created at model load time via ModelState.custom_sampler().
    Called through the normal self.sampler(logits, input_batch) path.

    Manages per-request denoising state:
    - _denoise_step: current iteration count per request
    - _initial_prompt_len: prompt prefix length to strip at commit
    - _scheduled_spec_decode_tokens: current draft block per request
    - _pending_draft_ids: next-step drafts for take_draft_token_ids()
    """

    def __init__(
        self,
        base_sampler: Any,
        model_state: Any,
        device: torch.device,
        mask_token_id: int,
        draft_size: int,
        threshold: float,
        max_denoise_iters: int,
        slot_width: int,
    ) -> None:
        self._base_sampler = base_sampler
        self.model_state = model_state
        self.device = device
        self._mask_id = mask_token_id
        self._draft_size = draft_size
        self._threshold = threshold
        self._max_denoise_iters = max_denoise_iters
        self._slot_width = slot_width

    def __getattr__(self, name: str) -> Any:
        try:
            base = object.__getattribute__(self, "_base_sampler")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(base, name)

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:

        ms = self.model_state
        has_drafts = bool(ms._scheduled_spec_decode_tokens)

        if not has_drafts:
            return self._bootstrap(logits, input_batch)

        return self._denoise(logits, input_batch)

    def _bootstrap(
        self, logits: torch.Tensor, input_batch: InputBatch
    ) -> SamplerOutput:
        """Prefill/bootstrap: initialize draft for each request.

        Preserves prompt_tail tokens at the start of the draft block
        so the model sees [prompt_tail + masks] rather than [masks].
        This matches dInfer's BlockDiffusionLLM behavior.
        """
        from vllm.v1.outputs import DraftTokenIds
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        ms = self.model_state
        mask_id = self._mask_id
        next_blocks: list[list[int]] = []
        for req_id in input_batch.req_ids:
            tail = ms._prompt_tail_ids.get(req_id, [])
            n_masks = self._draft_size - len(tail)
            next_blocks.append(list(tail) + [mask_id] * n_masks)

        ms._pending_draft_ids = DraftTokenIds(
            req_ids=list(input_batch.req_ids),
            draft_token_ids=next_blocks,
        )

        num_reqs = input_batch.num_reqs
        width = self._slot_width
        sampled = torch.full(
            (num_reqs, width), -1, dtype=torch.int64, device=self.device
        )

        # Return num_sampled=1 so that the post_update kernel advances
        # num_computed_tokens by the full query_len (query_len - 0 rejected).
        # With num_sampled=0, the kernel computes num_rejected = num_logits - 0
        # = 1, then advances by query_len - 1, causing a position off-by-one.
        base_output = self._base_sampler(logits, input_batch)
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=base_output.num_sampled,
        )

    def _denoise(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        """Denoising step: remask and check convergence."""
        from vllm.v1.outputs import DraftTokenIds
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        ms = self.model_state
        req_ids = input_batch.req_ids
        num_reqs = input_batch.num_reqs
        cu = input_batch.cu_num_logits_np
        mask_id = self._mask_id
        width = self._slot_width

        block_logits_list = []
        draft_tensors = []
        for i in range(num_reqs):
            lo, hi = int(cu[i]), int(cu[i + 1])
            block_logits_list.append(logits[lo:hi])

            req_id = req_ids[i]
            draft = ms._scheduled_spec_decode_tokens.get(req_id, ())
            if len(draft) < self._draft_size:
                draft = tuple(draft) + (mask_id,) * (self._draft_size - len(draft))
            draft_tensors.append(
                torch.tensor(
                    draft[: self._draft_size], dtype=torch.int64, device=self.device
                )
            )

            if req_id not in ms._initial_prompt_len:
                n_prompt = 0
                for t in draft:
                    if t != mask_id:
                        n_prompt += 1
                    else:
                        break
                ms._initial_prompt_len[req_id] = n_prompt

        batch_logits = torch.stack(block_logits_list)
        batch_draft = torch.stack(draft_tensors)

        updated, all_done, _ = batched_remask(
            logits=batch_logits,
            input_draft=batch_draft,
            mask_token_id=mask_id,
            threshold=self._threshold,
        )

        sampled = torch.full(
            (num_reqs, width), -1, dtype=torch.int64, device=self.device
        )
        nums: list[int] = []
        next_blocks: list[list[int]] = []

        all_done_cpu = all_done.cpu()
        updated_cpu = updated.cpu()

        for i in range(num_reqs):
            req_id = req_ids[i]
            done = bool(all_done_cpu[i])
            step_idx = ms._denoise_step.get(req_id, 0)

            if done:
                if not ms._kv_refresh_done.get(req_id, False):
                    # All masks resolved but KV cache still reflects the
                    # previous draft state. Return Commit-0 with the final
                    # block to force one more forward pass that refreshes
                    # the KV cache, matching dInfer's cross-block update.
                    ms._kv_refresh_done[req_id] = True
                    nums.append(0)
                    next_blocks.append(updated_cpu[i].tolist())
                    continue

                committed = updated_cpu[i]
                n_prompt = ms._initial_prompt_len.get(req_id, 0)
                if n_prompt > 0:
                    committed = committed[n_prompt:]
                n = committed.shape[0]
                sampled[i, :n] = committed.to(self.device)
                nums.append(n)
                next_blocks.append([mask_id] * self._draft_size)
                ms._denoise_step.pop(req_id, None)
                ms._initial_prompt_len.pop(req_id, None)
                ms._kv_refresh_done.pop(req_id, None)
            else:
                new_step = step_idx + 1
                if new_step >= self._max_denoise_iters:
                    logger.warning(
                        "req %s: denoise hit max iterations (%d)",
                        req_id,
                        self._max_denoise_iters,
                    )
                    row = updated_cpu[i]
                    non_mask = row[row != mask_id]
                    n = non_mask.shape[0]
                    sampled[i, :n] = non_mask.to(self.device)
                    nums.append(n)
                    next_blocks.append([mask_id] * self._draft_size)
                    ms._denoise_step.pop(req_id, None)
                    ms._initial_prompt_len.pop(req_id, None)
                else:
                    nums.append(0)
                    next_blocks.append(updated_cpu[i].tolist())
                    ms._denoise_step[req_id] = new_step

        num_sampled = torch.tensor(nums, dtype=torch.int32, device=self.device)

        ms._pending_draft_ids = DraftTokenIds(
            req_ids=list(req_ids),
            draft_token_ids=next_blocks,
        )

        return SamplerOutput(
            sampled_token_ids=sampled[:, :width],
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
        )
