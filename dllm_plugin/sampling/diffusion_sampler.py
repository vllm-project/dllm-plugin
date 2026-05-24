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
    is_masked = is_masked & (x0 != mask_token_id)

    x0 = torch.where(is_masked, x0, input_draft)
    neg_inf = torch.tensor(-torch.inf, device=logits.device)
    confidence = torch.where(is_masked, x0_p, neg_inf)

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

    Per-request denoising state lives in GPU tensors on the ModelState
    (``_denoise_step_t``, ``_kv_refresh_t``, ``_prompt_len_t``,
    ``_draft_block``). Convergence checking is fully vectorized on GPU.
    The only CPU touch is a single ``tolist()`` at step end to build
    ``DraftTokenIds`` for the scheduler interface.
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

        try:
            from dllm_plugin.sampling.triton_kernels import batched_remask_triton

            self._remask_fn = batched_remask_triton
        except ImportError:
            self._remask_fn = batched_remask

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
        """Prefill/bootstrap: initialize draft block for each request."""
        from vllm.v1.outputs import DraftTokenIds
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        ms = self.model_state
        mask_id = self._mask_id
        next_blocks: list[list[int]] = []
        for req_id in input_batch.req_ids:
            tail = ms._prompt_tail_ids.get(req_id, [])
            n_masks = self._draft_size - len(tail)
            block = list(tail) + [mask_id] * n_masks
            next_blocks.append(block)
            # Also write to the persistent draft block tensor
            slot = ms._slot_map.get(req_id)
            if slot is not None:
                ms._draft_block[slot] = torch.tensor(
                    block, dtype=ms._draft_block.dtype, device=self.device
                )
                ms._draft_block_valid[slot] = True

        ms._pending_draft_ids = DraftTokenIds(
            req_ids=list(input_batch.req_ids),
            draft_token_ids=next_blocks,
        )

        num_reqs = input_batch.num_reqs
        width = self._slot_width
        sampled = torch.full(
            (num_reqs, width), -1, dtype=torch.int64, device=self.device
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.zeros(num_reqs, dtype=torch.int32, device=self.device),
        )

    def _denoise(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        """Denoising step with GPU-resident convergence checking."""
        from vllm.v1.outputs import DraftTokenIds
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        ms = self.model_state
        req_ids = input_batch.req_ids
        num_reqs = input_batch.num_reqs
        cu = input_batch.cu_num_logits_np
        mask_id = self._mask_id
        width = self._slot_width

        # Map batch indices to slot indices
        all_slots = [ms._slot_map[rid] for rid in req_ids]

        # Split into decode requests (with draft logits) and prefill
        # requests (no draft logits). Only decode requests get denoised.
        decode_indices = []
        for i in range(num_reqs):
            lo, hi = int(cu[i]), int(cu[i + 1])
            if hi - lo == self._draft_size:
                decode_indices.append(i)

        if not decode_indices:
            # All requests are prefill — return base sampler output
            return self._base_sampler(logits, input_batch)

        decode_slots = torch.tensor(
            [all_slots[i] for i in decode_indices],
            dtype=torch.long,
            device=self.device,
        )
        slots = decode_slots

        block_logits_list = []
        for i in decode_indices:
            lo, hi = int(cu[i]), int(cu[i + 1])
            block_logits_list.append(logits[lo:hi])

        batch_logits = torch.stack(block_logits_list)
        batch_draft = ms._draft_block[slots]

        # Lazy init of prompt_len from first denoising step
        for idx, _i in enumerate(decode_indices):
            slot = slots[idx].item()
            if ms._prompt_len_t[slot] == 0 and ms._draft_block_valid[slot]:
                draft = batch_draft[idx]
                leading = (draft != mask_id).long()
                first_mask = (leading == 0).long().argmax().item()
                if first_mask == 0 and draft[0] == mask_id:
                    ms._prompt_len_t[slot] = 0
                else:
                    ms._prompt_len_t[slot] = first_mask

        updated, all_done, _ = self._remask_fn(
            logits=batch_logits,
            input_draft=batch_draft,
            mask_token_id=mask_id,
            threshold=self._threshold,
        )

        # --- Vectorized convergence on GPU ---
        kv_done = ms._kv_refresh_t[slots]
        needs_refresh = all_done & ~kv_done
        ms._kv_refresh_t[slots] = kv_done | all_done
        ready = all_done & kv_done

        steps = ms._denoise_step_t[slots]
        force_commit = ((steps + 1) >= self._max_denoise_iters) & ~all_done
        commit = ready | force_commit

        # Increment step for requests that continue denoising
        continues = ~commit & ~needs_refresh
        ms._denoise_step_t[slots] = torch.where(
            continues, steps + 1, ms._denoise_step_t[slots]
        )

        # Compute num_sampled on GPU
        prompt_lens = ms._prompt_len_t[slots]
        non_mask_counts = (updated != mask_id).sum(dim=-1).int()
        committed_counts = torch.where(
            ready,
            self._draft_size - prompt_lens,
            torch.where(force_commit, non_mask_counts, torch.zeros_like(prompt_lens)),
        )
        zeros = torch.zeros_like(committed_counts)
        num_sampled = torch.where(commit, committed_counts, zeros)

        # Build sampled output — map decode-local indices to batch indices
        num_decode = len(decode_indices)
        sampled = torch.full(
            (num_reqs, width), -1, dtype=torch.int64, device=self.device
        )
        full_num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)

        if commit.any().item():
            for di in range(num_decode):
                if not commit[di].item():
                    continue
                bi = decode_indices[di]
                if ready[di].item():
                    n_prompt = prompt_lens[di].item()
                    row = updated[di, n_prompt:]
                    n = row.shape[0]
                    sampled[bi, :n] = row
                else:
                    row = updated[di]
                    non_mask = row[row != mask_id]
                    n = non_mask.shape[0]
                    sampled[bi, :n] = non_mask

            commit_slots = slots[commit]
            ms._denoise_step_t[commit_slots] = 0
            ms._kv_refresh_t[commit_slots] = False
            ms._prompt_len_t[commit_slots] = 0

        # Write num_sampled for decode requests at their batch positions
        for di in range(num_decode):
            bi = decode_indices[di]
            full_num_sampled[bi] = num_sampled[di]

        # Update persistent draft block for decode requests
        ms._draft_block[slots] = torch.where(
            commit.unsqueeze(-1),
            torch.full_like(updated, mask_id),
            updated,
        )

        # Build DraftTokenIds for ALL requests (decode get updated
        # drafts, prefill get current draft_block state)
        all_slots_t = torch.tensor(all_slots, dtype=torch.long, device=self.device)
        next_blocks = ms._draft_block[all_slots_t].cpu().tolist()

        ms._pending_draft_ids = DraftTokenIds(
            req_ids=list(req_ids),
            draft_token_ids=next_blocks,
        )

        return SamplerOutput(
            sampled_token_ids=sampled[:, :width],
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=full_num_sampled,
        )
