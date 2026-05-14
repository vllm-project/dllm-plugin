# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dLLM GPU model runner: extends vLLM's GPUModelRunner with block remasking.

Overrides ``before_execute_model`` to extract dLLM metadata from scheduler
output, and ``sample`` to implement block diffusion remasking instead of
AR / rejection sampling.

Requires the vLLM fork (dllm-fork branch) which provides hook methods on
GPUModelRunner for plugin subclass customization.

See ``docs/DESIGN_MVP.md`` for the two-phase contract.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from vllm.v1.core.sched.output import GrammarOutput as GrammarOutputType
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutputType
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from dllm_plugin.config import (
    DLLM_MOCK_STACK_MODEL_ID,
    DRAFT_SIZE,
    LLADA2_ARCHITECTURE_NAME,
    LLADA2_DEFAULT_MASK_TOKEN_ID,
    LLADA2_HF_ARCHITECTURE_NAME,
)
from dllm_plugin.forward_context import (
    _num_prefix_tokens_list_ctx,
    set_num_prefix_tokens_list,
)
from dllm_plugin.grammar_utils import (
    apply_packed_bitmask_inplace_logits_row,
    grammar_extra_transfer_slots,
)
from dllm_plugin.vllm_compat import VllmConfig
from dllm_plugin.vllm_types import VllmConfigProtocol
from dllm_plugin.worker import DllmWorker

logger = logging.getLogger(__name__)


def dllm_architecture_match(vllm_config: VllmConfig | VllmConfigProtocol) -> bool:
    """Check if vLLM config architecture matches dLLM models.

    Args:
        vllm_config: vLLM configuration object.

    Returns:
        True if architecture is LLaDA2 or dLLM mock model.
    """
    # Safely extract hf_config with explicit error handling
    try:
        hf = vllm_config.model_config.hf_config
    except AttributeError:
        # model_config or hf_config missing
        return False

    archs = getattr(hf, "architectures", ()) or ()
    if isinstance(archs, str):
        archs = (archs,)
    names = {str(a) for a in archs}
    dllm_names = {
        DLLM_MOCK_STACK_MODEL_ID,
        LLADA2_ARCHITECTURE_NAME,
        LLADA2_HF_ARCHITECTURE_NAME,
    }
    return bool(names.intersection(dllm_names))


class DllmGPUModelRunner(GPUModelRunner):
    """v2 GPU model runner with dLLM block sampling in phase two."""

    def __init__(self, vllm_config: Any, device: torch.device) -> None:
        super().__init__(vllm_config, device)

        # dLLM uses the spec-decode infrastructure for 32-token draft blocks.
        # Parent init sets num_speculative_steps=0 (no speculative_config).
        # Resize buffers to fit DRAFT_SIZE-token blocks.
        if dllm_architecture_match(vllm_config):
            self._resize_for_draft_blocks(DRAFT_SIZE)

        #: Width for sampled-token tensor rows (rejection / post_update layout).
        self._dllm_slot_width = max(self.num_speculative_steps + 1, DRAFT_SIZE)
        self._dllm_helper = DllmWorker(require_v2_model_runner=True)
        self._dllm_scheduled_spec_decode_tokens: dict[str, tuple[int, ...]] = {}
        self._dllm_so_frontier_flat_indices: dict[str, int] | None = None
        self._dllm_so_frontier_block_rows: dict[str, int | None] | None = None
        self._dllm_so_valid_prefix_lens: dict[str, int] | None = None
        self._dllm_num_prefix_tokens: dict[str, int] | None = None
        self._dllm_pending_draft_ids: Any = None
        self._dllm_denoise_step: dict[str, int] = {}
        self._dllm_initial_prompt_len: dict[str, int] = {}

    def _resize_for_draft_blocks(self, draft_size: int) -> None:
        """Resize spec-decode buffers to fit draft_size-token blocks.

        Parent GPUModelRunner allocates draft_tokens as [max_reqs, 0] when
        no speculative_config is set. dLLM reuses this buffer for 32-token
        blocks, so we resize it and all dependent objects.
        """
        from vllm.v1.worker.gpu.states import RequestState

        self.num_speculative_steps = draft_size
        self.uniform_decode_query_len = 1 + draft_size
        self.decode_query_len = draft_size + 1

        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=draft_size,
            vocab_size=self.vocab_size,
            device=self.device,
        )

        if self.is_last_pp_rank:
            from vllm.v1.worker.gpu.sample.sampler import Sampler
            from vllm.v1.worker.gpu.structured_outputs import (
                StructuredOutputsWorker,
            )

            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=draft_size + 1,
            )
            self.structured_outputs_worker = StructuredOutputsWorker(
                max_num_logits=self.max_num_reqs * (draft_size + 1),
                vocab_size=self.vocab_size,
                device=self.device,
            )

    def shutdown(self) -> None:
        """vLLM v1 worker calls this during engine teardown."""
        self._dllm_denoise_step.clear()
        self._dllm_initial_prompt_len.clear()
        self._dllm_scheduled_spec_decode_tokens.clear()

        parent = super()
        fn = getattr(parent, "shutdown", None)
        if callable(fn):
            fn()

    def prepare_inputs(self, scheduler_output, batch_desc):
        """Copy dLLM draft tokens into req_states.draft_tokens before the
        combine_sampled_and_draft_tokens kernel reads them.

        At this point add_requests() has already run, so req_id_to_index
        is populated for all scheduled requests.
        """
        if self._dllm_scheduled_spec_decode_tokens:
            for req_id, tokens in self._dllm_scheduled_spec_decode_tokens.items():
                req_idx = self.req_states.req_id_to_index.get(req_id)
                if req_idx is not None and len(tokens) > 0:
                    n = min(len(tokens), self.req_states.draft_tokens.shape[1])
                    self.req_states.draft_tokens[req_idx, :n] = torch.tensor(
                        tokens[:n],
                        dtype=self.req_states.draft_tokens.dtype,
                        device=self.req_states.draft_tokens.device,
                    )
        return super().prepare_inputs(scheduler_output, batch_desc)

    def get_expand_idx_mapping_block_size(self, max_logits_per_req: int) -> int:
        n = super().get_expand_idx_mapping_block_size(max_logits_per_req)
        if dllm_architecture_match(self.vllm_config):
            return max(n, DRAFT_SIZE)
        return n

    def get_pp_receive_max_sample_len(self) -> int:
        if dllm_architecture_match(self.vllm_config):
            return self._dllm_slot_width
        return super().get_pp_receive_max_sample_len()

    def adapt_sampler_output_for_pp_broadcast(
        self,
        sampler_output: SamplerOutput,
    ) -> SamplerOutput:
        if not self.use_pp:
            return sampler_output
        need = self.get_pp_receive_max_sample_len()
        cur = sampler_output.sampled_token_ids
        if cur.shape[1] >= need:
            return sampler_output
        padded = torch.full(
            (cur.shape[0], need),
            -1,
            dtype=torch.int64,
            device=cur.device,
        )
        padded[:, : cur.shape[1]] = cur
        return SamplerOutput(
            sampled_token_ids=padded,
            logprobs_tensors=sampler_output.logprobs_tensors,
            num_nans=sampler_output.num_nans,
        )

    def should_run_speculator_proposal_phase(self, input_batch: InputBatch) -> bool:
        if (
            dllm_architecture_match(self.vllm_config)
            and input_batch.num_draft_tokens > 0
        ):
            return False
        return super().should_run_speculator_proposal_phase(input_batch)

    def before_execute_model(
        self,
        scheduler_output: SchedulerOutputType,
        *,
        dummy_run: bool,
    ) -> None:
        self._dllm_pending_draft_ids = None
        if dummy_run:
            return
        if not dllm_architecture_match(self.vllm_config):
            return

        # Store scheduled drafts from scheduler
        raw = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
        self._dllm_scheduled_spec_decode_tokens = {k: tuple(v) for k, v in raw.items()}

        # Clean up per-request state for requests no longer scheduled
        # (aborted, completed, or preempted).
        active = set(getattr(scheduler_output, "num_scheduled_tokens", {}).keys())
        for stale in set(self._dllm_denoise_step) - active:
            self._dllm_denoise_step.pop(stale, None)
            self._dllm_initial_prompt_len.pop(stale, None)

        # Continue with existing metadata extraction
        self._dllm_so_frontier_flat_indices = getattr(
            scheduler_output, "dllm_so_frontier_flat_indices", None
        )
        self._dllm_so_frontier_block_rows = getattr(
            scheduler_output, "dllm_so_frontier_block_rows", None
        )
        self._dllm_so_valid_prefix_lens = getattr(
            scheduler_output, "dllm_so_valid_prefix_lens", None
        )
        self._dllm_num_prefix_tokens = getattr(
            scheduler_output, "dllm_num_prefix_tokens", None
        )

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutputType | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        is_dllm = dllm_architecture_match(self.vllm_config)
        has_dllm_drafts = (
            is_dllm
            and input_batch.num_draft_tokens > 0
            and self._dllm_scheduled_spec_decode_tokens
        )
        if not has_dllm_drafts:
            # No dllm-scheduled drafts (warmup, prefill, or non-dllm).
            # Force num_draft_tokens=0 for parent since we have no rejection_sampler.
            orig_num_draft = input_batch.num_draft_tokens
            input_batch.num_draft_tokens = 0
            try:
                result = super().sample(hidden_states, input_batch, grammar_output)
            finally:
                input_batch.num_draft_tokens = orig_num_draft

            # For dLLM prefill steps (no drafts), suppress the AR token —
            # the first block's committed output will contain the real tokens.
            if is_dllm and not self._dllm_scheduled_spec_decode_tokens:
                so = result[0]
                so.sampled_token_ids.zero_()
                if so.num_sampled is not None:
                    so.num_sampled.zero_()

            return result

        # Late import avoids circular import with runtime_worker.
        from dllm_plugin.runtime_worker import (
            run_block_contract_from_model_output,
            validate_runtime_input_draft,
        )

        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # Two-stage grammar: vLLM applies the full batch bitmask on GPU; we then
        # refine the frontier row on CPU-float logits for dLLM remask (first invalid
        # position per scheduler metadata — consistent with packed bitmask layout).
        if grammar_output is not None:
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        go = grammar_output
        flat_indices = self._dllm_so_frontier_flat_indices
        block_rows = self._dllm_so_frontier_block_rows
        prefix_lens = self._dllm_so_valid_prefix_lens

        req_ids = input_batch.req_ids
        cu = input_batch.cu_num_logits_np
        width = self._dllm_slot_width
        sampled = torch.full(
            (input_batch.num_reqs, width),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        nums: list[int] = []
        next_blocks: list[list[int]] = []

        for i, req_id in enumerate(req_ids):
            lo, hi = int(cu[i]), int(cu[i + 1])
            # Extract logits for this request
            # cu_num_logits includes primary token + draft tokens (e.g., 1 + 32 = 33)
            # dLLM remasking only processes draft tokens (32), so skip first row
            all_logits = logits[lo:hi]
            if all_logits.shape[0] > self._dllm_helper.draft_size:
                # Skip primary token logits (first row), keep only draft token logits
                block_logits_tensor = all_logits[1:]
            else:
                # Fallback for edge cases (shouldn't happen with spec decode)
                block_logits_tensor = all_logits

            # Validate logits before remasking (debug mode only - CPU-GPU sync overhead)
            if os.getenv("VLLM_DEBUG", "0") == "1":
                if torch.isnan(block_logits_tensor).any():
                    nan_count = torch.isnan(block_logits_tensor).sum().item()
                    raise ValueError(
                        f"Logits contain {nan_count} NaN value(s) for "
                        f"request {req_id}. Shape: {block_logits_tensor.shape}"
                    )
                if torch.isinf(block_logits_tensor).any():
                    inf_count = torch.isinf(block_logits_tensor).sum().item()
                    raise ValueError(
                        f"Logits contain {inf_count} inf value(s) for "
                        f"request {req_id}. Shape: {block_logits_tensor.shape}"
                    )

            block_logits = self._tensor_block_to_rows(block_logits_tensor)

            # Use scheduled draft (includes first block from scheduler for new requests)
            scheduled_draft = self._dllm_scheduled_spec_decode_tokens.get(req_id, ())
            input_draft = validate_runtime_input_draft(
                request_id=req_id,
                input_draft=list(scheduled_draft),
                draft_size=self._dllm_helper.draft_size,
            )

            so_reqs = getattr(go, "structured_output_request_ids", None) if go else None
            if (
                go is not None
                and flat_indices is not None
                and block_rows is not None
                and so_reqs is not None
                and req_id in so_reqs
            ):
                br = block_rows.get(req_id)
                fi = flat_indices.get(req_id)
                if br is not None and fi is not None:
                    row_bm = go.grammar_bitmask[int(fi)]
                    apply_packed_bitmask_inplace_logits_row(block_logits[br], row_bm)

            extra_transfer = 0
            if prefix_lens is not None and req_id in prefix_lens:
                extra_transfer = grammar_extra_transfer_slots(
                    draft_tokens=input_draft,
                    valid_prefix_len=prefix_lens[req_id],
                )
            step_idx = self._dllm_denoise_step.get(req_id, 0)
            remasking_cfg: dict[str, int | float] = {
                "denoise_step_index": step_idx,
            }
            if extra_transfer:
                remasking_cfg["grammar_extra_transfer"] = extra_transfer

            step = run_block_contract_from_model_output(
                helper=self._dllm_helper,
                request_id=req_id,
                input_draft=input_draft,
                logits=block_logits,
                remasking_config=remasking_cfg,
            )
            committed = list(step.sampled_token_ids)

            # On first denoising step for a block, record how many leading
            # positions are prompt tokens (not masks) so we can strip them
            # at commit time. Assumes prompt tokens form a contiguous prefix
            # before mask tokens. Safe because LLADA2_DEFAULT_MASK_TOKEN_ID
            # (156895) is far above typical prompt token IDs.
            if req_id not in self._dllm_initial_prompt_len:
                n_prompt = 0
                for t in scheduled_draft:
                    if t != LLADA2_DEFAULT_MASK_TOKEN_ID:
                        n_prompt += 1
                    else:
                        break
                self._dllm_initial_prompt_len[req_id] = n_prompt

            # Strip prompt prefix from committed block
            if committed:
                n_prompt = self._dllm_initial_prompt_len.get(req_id, 0)
                if n_prompt > 0:
                    committed = committed[n_prompt:]

            # Track denoise step and check max-iteration guard BEFORE
            # appending to next_blocks, so force-commit uses the right block.
            max_denoise_iters = 2 * DRAFT_SIZE
            force_committed = False
            if len(step.sampled_token_ids) == 0:
                new_step = step_idx + 1
                if new_step >= max_denoise_iters:
                    logger.warning(
                        "req %s: denoise hit max iterations (%d), "
                        "force-committing block",
                        req_id,
                        max_denoise_iters,
                    )
                    committed = [
                        t
                        for t in step.next_input_block
                        if t != LLADA2_DEFAULT_MASK_TOKEN_ID
                    ]
                    force_committed = True
                    self._dllm_denoise_step.pop(req_id, None)
                    self._dllm_initial_prompt_len.pop(req_id, None)
                else:
                    self._dllm_denoise_step[req_id] = new_step
            else:
                self._dllm_denoise_step.pop(req_id, None)
                self._dllm_initial_prompt_len.pop(req_id, None)

            nums.append(len(committed))
            for j, tok in enumerate(committed):
                if j < width:
                    sampled[i, j] = tok

            if force_committed:
                # Force-commit: next block is all-mask (new block)
                next_blocks.append([LLADA2_DEFAULT_MASK_TOKEN_ID] * DRAFT_SIZE)
            else:
                next_blocks.append(list(self._dllm_helper.take_draft_token_ids(step)))

        num_sampled = torch.tensor(nums, dtype=torch.int32, device=self.device)

        sampler_output = SamplerOutput(
            sampled_token_ids=sampled[:, :width],
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )

        from vllm.v1.outputs import DraftTokenIds

        self._dllm_pending_draft_ids = DraftTokenIds(
            req_ids=list(req_ids),
            draft_token_ids=next_blocks,
        )
        return sampler_output, num_sampled, num_rejected

    @staticmethod
    def _tensor_block_to_rows(block: torch.Tensor) -> list[list[float]]:
        """Flatten GPU logits rows to Python floats for remask policy."""

        b = block.float().detach().cpu()
        return [row.tolist() for row in b]

    def execute_model(
        self,
        scheduler_output: SchedulerOutputType,
        intermediate_tensors: Any | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Override to inject num_prefix_tokens context for virtual batch attention.

        Sets context variable for chunked block attention before calling parent's
        execute_model (Phase 7 virtual batch decomposition).

        Note: before_execute_model() is called by the parent's execute_model(),
        so we don't call it explicitly here.
        """
        # Extract num_prefix_tokens directly from scheduler_output (Phase 7.1)
        num_prefix_tokens_list = None
        if dllm_architecture_match(self.vllm_config) and not dummy_run:
            # Get dllm_num_prefix_tokens dict from scheduler output
            dllm_npt = getattr(scheduler_output, "dllm_num_prefix_tokens", None)
            dllm_num_prefix_tokens = dllm_npt

            if dllm_num_prefix_tokens and scheduler_output.num_scheduled_tokens:
                # Build list preserving order of scheduled requests
                req_ids = list(scheduler_output.num_scheduled_tokens.keys())
                num_prefix_tokens_list = [
                    dllm_num_prefix_tokens.get(req_id, 0) for req_id in req_ids
                ]

        # Set context for chunked block attention (Strategy 2)
        # This allows LLaDA2BlockAttention to access num_prefix_tokens_list
        # without modifying all layer signatures
        token = set_num_prefix_tokens_list(num_prefix_tokens_list)
        try:
            return super().execute_model(
                scheduler_output=scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
                **kwargs,
            )
        finally:
            # Reset context after forward pass
            _num_prefix_tokens_list_ctx.reset(token)

    def take_dllm_draft_token_ids(self) -> Any | None:
        """Pop draft blocks produced by dLLM remasking (phase two).

        Named distinctly from upstream runner ``take_draft_token_ids`` so dLLM block
        handoff does not collide with Eagle/spec-decoder drafts; the worker delegates
        from its ``take_draft_token_ids`` (issue #10 — intentional naming deviation).
        """
        out = self._dllm_pending_draft_ids
        self._dllm_pending_draft_ids = None
        return out


__all__ = [
    "DllmGPUModelRunner",
    "GPUModelRunner",
    "dllm_architecture_match",
]
