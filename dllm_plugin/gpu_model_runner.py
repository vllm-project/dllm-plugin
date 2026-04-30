# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dLLM GPU model runner (v2): extends :class:`HookedGPUModelRunner` with block remask.

Phase one remains stock ``execute_model`` → ``None``; phase two uses inherited
``sample_tokens`` from :mod:`~dllm_plugin.vllm_gpu_model_runner_fork`, which calls
overridable hooks then :meth:`sample`. Only dLLM block batches replace AR /
rejection sampling inside :meth:`sample`.

**Target vLLM:** ``0.20.x`` (fork baseline tag ``v0.20.0``). See
:class:`~dllm_plugin.vllm_gpu_model_runner_fork.HookedGPUModelRunner` for upstream
parity and rebase notes.

See ``docs/DESIGN_MVP.md`` for the two-phase contract.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.v1.core.sched.output import GrammarOutput as GrammarOutputType
from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutputType
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from dllm_plugin.config import (
    DLLM_MOCK_STACK_MODEL_ID,
    DRAFT_SIZE,
    LLADA2_ARCHITECTURE_NAME,
)
from dllm_plugin.grammar_utils import (
    apply_packed_bitmask_inplace_logits_row,
    grammar_extra_transfer_slots,
)
from dllm_plugin.vllm_gpu_model_runner_fork import HookedGPUModelRunner
from dllm_plugin.worker import DllmWorker


def dllm_architecture_match(vllm_config: Any) -> bool:
    hf = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    if hf is None:
        return False
    archs = getattr(hf, "architectures", ()) or ()
    if isinstance(archs, str):
        archs = (archs,)
    names = {str(a) for a in archs}
    dllm_names = {DLLM_MOCK_STACK_MODEL_ID, LLADA2_ARCHITECTURE_NAME}
    return bool(names.intersection(dllm_names))


class DllmGPUModelRunner(HookedGPUModelRunner):
    """v2 GPU model runner with dLLM block sampling in phase two."""

    def __init__(self, vllm_config: Any, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        #: Width for sampled-token tensor rows (rejection / post_update layout).
        self._dllm_slot_width = max(self.num_speculative_steps + 1, DRAFT_SIZE)
        self._dllm_helper = DllmWorker(require_v2_model_runner=True)
        self._dllm_scheduled_spec_decode_tokens: dict[str, tuple[int, ...]] = {}
        self._dllm_so_frontier_flat_indices: dict[str, int] | None = None
        self._dllm_so_frontier_block_rows: dict[str, int | None] | None = None
        self._dllm_so_valid_prefix_lens: dict[str, int] | None = None
        self._dllm_pending_draft_ids: Any = None

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
        raw = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
        self._dllm_scheduled_spec_decode_tokens = {k: tuple(v) for k, v in raw.items()}
        self._dllm_so_frontier_flat_indices = getattr(
            scheduler_output, "dllm_so_frontier_flat_indices", None
        )
        self._dllm_so_frontier_block_rows = getattr(
            scheduler_output, "dllm_so_frontier_block_rows", None
        )
        self._dllm_so_valid_prefix_lens = getattr(
            scheduler_output, "dllm_so_valid_prefix_lens", None
        )

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutputType | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        if not (
            dllm_architecture_match(self.vllm_config)
            and input_batch.num_draft_tokens > 0
        ):
            return super().sample(hidden_states, input_batch, grammar_output)

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
            block_logits_tensor = logits[lo:hi]
            block_logits = self._tensor_block_to_rows(block_logits_tensor)

            input_draft = validate_runtime_input_draft(
                request_id=req_id,
                input_draft=list(
                    self._dllm_scheduled_spec_decode_tokens.get(req_id, ()),
                ),
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
            remasking_cfg = (
                {"grammar_extra_transfer": extra_transfer} if extra_transfer else None
            )

            step = run_block_contract_from_model_output(
                helper=self._dllm_helper,
                request_id=req_id,
                input_draft=input_draft,
                logits=block_logits,
                remasking_config=remasking_cfg,
            )
            committed = list(step.sampled_token_ids)
            nums.append(len(committed))
            for j, tok in enumerate(committed):
                if j < width:
                    sampled[i, j] = tok
            next_blocks.append(list(self._dllm_helper.take_draft_token_ids(step)))

        num_sampled = torch.tensor(nums, dtype=torch.int32, device=self.device)

        sampler_output = SamplerOutput(
            sampled_token_ids=sampled[:, :width],
            logprobs_tensors=None,
            num_nans=None,
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
    "HookedGPUModelRunner",
    "dllm_architecture_match",
]
