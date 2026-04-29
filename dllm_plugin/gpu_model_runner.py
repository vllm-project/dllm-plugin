# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dLLM-aligned GPU model runner (v2): phase-one ``execute_model`` → ``None``, phase-two
``sample_tokens(grammar_output)`` with remasking and draft handoff.

Keeps parity with stock :class:`~vllm.v1.worker.gpu.model_runner.GPUModelRunner` for
non-dLLM batches; on dLLM block decode (scheduled spec decode slots), overrides
:meth:`sample` to run block remasking instead of AR sampling + rejection.

See ``docs/DESIGN_MVP.md`` for the two-phase contract.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch

from dllm_plugin.config import (
    DLLM_MOCK_STACK_MODEL_ID,
    DRAFT_SIZE,
    LLADA2_ARCHITECTURE_NAME,
)
from dllm_plugin.grammar_utils import (
    apply_packed_bitmask_inplace_logits_row,
    grammar_extra_transfer_slots,
)
from dllm_plugin.worker import DllmWorker

try:
    from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
except ImportError:  # pragma: no cover - vLLM 0.14.x had no standalone helper

    def async_copy_to_gpu(
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Match newer ``vllm.v1.worker.gpu.buffer_utils.async_copy_to_gpu``."""

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        assert x.is_cpu

        if out is None:
            assert device is not None
            out = torch.empty_like(x, device=device)

        tmp = x.pin_memory()
        assert tmp is not x
        return out.copy_(tmp, non_blocking=True)


try:
    from vllm.sequence import IntermediateTensors as IntermediateTensorsType
    from vllm.v1.core.sched.output import GrammarOutput as GrammarOutputType
    from vllm.v1.core.sched.output import SchedulerOutput as SchedulerOutputType
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.input_batch import (
        InputBatch,
        combine_sampled_and_draft_tokens,
        expand_idx_mapping,
        get_num_sampled_and_rejected,
        prepare_pos_seq_lens,
        prepare_prefill_inputs,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.pp_utils import pp_broadcast, pp_receive
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    _VLLM = True
except ImportError:  # pragma: no cover
    _VLLM = False
    GPUModelRunner = object
    IntermediateTensorsType = Any
    GrammarOutputType = Any
    SchedulerOutputType = Any


def _dllm_architecture_match(vllm_config: Any) -> bool:
    hf = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    if hf is None:
        return False
    archs = getattr(hf, "architectures", ()) or ()
    if isinstance(archs, str):
        archs = (archs,)
    names = {str(a) for a in archs}
    dllm_names = {DLLM_MOCK_STACK_MODEL_ID, LLADA2_ARCHITECTURE_NAME}
    return bool(names.intersection(dllm_names))


class DllmGPUModelRunner(GPUModelRunner):
    """v2 GPU model runner with dLLM block sampling in phase two."""

    def __init__(self, vllm_config: Any, device: torch.device) -> None:
        if not _VLLM:
            raise RuntimeError("DllmGPUModelRunner requires vLLM.")
        super().__init__(vllm_config, device)
        #: Width for sampled-token tensor rows (rejection / post_update layout).
        self._dllm_slot_width = max(self.num_speculative_steps + 1, DRAFT_SIZE)
        self._dllm_helper = DllmWorker(require_v2_model_runner=True)
        self._dllm_scheduled_spec_decode_tokens: dict[str, tuple[int, ...]] = {}
        self._dllm_so_frontier_flat_indices: dict[str, int] | None = None
        self._dllm_so_frontier_block_rows: dict[str, int | None] | None = None
        self._dllm_so_valid_prefix_lens: dict[str, int] | None = None
        self._dllm_pending_draft_ids: Any = None

    def _dllm_capture_scheduler_extras(
        self, scheduler_output: SchedulerOutputType
    ) -> None:
        if not _dllm_architecture_match(self.vllm_config):
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

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutputType,
        intermediate_tensors: IntermediateTensorsType | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensorsType | None:
        self._dllm_pending_draft_ids = None
        if not dummy_run:
            self._dllm_capture_scheduler_extras(scheduler_output)
        return super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
        )

    def prepare_inputs(
        self, scheduler_output: SchedulerOutputType, num_tokens_after_padding: int
    ) -> InputBatch:
        """Same as upstream, but ``expand_idx_mapping`` must cover full dLLM blocks.

        Stock code uses ``num_speculative_steps + 1`` as the triton block size; dLLM
        schedules ``DRAFT_SIZE`` logits rows per request without Eagle, so we take
        ``max(num_speculative_steps + 1, max_logits_per_request)``.
        """
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        req_ids = sorted(num_tokens_per_req, key=num_tokens_per_req.get)
        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not draft_tokens:
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            num_draft_tokens = np.array(
                [len(draft_tokens.get(req_id, ())) for req_id in req_ids],
                dtype=np.int32,
            )
            total_num_draft_tokens = int(num_draft_tokens.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            num_logits = num_draft_tokens + 1
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_logits_per_req = int(np.max(num_logits))
            max_expand_len = max(self.num_speculative_steps + 1, max_logits_per_req)
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
        query_start_loc_np = query_start_loc_np[: num_reqs + 1]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]

        if self.req_states.any_prefills(idx_mapping_np):
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]

        dcp_local_seq_lens = None
        if self.use_dcp:
            from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens

            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs]

        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
        )

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            dcp_local_seq_lens=dcp_local_seq_lens,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
        )

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutputType | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        if not (
            _dllm_architecture_match(self.vllm_config)
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

        if grammar_output is not None:
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

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutputType | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        # Keep logic aligned with vllm GPUModelRunner.sample_tokens (last synced:
        # upstream execute_model → None, sample_tokens applies grammar + sampling).
        if self.execute_model_state is None:
            return None
        (
            input_batch,
            model_inputs,
            attn_metadata,
            slot_mappings_by_layer,
            hidden_states,
            aux_hidden_states,
            kv_connector_output,
        ) = self.execute_model_state
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            sampled, num_sampled, num_rejected = pp_receive(
                input_batch.num_reqs,
                max_sample_len=self.num_speculative_steps + 1,
            )
            self.postprocess(input_batch, sampled, num_sampled, num_rejected)
            return None

        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.use_pp:
            pp_broadcast(sampler_output.sampled_token_ids, num_sampled, num_rejected)

        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
            self.req_states.prefill_len.np,
            self.req_states.num_computed_prefill_tokens,
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
        )
        _ao_params = inspect.signature(AsyncOutput.__init__).parameters
        _ao_kw: dict[str, Any] = {
            "model_runner_output": model_runner_output,
            "sampler_output": sampler_output,
            "num_sampled_tokens": num_sampled,
            "copy_stream": self.output_copy_stream,
            "copy_event": self.output_copy_event,
        }
        if "main_stream" in _ao_params:
            _ao_kw["main_stream"] = self.main_stream
        async_output = AsyncOutput(**_ao_kw)

        self.postprocess(
            input_batch, sampler_output.sampled_token_ids, num_sampled, num_rejected
        )

        dllm_block = (
            _dllm_architecture_match(self.vllm_config)
            and input_batch.num_draft_tokens > 0
        )
        if dllm_block:
            if self.use_async_scheduling:
                return async_output
            return async_output.get_output()

        if self.speculator is not None:
            draft_tokens = self.speculator.propose(
                input_batch,
                attn_metadata,
                slot_mappings_by_layer,
                hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                self.req_states.last_sampled_tokens,
                self.req_states.next_prefill_tokens,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
            )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            self.draft_tokens_handler.set_draft_tokens(input_batch, draft_tokens)

        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    def take_dllm_draft_token_ids(self) -> Any | None:
        """Pop draft blocks produced by dLLM remasking (phase two)."""

        out = self._dllm_pending_draft_ids
        self._dllm_pending_draft_ids = None
        return out


__all__ = ["DllmGPUModelRunner", "_dllm_architecture_match"]
