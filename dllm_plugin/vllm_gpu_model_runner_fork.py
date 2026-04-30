# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM **v0.20.x** :class:`~vllm.v1.worker.gpu.model_runner.GPUModelRunner` fork layer.

Tracks upstream ``GPUModelRunner.prepare_inputs`` and ``sample_tokens`` at tag
**v0.20.0** with **tiny hook points** so plugin subclasses avoid copying phase-two
orchestration.

Rebase baseline: https://github.com/vllm-project/vllm/tree/v0.20.0
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.eplb_utils import step_eplb_after
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.pp_utils import pp_broadcast, pp_receive
from vllm.v1.worker.gpu.sample.output import SamplerOutput

_gpu_mr_cls = cast(Any, GPUModelRunner)
_EXECUTE_MODEL_PARAM_NAMES = frozenset(
    inspect.signature(_gpu_mr_cls.execute_model).parameters,
)


class HookedGPUModelRunner(GPUModelRunner):
    """Runner aligned to v0.20 ``GPUModelRunner`` with fork hooks."""

    def get_expand_idx_mapping_block_size(self, max_logits_per_req: int) -> int:
        """Triton ``BLOCK_SIZE`` for :func:`~expand_idx_mapping`.

        Stock v0.20 uses ``num_speculative_steps + 1`` only; this fork also bounds
        below by ``max_logits_per_req`` so wide per-request logit rows stay valid.
        """

        return max(self.num_speculative_steps + 1, max_logits_per_req)

    def get_pp_receive_max_sample_len(self) -> int:
        """Column width for PP ``sampled_token_ids`` tensors."""

        return self.num_speculative_steps + 1

    def adapt_sampler_output_for_pp_broadcast(
        self,
        sampler_output: SamplerOutput,
    ) -> SamplerOutput:
        """Adjust ``SamplerOutput`` before ``pp_broadcast`` (default: no-op)."""

        return sampler_output

    def should_run_speculator_proposal_phase(self, input_batch: InputBatch) -> bool:
        """Whether to run Eagle multimodal gather + ``speculator.propose``."""

        return True

    def before_execute_model(
        self,
        scheduler_output: SchedulerOutput,
        *,
        dummy_run: bool,
    ) -> None:
        """Hook invoked at the start of :meth:`execute_model` (after guards)."""

        return None

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        **kwargs: Any,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        self.before_execute_model(scheduler_output, dummy_run=dummy_run)
        kw: dict[str, Any] = {
            "scheduler_output": scheduler_output,
            "intermediate_tensors": intermediate_tensors,
            "dummy_run": dummy_run,
            "skip_attn_for_dummy_run": skip_attn_for_dummy_run,
            "is_profile": is_profile,
        }
        for name, val in kwargs.items():
            if name in _EXECUTE_MODEL_PARAM_NAMES:
                kw[name] = val
        return super().execute_model(**kw)

    @torch.inference_mode()
    def prepare_inputs(
        self, scheduler_output: SchedulerOutput, batch_desc: BatchExecutionDescriptor
    ) -> InputBatch:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        num_tokens_after_padding = batch_desc.num_tokens
        assert num_tokens > 0
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(num_tokens_per_req, key=num_tokens_per_req.get)
        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not draft_tokens:
            # No draft token scheduled (common case).
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
            num_draft_tokens = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            total_num_draft_tokens = int(num_draft_tokens.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            num_logits = num_draft_tokens + 1
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_logits_per_req = int(np.max(num_logits))
            max_expand_len = self.get_expand_idx_mapping_block_size(max_logits_per_req)
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        # Get query_start_loc.
        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]

        # Get prefill tokens if any.
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

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

        dcp_local_seq_lens = None
        if self.use_dcp:
            # Prepare dcp local seq_lens.
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
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

        # CPU upper bound on seq_lens; padded entries left at zero.
        # Matches vLLM v0.20.0 ``GPUModelRunner.prepare_inputs`` / ``InputBatch``.
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            self.req_states.num_computed_tokens_np[idx_mapping_np],
            num_scheduled_tokens,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
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
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
        )

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        kv_connector_output = self.execute_model_state.kv_connector_output
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # Non-last PP rank: hidden_states is None because this rank produced
            # IntermediateTensors instead of final hidden states. Receive the
            # sampled tokens broadcast from the last rank and update local state.
            sampled, num_sampled, num_rejected = pp_receive(
                input_batch.num_reqs,
                max_sample_len=self.get_pp_receive_max_sample_len(),
            )
            self.postprocess(input_batch, sampled, num_sampled, num_rejected)
            return None

        # Last rank: sample tokens
        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        sampler_output = self.adapt_sampler_output_for_pp_broadcast(sampler_output)

        if self.use_pp:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            pp_broadcast(sampler_output.sampled_token_ids, num_sampled, num_rejected)

        assert self.prompt_logprobs_worker is not None
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

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
        )
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        run_spec_phase = (
            self.speculator is not None
            and self.should_run_speculator_proposal_phase(input_batch)
        )

        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
        if run_spec_phase and self.speculator.supports_mm_inputs:
            # Get cached multimodal embeddings for draft forward.
            # NOTE: This is done here because postprocess updates
            # num_computed_prefill_tokens.
            prefill_lens = self.req_states.prefill_len.np[input_batch.idx_mapping_np]
            computed_prefill_lens = self.req_states.num_computed_prefill_tokens[
                input_batch.idx_mapping_np
            ]
            mm_inputs = self.model_state.encoder_runner.gather_mm_embeddings(
                input_batch.req_ids,
                input_batch.num_tokens,
                input_batch.num_scheduled_tokens,
                input_batch.query_start_loc_np,
                prefill_lens,
                computed_prefill_lens + 1,  # +1 to consider the skew in eagle
            )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess(
            input_batch, sampler_output.sampled_token_ids, num_sampled, num_rejected
        )

        if run_spec_phase:
            assert self.sampler is not None
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
                mm_inputs=mm_inputs,
            )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            self.draft_tokens_handler.set_draft_tokens(input_batch, draft_tokens)

        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()


__all__ = ["HookedGPUModelRunner"]
