# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LLaDA2 ModelState for MRV2 integration.

Implements the ModelState interface for block diffusion models.
Handles non-causal attention, per-request denoising state, draft token
management, and batched GPU diffusion sampling — replacing the
DllmGPUModelRunner subclass.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.tasks import GenerationTask  # Literal type alias
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from dllm_plugin.config import (
    DRAFT_SIZE,
    LLADA2_DEFAULT_MASK_TOKEN_ID,
)

logger = logging.getLogger("vllm.dllm_plugin.llada2_model_state")


class LLaDA2ModelState(ModelState):
    """ModelState for LLaDA2 block diffusion models.

    Replaces DllmGPUModelRunner — all dLLM logic lives here via
    ModelState composition hooks (prepare_attn, custom_sample,
    before_step, take_draft_token_ids, etc.).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.max_model_len = self.model_config.max_model_len
        self.device = device

        diff_cfg = getattr(vllm_config, "diffusion_config", None)
        self._mask_id = (
            diff_cfg.mask_token_id if diff_cfg else LLADA2_DEFAULT_MASK_TOKEN_ID
        )
        self._draft_size = diff_cfg.canvas_length if diff_cfg else DRAFT_SIZE
        self._threshold = diff_cfg.commit_threshold if diff_cfg else 0.9
        self._max_denoise_iters = (
            diff_cfg.max_denoise_steps if diff_cfg else 2 * DRAFT_SIZE
        )
        self._slot_width = max(
            (diff_cfg.num_speculative_tokens if diff_cfg else DRAFT_SIZE) + 1,
            self._draft_size,
        )

        self._denoise_step: dict[str, int] = {}
        self._initial_prompt_len: dict[str, int] = {}
        self._scheduled_spec_decode_tokens: dict[str, tuple[int, ...]] = {}
        self._pending_draft_ids: DraftTokenIds | None = None

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        return ("generate",)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        return None

    def remove_request(self, req_id: str) -> None:
        self._denoise_step.pop(req_id, None)
        self._initial_prompt_len.pop(req_id, None)

    def apply_staged_writes(self) -> None:
        return None

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        return None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        if self._scheduled_spec_decode_tokens:
            for req_id, tokens in self._scheduled_spec_decode_tokens.items():
                req_idx = req_states.req_id_to_index.get(req_id)
                if req_idx is not None and len(tokens) > 0:
                    n = min(len(tokens), req_states.draft_tokens.shape[1])
                    req_states.draft_tokens[req_idx, :n] = torch.tensor(
                        tokens[:n],
                        dtype=req_states.draft_tokens.dtype,
                        device=req_states.draft_tokens.device,
                    )
        return {}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {}

    def before_step(
        self,
        scheduler_output: Any,
        dummy_run: bool = False,
    ) -> None:
        from dllm_plugin.forward_context import set_num_prefix_tokens_list

        self._pending_draft_ids = None
        if dummy_run:
            set_num_prefix_tokens_list(None)
            return

        raw = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
        self._scheduled_spec_decode_tokens = {k: tuple(v) for k, v in raw.items()}
        logger.info(
            "before_step: scheduled_spec_decode_tokens=%d reqs, sizes=%s",
            len(self._scheduled_spec_decode_tokens),
            {k: len(v) for k, v in self._scheduled_spec_decode_tokens.items()}
            if self._scheduled_spec_decode_tokens
            else "empty",
        )

        active = set(getattr(scheduler_output, "num_scheduled_tokens", {}).keys())
        for stale in set(self._denoise_step) - active:
            self._denoise_step.pop(stale, None)
            self._initial_prompt_len.pop(stale, None)

        dllm_npt = getattr(scheduler_output, "dllm_num_prefix_tokens", None)
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if dllm_npt and num_scheduled:
            req_ids = list(num_scheduled.keys())
            prefix_list = [dllm_npt.get(rid, 0) for rid in req_ids]
            set_num_prefix_tokens_list(prefix_list)
        else:
            set_num_prefix_tokens_list(None)

    def custom_sample(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor] | None:
        has_drafts = bool(self._scheduled_spec_decode_tokens)
        logger.info(
            "custom_sample: has_drafts=%s, num_draft_tokens=%d, "
            "scheduled=%d, num_reqs=%d",
            has_drafts,
            input_batch.num_draft_tokens,
            len(self._scheduled_spec_decode_tokens),
            input_batch.num_reqs,
        )
        if has_drafts and input_batch.num_draft_tokens == 0:
            logger.warning(
                "Scheduler sent %d draft requests but num_draft_tokens=0. "
                "Check DiffusionConfig wiring and num_speculative_steps.",
                len(self._scheduled_spec_decode_tokens),
            )
        if not has_drafts:
            # Prefill or no drafts yet — suppress AR token emission.
            # The first block's committed output will contain the real tokens.
            num_reqs = input_batch.num_reqs
            width = self._slot_width
            sampled = torch.zeros(
                (num_reqs, width), dtype=torch.int64, device=self.device
            )
            num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
            sampler_output = SamplerOutput(
                sampled_token_ids=sampled,
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=num_sampled,
            )
            num_sampled, num_rejected = get_num_sampled_and_rejected(
                num_sampled,
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                input_batch.idx_mapping,
                req_states.prefill_len.gpu,
            )
            return sampler_output, num_sampled, num_rejected

        from dllm_plugin.sampling.diffusion_sampler import batched_remask

        req_ids = input_batch.req_ids
        num_reqs = input_batch.num_reqs
        cu = input_batch.cu_num_logits_np
        mask_id = self._mask_id
        width = self._slot_width

        block_logits_list = []
        draft_tensors = []
        for i in range(num_reqs):
            lo, hi = int(cu[i]), int(cu[i + 1])
            all_logits = logits[lo:hi]
            if all_logits.shape[0] > self._draft_size:
                block_logits_list.append(all_logits[1:])
            else:
                block_logits_list.append(all_logits)

            req_id = req_ids[i]
            draft = self._scheduled_spec_decode_tokens.get(req_id, ())
            if len(draft) < self._draft_size:
                draft = tuple(draft) + (mask_id,) * (self._draft_size - len(draft))
            draft_tensors.append(
                torch.tensor(
                    draft[: self._draft_size], dtype=torch.int64, device=self.device
                )
            )

            if req_id not in self._initial_prompt_len:
                n_prompt = 0
                for t in draft:
                    if t != mask_id:
                        n_prompt += 1
                    else:
                        break
                self._initial_prompt_len[req_id] = n_prompt

        batch_logits = torch.stack(block_logits_list)
        batch_draft = torch.stack(draft_tensors)

        canvas, all_done, _ = batched_remask(
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

        for i in range(num_reqs):
            req_id = req_ids[i]
            done = all_done[i].item()
            step_idx = self._denoise_step.get(req_id, 0)

            if done:
                committed_t = canvas[i]
                n_prompt = self._initial_prompt_len.get(req_id, 0)
                if n_prompt > 0:
                    committed_t = committed_t[n_prompt:]
                n = committed_t.shape[0]
                sampled[i, :n] = committed_t
                nums.append(n)
                next_blocks.append([mask_id] * self._draft_size)
                self._denoise_step.pop(req_id, None)
                self._initial_prompt_len.pop(req_id, None)
            else:
                new_step = step_idx + 1
                if new_step >= self._max_denoise_iters:
                    logger.warning(
                        "req %s: denoise hit max iterations (%d)",
                        req_id,
                        self._max_denoise_iters,
                    )
                    non_mask = canvas[i][canvas[i] != mask_id]
                    n = non_mask.shape[0]
                    sampled[i, :n] = non_mask
                    nums.append(n)
                    next_blocks.append([mask_id] * self._draft_size)
                    self._denoise_step.pop(req_id, None)
                    self._initial_prompt_len.pop(req_id, None)
                else:
                    nums.append(0)
                    next_blocks.append(canvas[i].tolist())
                    self._denoise_step[req_id] = new_step

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
            req_states.prefill_len.gpu,
        )

        self._pending_draft_ids = DraftTokenIds(
            req_ids=list(req_ids),
            draft_token_ids=next_blocks,
        )
        return sampler_output, num_sampled, num_rejected

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        out = self._pending_draft_ids
        self._pending_draft_ids = None
        return out

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound

        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max().item())

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            causal=False,
        )
