# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LLaDA2 ModelState for MRV2 integration.

Implements the ModelState interface for block diffusion models.
Handles non-causal attention, per-request denoising state, draft token
management, and diffusion sampling via DiffusionSampler.
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
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from dllm_plugin.config import (
    DRAFT_SIZE,
    LLADA2_DEFAULT_MASK_TOKEN_ID,
)

logger = logging.getLogger(__name__)


class LLaDA2ModelState(ModelState):
    """ModelState for LLaDA2 block diffusion models.

    All dLLM logic lives here via ModelState composition hooks:
    prepare_attn, custom_sampler, before_step, take_draft_token_ids.
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
        self._draft_size = diff_cfg.draft_length if diff_cfg else DRAFT_SIZE
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
        self._prefix_lengths: list[int] | None = None

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        return ("generate",)

    @property
    def num_bonus_tokens(self) -> int:
        return 0

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
        self._pending_draft_ids = None
        self._prefix_lengths = None
        if dummy_run:
            return

        raw = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
        self._scheduled_spec_decode_tokens = {k: tuple(v) for k, v in raw.items()}

        active = set(getattr(scheduler_output, "num_scheduled_tokens", {}).keys())
        for stale in set(self._denoise_step) - active:
            self._denoise_step.pop(stale, None)
            self._initial_prompt_len.pop(stale, None)

        dllm_npt = getattr(scheduler_output, "dllm_num_prefix_tokens", None)
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if dllm_npt and num_scheduled:
            req_ids = list(num_scheduled.keys())
            self._prefix_lengths = [dllm_npt.get(rid, 0) for rid in req_ids]

    def custom_sampler(
        self,
        sampler: Any,
        config: Any,
    ) -> tuple[Any, Any] | None:
        from dllm_plugin.sampling.diffusion_sampler import DiffusionSampler

        return (
            DiffusionSampler(
                model_state=self,
                device=self.device,
                mask_token_id=self._mask_id,
                draft_size=self._draft_size,
                threshold=self._threshold,
                max_denoise_iters=self._max_denoise_iters,
                slot_width=self._slot_width,
            ),
            None,
        )

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
        # LLaDA2 uses bidirectional attention within blocks for ALL steps
        # (including prefill). The block-causal structure (causal across
        # blocks, non-causal within) is handled by the virtual batch
        # decomposition in LLaDA2BidirectionalAttentionBuilder.build(),
        # not by the per-request causal flag. Per-request causal tensors
        # would be needed for mixed AR+diffusion batches, which aren't
        # supported yet.
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
            dllm_prefix_lengths=self._prefix_lengths,
        )
