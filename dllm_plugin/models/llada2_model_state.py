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

    .. note::
        Per-request state (``_denoise_step``, ``_initial_prompt_len``, etc.)
        is stored in Python dicts. For CUDA graph compatibility these should
        migrate to pre-allocated GPU tensors indexed by request slot.
        ``prepare_inputs()`` clones tensors per step which also prevents
        graph capture; persistent buffers updated in-place are needed.
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
        self._slot_width = diff_cfg.num_speculative_tokens if diff_cfg else DRAFT_SIZE

        # GPU-resident per-request state, indexed by slot.
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        self._denoise_step_t = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._kv_refresh_t = torch.zeros(max_reqs, dtype=torch.bool, device=device)
        self._prompt_len_t = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._draft_block = torch.full(
            (max_reqs, self._draft_size),
            self._mask_id,
            dtype=torch.int64,
            device=device,
        )
        self._draft_block_valid = torch.zeros(max_reqs, dtype=torch.bool, device=device)
        self._slot_map: dict[str, int] = {}

        # Persistent input buffers for CUDA graph capture. prepare_inputs()
        # writes into these via .copy_() so the graph replay reads updated
        # values from the same memory addresses.
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._input_ids_buf = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self._positions_buf = torch.zeros(max_tokens, dtype=torch.long, device=device)

        # Prompt tail: immutable after add_request, not on hot path (A2).
        self._prompt_tail_ids: dict[str, list[int]] = {}

        self._scheduled_spec_decode_tokens: dict[str, tuple[int, ...]] = {}
        self._pending_draft_ids: DraftTokenIds | None = None
        self._prefix_lengths: list[int] | None = None

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        return ("generate",)

    @property
    def num_bonus_tokens(self) -> int:
        # Diffusion models don't produce a "next token sample" like AR
        # spec-decode. Each denoising step processes the full draft block
        # (no bonus token prepended). EOS is generated within block content
        # by the diffusion decoder, not as a separate bonus token.
        return 0

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        req_id = new_req_data.req_id
        self._slot_map[req_id] = req_index
        prompt_ids = new_req_data.prompt_token_ids
        if prompt_ids:
            prompt_in_block = len(prompt_ids) % self._draft_size
            if prompt_in_block == 0 and len(prompt_ids) > 0:
                prompt_in_block = self._draft_size
            self._prompt_len_t[req_index] = prompt_in_block
            self._prompt_tail_ids[req_id] = list(prompt_ids[-prompt_in_block:])
        self._denoise_step_t[req_index] = 0
        self._kv_refresh_t[req_index] = False
        self._draft_block[req_index].fill_(self._mask_id)
        self._draft_block_valid[req_index] = False

    def remove_request(self, req_id: str) -> None:
        slot = self._slot_map.pop(req_id, None)
        if slot is not None:
            self._denoise_step_t[slot] = 0
            self._kv_refresh_t[slot] = False
            self._prompt_len_t[slot] = 0
            self._draft_block[slot].fill_(self._mask_id)
            self._draft_block_valid[slot] = False
        self._prompt_tail_ids.pop(req_id, None)

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

        if not self._scheduled_spec_decode_tokens or not self._prefix_lengths:
            return {}

        num_tokens = input_batch.num_tokens
        if num_tokens == 0 or input_batch.num_reqs == 0:
            return {}

        # Full-block recomp: only for the first block where the prompt tail
        # is within the current block (prefix < draft_size). For subsequent
        # blocks the standard virtual batch concatenation handles prefix KV.
        if input_batch.num_reqs != 1:
            logger.warning(
                "Full-block recomputation skipped: requires num_reqs=1 "
                "(got %d). First-block prompt-tail injection disabled.",
                input_batch.num_reqs,
            )
            return {}

        prefix_len = self._prefix_lengths[0]
        if not (0 < prefix_len < self._draft_size):
            return {}

        req_id = input_batch.req_ids[0]
        tail_ids = self._prompt_tail_ids.get(req_id)
        if not tail_ids:
            return {}

        # Write into persistent buffers (stable addresses for graph replay)
        ids_buf = self._input_ids_buf[:num_tokens]
        ids_buf.copy_(input_batch.input_ids[:num_tokens])
        n_tail = len(tail_ids)
        if n_tail > 0 and num_tokens >= n_tail:
            ids_buf[:n_tail] = torch.tensor(
                tail_ids,
                dtype=ids_buf.dtype,
                device=ids_buf.device,
            )

        pos_buf = self._positions_buf[:num_tokens]
        pos_buf.copy_(input_batch.positions[:num_tokens])
        pos_buf -= prefix_len
        pos_buf.clamp_(min=0)

        return {"input_ids": ids_buf, "positions": pos_buf}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {
            "input_ids": self._input_ids_buf[:num_tokens],
            "positions": self._positions_buf[:num_tokens],
        }

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

        # Copy scheduler draft tokens to persistent draft block tensor
        for req_id, tokens in self._scheduled_spec_decode_tokens.items():
            slot = self._slot_map.get(req_id)
            if slot is not None and len(tokens) > 0:
                n = min(len(tokens), self._draft_size)
                self._draft_block[slot, :n] = torch.tensor(
                    tokens[:n], dtype=self._draft_block.dtype, device=self.device
                )
                self._draft_block_valid[slot] = True

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
        if not hasattr(self, "_diffusion_sampler"):
            from dllm_plugin.sampling.diffusion_sampler import DiffusionSampler

            self._diffusion_sampler = DiffusionSampler(
                base_sampler=sampler,
                model_state=self,
                device=self.device,
                mask_token_id=self._mask_id,
                draft_size=self._draft_size,
                threshold=self._threshold,
                max_denoise_iters=self._max_denoise_iters,
                slot_width=self._slot_width,
            )
        return (self._diffusion_sampler, None)

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

        # Full-block recomputation: remap slot_mappings so both the KV
        # cache WRITE (via forward context) and READ (via attention metadata)
        # target positions [0..31] instead of [prefix..prefix+31].
        effective_prefix_lengths = self._prefix_lengths
        effective_seq_lens = input_batch.seq_lens
        effective_slot_mappings = slot_mappings
        remapped_slot_mappings = None

        if (
            self._prefix_lengths
            and self._scheduled_spec_decode_tokens
            and num_tokens > 0
            and num_reqs == 1
        ):
            prefix_len = self._prefix_lengths[0]
            if 0 < prefix_len < self._draft_size:
                kv_block_size = kv_cache_config.kv_cache_groups[
                    0
                ].kv_cache_spec.block_size
                bt = block_tables[0]  # [1, max_blocks]

                positions = input_batch.positions[:num_tokens]
                new_pos = (positions - prefix_len).clamp(min=0)
                page_idx = new_pos // kv_block_size
                page_offset = new_pos % kv_block_size
                page_nums = bt[0][page_idx.long()]
                new_slot_vals = page_nums * kv_block_size + page_offset
                valid = page_nums >= 0
                new_slots = slot_mappings.clone()
                new_slots[0, :num_tokens] = torch.where(
                    valid, new_slot_vals, slot_mappings[0, :num_tokens]
                )
                effective_slot_mappings = new_slots
                remapped_slot_mappings = new_slots

                effective_prefix_lengths = [0] * len(self._prefix_lengths)
                effective_seq_lens = torch.full_like(
                    input_batch.seq_lens,
                    self._draft_size,
                )
                seq_lens_cpu_upper_bound = torch.tensor(
                    [self._draft_size] * num_reqs,
                    dtype=seq_lens_cpu_upper_bound.dtype,
                )
                max_seq_len = self._draft_size

        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=effective_seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=effective_slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            causal=False,
            dllm_prefix_lengths=effective_prefix_lengths,
        )
        if remapped_slot_mappings is not None:
            attn_metadata["remapped_slot_mappings"] = remapped_slot_mappings
        return attn_metadata
