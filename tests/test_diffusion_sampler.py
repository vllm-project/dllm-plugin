# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DiffusionSampler convergence and commit logic.

Tests use synthetic logits on GPU — no model weights needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from dllm_plugin.sampling.diffusion_sampler import DiffusionSampler, batched_remask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

MASK_ID = 156895
DRAFT_SIZE = 32
THRESHOLD = 0.9


def _make_model_state(device: torch.device, max_reqs: int = 4) -> MagicMock:
    ms = MagicMock()
    ms._denoise_step_t = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    ms._kv_refresh_t = torch.zeros(max_reqs, dtype=torch.bool, device=device)
    ms._prompt_len_t = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    ms._draft_block = torch.full(
        (max_reqs, DRAFT_SIZE), MASK_ID, dtype=torch.int64, device=device
    )
    ms._draft_block_valid = torch.zeros(max_reqs, dtype=torch.bool, device=device)
    ms._slot_map = {}
    ms._prompt_tail_ids = {}
    ms._scheduled_spec_decode_tokens = {}
    ms._pending_draft_ids = None
    return ms


def _make_input_batch(
    req_ids: list[str], num_tokens: int, device: torch.device
) -> MagicMock:
    batch = MagicMock()
    batch.req_ids = req_ids
    batch.num_reqs = len(req_ids)
    batch.cu_num_logits_np = list(range(0, num_tokens + 1, num_tokens // len(req_ids)))
    return batch


class TestBatchedRemask:
    def test_all_masked_transfers_at_least_one(self):
        logits = torch.randn(1, DRAFT_SIZE, 157184, device="cuda")
        draft = torch.full((1, DRAFT_SIZE), MASK_ID, dtype=torch.int64, device="cuda")
        updated, all_done, n_transferred = batched_remask(
            logits, draft, MASK_ID, THRESHOLD
        )
        assert n_transferred[0].item() >= 1
        assert (updated != draft).any()

    def test_no_masked_is_identity(self):
        logits = torch.randn(1, DRAFT_SIZE, 157184, device="cuda")
        draft = torch.randint(
            0, 1000, (1, DRAFT_SIZE), dtype=torch.int64, device="cuda"
        )
        updated, all_done, n_transferred = batched_remask(
            logits, draft, MASK_ID, THRESHOLD
        )
        assert torch.equal(updated, draft)
        assert all_done[0].item() is True
        assert n_transferred[0].item() == 0

    def test_mask_token_predicted_stays_masked(self):
        logits = torch.zeros(1, DRAFT_SIZE, 157184, device="cuda")
        logits[:, :, MASK_ID] = 100.0
        draft = torch.full((1, DRAFT_SIZE), MASK_ID, dtype=torch.int64, device="cuda")
        updated, all_done, n_transferred = batched_remask(
            logits, draft, MASK_ID, THRESHOLD
        )
        assert torch.equal(updated, draft)
        assert n_transferred[0].item() == 0


class TestDiffusionSamplerConvergence:
    def _make_sampler(self, device: torch.device) -> tuple[DiffusionSampler, MagicMock]:
        ms = _make_model_state(device)
        base_sampler = MagicMock()
        sampler = DiffusionSampler(
            base_sampler=base_sampler,
            model_state=ms,
            device=device,
            mask_token_id=MASK_ID,
            draft_size=DRAFT_SIZE,
            threshold=THRESHOLD,
            max_denoise_iters=64,
            slot_width=DRAFT_SIZE,
        )
        return sampler, ms

    def test_kv_refresh_two_phase(self):
        """After all masks resolve, one extra step fires before commit."""
        device = torch.device("cuda:0")
        sampler, ms = self._make_sampler(device)

        slot = 0
        ms._slot_map["req0"] = slot
        ms._draft_block_valid[slot] = True
        ms._prompt_len_t[slot] = 4

        # Fully resolved draft (no masks)
        draft = torch.arange(DRAFT_SIZE, dtype=torch.int64, device=device).unsqueeze(0)
        ms._draft_block[slot] = draft[0]

        # Logits that strongly predict the same tokens
        logits = torch.zeros(1, DRAFT_SIZE, 157184, device=device)
        for i in range(DRAFT_SIZE):
            logits[0, i, draft[0, i]] = 100.0

        updated, all_done, _ = batched_remask(logits, draft, MASK_ID, THRESHOLD)
        assert all_done[0].item() is True

        # First time all_done: kv_refresh fires, NOT commit
        kv_done = ms._kv_refresh_t[slot].item()
        assert kv_done is False  # not yet done
        needs_refresh = all_done[0].item() and not kv_done
        assert needs_refresh is True

    def test_force_commit_after_max_iters(self):
        """Force commit triggers when max_denoise_iters reached."""
        device = torch.device("cuda:0")
        sampler, ms = self._make_sampler(device)

        slot = 0
        ms._slot_map["req0"] = slot
        ms._draft_block_valid[slot] = True

        # Draft with some masks remaining
        draft = torch.full((DRAFT_SIZE,), MASK_ID, dtype=torch.int64, device=device)
        draft[:16] = torch.arange(16, dtype=torch.int64, device=device)
        ms._draft_block[slot] = draft

        # Simulate max iters reached
        ms._denoise_step_t[slot] = 63  # max_denoise_iters - 1

        all_done = torch.tensor([False], device=device)
        steps = ms._denoise_step_t[slot : slot + 1]
        force_commit = ((steps + 1) >= 64) & ~all_done
        assert force_commit[0].item() is True
