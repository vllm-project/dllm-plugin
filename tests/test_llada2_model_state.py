# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LLaDA2ModelState hook behavior.

Tests use mocked vLLM objects — no model weights or GPU needed for
most tests. GPU-specific tests are marked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_vllm_config():
    config = MagicMock()
    config.model_config.max_model_len = 2048
    diff_cfg = MagicMock()
    diff_cfg.mask_token_id = 156895
    diff_cfg.draft_length = 32
    diff_cfg.commit_threshold = 0.9
    diff_cfg.max_denoise_steps = 64
    diff_cfg.num_speculative_tokens = 32
    config.diffusion_config = diff_cfg
    config.scheduler_config.max_num_seqs = 4
    config.scheduler_config.max_num_batched_tokens = 256
    return config


class TestModelStateInit:
    def test_num_bonus_tokens_is_zero(self, mock_vllm_config):
        pytest.importorskip("torch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from dllm_plugin.models.llada2_model_state import LLaDA2ModelState

        with patch("vllm.v1.worker.gpu.attn_utils.build_attn_metadata"):
            ms = LLaDA2ModelState(
                vllm_config=mock_vllm_config,
                model=MagicMock(),
                encoder_cache=None,
                device=torch.device("cuda:0"),
            )
        assert ms.num_bonus_tokens == 0

    def test_add_remove_request(self, mock_vllm_config):
        pytest.importorskip("torch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from dllm_plugin.models.llada2_model_state import LLaDA2ModelState

        with patch("vllm.v1.worker.gpu.attn_utils.build_attn_metadata"):
            ms = LLaDA2ModelState(
                vllm_config=mock_vllm_config,
                model=MagicMock(),
                encoder_cache=None,
                device=torch.device("cuda:0"),
            )

        req_data = MagicMock()
        req_data.req_id = "test_req_0"
        req_data.prompt_token_ids = [1, 2, 3, 4]

        ms.add_request(0, req_data)
        assert "test_req_0" in ms._slot_map
        assert ms._slot_map["test_req_0"] == 0
        assert ms._prompt_tail_ids["test_req_0"] == [1, 2, 3, 4]

        ms.remove_request("test_req_0")
        assert "test_req_0" not in ms._slot_map
        assert "test_req_0" not in ms._prompt_tail_ids

    def test_take_draft_token_ids_returns_pending(self, mock_vllm_config):
        pytest.importorskip("torch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from dllm_plugin.models.llada2_model_state import LLaDA2ModelState

        with patch("vllm.v1.worker.gpu.attn_utils.build_attn_metadata"):
            ms = LLaDA2ModelState(
                vllm_config=mock_vllm_config,
                model=MagicMock(),
                encoder_cache=None,
                device=torch.device("cuda:0"),
            )

        assert ms.take_draft_token_ids() is None

        mock_draft = MagicMock()
        ms._pending_draft_ids = mock_draft
        result = ms.take_draft_token_ids()
        assert result is mock_draft
        assert ms._pending_draft_ids is None

    def test_before_step_dummy_run_clears_state(self, mock_vllm_config):
        pytest.importorskip("torch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from dllm_plugin.models.llada2_model_state import LLaDA2ModelState

        with patch("vllm.v1.worker.gpu.attn_utils.build_attn_metadata"):
            ms = LLaDA2ModelState(
                vllm_config=mock_vllm_config,
                model=MagicMock(),
                encoder_cache=None,
                device=torch.device("cuda:0"),
            )

        ms._pending_draft_ids = MagicMock()
        ms._prefix_lengths = [32]
        ms.before_step(MagicMock(), dummy_run=True)
        assert ms._pending_draft_ids is None
        assert ms._prefix_lengths is None
