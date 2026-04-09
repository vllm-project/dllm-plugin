# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm_dllm_plugin.config``."""

from __future__ import annotations

from vllm_dllm_plugin import config


def test_draft_size_matches_design_mvp_default() -> None:
    assert config.DRAFT_SIZE == 32


def test_model_and_flag_constants_are_non_empty() -> None:
    assert config.LLADA2_ARCHITECTURE_NAME
    assert config.DLLM_MOCK_STACK_MODEL_ID
    assert isinstance(config.DLLM_STRICT_STACK_VALIDATION_DEFAULT, bool)
