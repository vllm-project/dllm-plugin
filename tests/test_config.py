# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm_dllm_plugin.config``."""

from __future__ import annotations

import importlib

import pytest

from vllm_dllm_plugin import config


def test_draft_size_matches_design_mvp_default() -> None:
    assert config.DRAFT_SIZE == 32


def test_model_and_flag_constants_are_non_empty() -> None:
    assert config.LLADA2_ARCHITECTURE_NAME
    assert config.DLLM_MOCK_STACK_MODEL_ID
    assert isinstance(config.DLLM_STRICT_STACK_VALIDATION_DEFAULT, bool)


def test_draft_size_can_be_configured_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DLLM_DRAFT_SIZE_ENV_VAR, "48")
    reloaded = importlib.reload(config)
    assert reloaded.DRAFT_SIZE == 48
    monkeypatch.delenv(config.DLLM_DRAFT_SIZE_ENV_VAR, raising=False)
    importlib.reload(config)
