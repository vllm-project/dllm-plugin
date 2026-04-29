# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``dllm_plugin.config``."""

from __future__ import annotations

import importlib

import pytest

from dllm_plugin import config


def test_draft_size_matches_design_mvp_default() -> None:
    assert config.DRAFT_SIZE == 32


def test_model_and_flag_constants_are_non_empty() -> None:
    assert config.LLADA2_ARCHITECTURE_NAME
    assert config.DLLM_MOCK_STACK_MODEL_ID
    assert config.DLLM_STRICT_STACK_VALIDATION_ENV_VAR
    assert isinstance(config.LLADA2_DEFAULT_MASK_TOKEN_ID, int)
    assert config.LLADA2_DEFAULT_MASK_TOKEN_ID >= 0
    assert isinstance(config.LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD, float)
    assert 0.0 < config.LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD < 1.0


def test_resolve_strict_stack_validation_explicit_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(config.DLLM_STRICT_STACK_VALIDATION_ENV_VAR, "0")
    assert config.resolve_strict_stack_validation(True) is True
    assert config.resolve_strict_stack_validation(False) is False
    assert config.resolve_strict_stack_validation(None) is False


def test_resolve_strict_stack_validation_invalid_env_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(config.DLLM_STRICT_STACK_VALIDATION_ENV_VAR, "bogus")
    with pytest.raises(ValueError, match="must be one of"):
        config.resolve_strict_stack_validation(None)


def test_draft_size_can_be_configured_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DLLM_DRAFT_SIZE_ENV_VAR, "48")
    reloaded = importlib.reload(config)
    assert reloaded.DRAFT_SIZE == 48
    monkeypatch.delenv(config.DLLM_DRAFT_SIZE_ENV_VAR, raising=False)
    importlib.reload(config)
