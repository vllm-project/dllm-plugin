# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``RemaskingPolicy`` / ``RemaskStepResult`` contract surface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from vllm_dllm_plugin.config import DRAFT_SIZE
from vllm_dllm_plugin.remasking import (
    RemaskingPolicy,
    RemaskStepResult,
    validate_remask_step_result,
)


class _StubPolicy:
    """Minimal structural implementation of :class:`RemaskingPolicy`."""

    def apply(
        self,
        *,
        input_draft: list[int] | tuple[int, ...],
        logits: Any | None = None,
        remasking_config: Mapping[str, Any] | None = None,
    ) -> RemaskStepResult:
        del input_draft, logits, remasking_config
        return RemaskStepResult(
            committed_token_ids=(),
            next_input_block=(0,) * DRAFT_SIZE,
        )


def test_remasking_policy_structural_subtype() -> None:
    stub = _StubPolicy()
    assert isinstance(stub, RemaskingPolicy)
    out = stub.apply(input_draft=tuple(range(DRAFT_SIZE)))
    validate_remask_step_result(out)


def test_validate_accepts_committed_length_equal_draft_size() -> None:
    """Upper inclusive bound: len(committed_token_ids) == DRAFT_SIZE is valid."""
    full_committed = tuple(range(DRAFT_SIZE))
    ok = RemaskStepResult(
        committed_token_ids=full_committed,
        next_input_block=(0,) * DRAFT_SIZE,
    )
    validate_remask_step_result(ok)


def test_validate_remask_step_result_rejects_wrong_next_block_length() -> None:
    bad = RemaskStepResult(committed_token_ids=(), next_input_block=(0,))
    with pytest.raises(ValueError, match="next_input_block"):
        validate_remask_step_result(bad)


def test_validate_remask_step_result_rejects_too_many_committed() -> None:
    too_many = tuple(range(DRAFT_SIZE + 1))
    bad = RemaskStepResult(
        committed_token_ids=too_many,
        next_input_block=(0,) * DRAFT_SIZE,
    )
    with pytest.raises(ValueError, match="committed_token_ids"):
        validate_remask_step_result(bad)
