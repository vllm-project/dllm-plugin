# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remasking policies: composable post-forward draft updates (MVP)."""

from __future__ import annotations

from vllm_dllm_plugin.remasking.base import (
    RemaskingPolicy,
    RemaskStepResult,
    validate_remask_step_result,
)
from vllm_dllm_plugin.remasking.handoff import (
    assert_block_logits_shape,
    remask_after_block_forward,
)
from vllm_dllm_plugin.remasking.llada2_default import Llada2DefaultRemaskingPolicy

__all__ = [
    "Llada2DefaultRemaskingPolicy",
    "RemaskStepResult",
    "RemaskingPolicy",
    "assert_block_logits_shape",
    "remask_after_block_forward",
    "validate_remask_step_result",
]
