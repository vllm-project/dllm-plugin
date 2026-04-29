# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 4 worker helpers for one-block dLLM execution.

This module keeps the worker/scheduler contract explicit while remaining usable
in dev environments that do not install ``vllm`` (optional dependency).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dllm_plugin.config import DRAFT_SIZE
from dllm_plugin.remasking import RemaskingPolicy, remask_after_block_forward
from dllm_plugin.scheduler import DllmWorkerResult


def is_v2_model_runner_enabled() -> bool:
    """Return ``True`` when ``VLLM_USE_V2_MODEL_RUNNER`` is enabled."""

    value = os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class DllmWorkerStep:
    """Result of one block forward + remask handoff for one request."""

    request_id: str
    sampled_token_ids: tuple[int, ...]
    next_input_block: tuple[int, ...]


class DllmWorker:
    """Thin worker façade aligned with issue #10's one-block contract."""

    def __init__(self, *, require_v2_model_runner: bool = True) -> None:
        self.draft_size = DRAFT_SIZE
        self.require_v2_model_runner = require_v2_model_runner
        self.v2_model_runner_enabled = is_v2_model_runner_enabled()
        if self.require_v2_model_runner and not self.v2_model_runner_enabled:
            raise RuntimeError(
                "DllmWorker requires VLLM_USE_V2_MODEL_RUNNER=1 for MVP Phase 4.",
            )

    def run_one_block(
        self,
        *,
        request_id: str,
        input_draft: Sequence[int],
        logits: Any,
        policy: RemaskingPolicy,
        remasking_config: Mapping[str, Any] | None = None,
    ) -> DllmWorkerStep:
        """Execute one dLLM block and map remask output to worker fields."""

        result = remask_after_block_forward(
            input_draft=input_draft,
            logits=logits,
            policy=policy,
            remasking_config=remasking_config,
            draft_size=self.draft_size,
        )
        return DllmWorkerStep(
            request_id=request_id,
            sampled_token_ids=tuple(int(tok) for tok in result.committed_token_ids),
            next_input_block=tuple(int(tok) for tok in result.next_input_block),
        )

    def take_draft_token_ids(self, step: DllmWorkerStep) -> tuple[int, ...]:
        """Return the next draft block for scheduler ``update_draft_token_ids``."""

        if len(step.next_input_block) != self.draft_size:
            raise ValueError(
                "next_input_block length mismatch in take_draft_token_ids: "
                f"expected {self.draft_size}, got {len(step.next_input_block)}",
            )
        return step.next_input_block

    def as_scheduler_result(self, step: DllmWorkerStep) -> DllmWorkerResult:
        """Convert worker output to scheduler accounting payload."""

        return DllmWorkerResult(
            request_id=step.request_id,
            sampled_token_ids=step.sampled_token_ids,
        )


__all__ = [
    "DllmWorker",
    "DllmWorkerStep",
    "is_v2_model_runner_enabled",
]
