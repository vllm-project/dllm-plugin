# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 4 worker tests (issue #10)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from vllm_dllm_plugin.config import DRAFT_SIZE, LLADA2_DEFAULT_MASK_TOKEN_ID
from vllm_dllm_plugin.remasking import Llada2DefaultRemaskingPolicy, RemaskStepResult
from vllm_dllm_plugin.worker import (
    DllmWorker,
    is_v2_model_runner_enabled,
)


def _draft_all_mask() -> tuple[int, ...]:
    return (LLADA2_DEFAULT_MASK_TOKEN_ID,) * DRAFT_SIZE


def _mock_logits(*, vocab_size: int = 128) -> list[list[float]]:
    rows: list[list[float]] = []
    for _ in range(DRAFT_SIZE):
        row = [0.0] * vocab_size
        row[0] = 1.0
        rows.append(row)
    return rows


class _BadBlockPolicy:
    """Returns an invalid next block length for contract-boundary testing."""

    __test__ = False

    def apply(
        self,
        *,
        input_draft: Sequence[int],
        logits: Any | None = None,
        remasking_config: Mapping[str, Any] | None = None,
    ) -> RemaskStepResult:
        del input_draft, logits, remasking_config
        return RemaskStepResult(
            committed_token_ids=(),
            next_input_block=(LLADA2_DEFAULT_MASK_TOKEN_ID,),
        )


def test_v2_model_runner_flag_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    assert is_v2_model_runner_enabled()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "false")
    assert not is_v2_model_runner_enabled()


def test_worker_requires_v2_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
        DllmWorker(require_v2_model_runner=True)


def test_worker_one_block_flow_maps_to_scheduler_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    worker = DllmWorker(require_v2_model_runner=True)
    policy = Llada2DefaultRemaskingPolicy()

    step = worker.run_one_block(
        request_id="r1",
        input_draft=_draft_all_mask(),
        logits=_mock_logits(),
        policy=policy,
        remasking_config={"num_transfer": DRAFT_SIZE},
    )

    assert step.request_id == "r1"
    assert len(step.sampled_token_ids) == DRAFT_SIZE
    assert len(step.next_input_block) == DRAFT_SIZE
    assert step.next_input_block == (LLADA2_DEFAULT_MASK_TOKEN_ID,) * DRAFT_SIZE
    assert worker.take_draft_token_ids(step) == step.next_input_block

    sched_result = worker.as_scheduler_result(step)
    assert sched_result.request_id == "r1"
    assert sched_result.sampled_token_ids == step.sampled_token_ids


def test_worker_rejects_malformed_policy_next_input_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    worker = DllmWorker(require_v2_model_runner=True)

    with pytest.raises(ValueError, match="next_input_block"):
        worker.run_one_block(
            request_id="r1",
            input_draft=_draft_all_mask(),
            logits=_mock_logits(),
            policy=_BadBlockPolicy(),
        )
