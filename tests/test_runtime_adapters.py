# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vLLM-facing runtime adapter classes."""

from __future__ import annotations

import importlib

import pytest

from vllm_dllm_plugin.config import DRAFT_SIZE, LLADA2_DEFAULT_MASK_TOKEN_ID
from vllm_dllm_plugin.scheduler import DllmScheduler
from vllm_dllm_plugin.worker import DllmWorker


def test_runtime_adapter_fqcn_targets_resolve() -> None:
    mod_sched = importlib.import_module("vllm_dllm_plugin.runtime_scheduler")
    mod_worker = importlib.import_module("vllm_dllm_plugin.runtime_worker")
    assert hasattr(mod_sched, "DllmRuntimeScheduler")
    assert hasattr(mod_worker, "DllmRuntimeWorker")


def test_runtime_scheduler_behavior_depends_on_vllm_availability() -> None:
    from vllm_dllm_plugin.runtime_scheduler import (
        _VLLM_AVAILABLE,
        DllmRuntimeScheduler,
    )

    if not _VLLM_AVAILABLE:
        with pytest.raises(RuntimeError, match="requires vLLM"):
            DllmRuntimeScheduler()
    else:
        from vllm.v1.core.sched.scheduler import Scheduler

        assert issubclass(DllmRuntimeScheduler, Scheduler)


def test_runtime_worker_behavior_depends_on_vllm_availability() -> None:
    from vllm_dllm_plugin.runtime_worker import (
        _VLLM_AVAILABLE,
        DllmRuntimeWorker,
    )

    if not _VLLM_AVAILABLE:
        with pytest.raises(RuntimeError, match="requires vLLM"):
            DllmRuntimeWorker()
    else:
        from vllm.v1.worker.gpu_worker import Worker

        assert issubclass(DllmRuntimeWorker, Worker)


class _FakeModelRunnerOutput:
    def __init__(self, req_ids: list[str], sampled_token_ids: list[list[int]]) -> None:
        self.req_ids = req_ids
        self.sampled_token_ids = sampled_token_ids


def test_runtime_contract_progress_with_default_policy() -> None:
    from vllm_dllm_plugin.runtime_scheduler import validate_scheduler_worker_contract
    from vllm_dllm_plugin.runtime_worker import run_block_contract_from_model_output

    worker_helper = DllmWorker(require_v2_model_runner=False)
    scheduler_helper = DllmScheduler()
    input_draft = [LLADA2_DEFAULT_MASK_TOKEN_ID] * DRAFT_SIZE

    step = run_block_contract_from_model_output(
        helper=worker_helper,
        request_id="r1",
        input_draft=input_draft,
        sampled_token_ids=[],
    )
    assert len(step.sampled_token_ids) > 0
    assert len(step.next_input_block) == DRAFT_SIZE

    fake_out = _FakeModelRunnerOutput(
        req_ids=["r1"],
        sampled_token_ids=[list(step.sampled_token_ids)],
    )
    validate_scheduler_worker_contract(
        helper=scheduler_helper,
        expected_req_ids=("r1",),
        model_runner_output=fake_out,
    )


def test_runtime_scheduler_contract_rejects_missing_output_coverage() -> None:
    from vllm_dllm_plugin.runtime_scheduler import validate_scheduler_worker_contract

    fake_out = _FakeModelRunnerOutput(req_ids=["r1"], sampled_token_ids=[[]])
    with pytest.raises(ValueError, match="missing worker results"):
        validate_scheduler_worker_contract(
            helper=DllmScheduler(),
            expected_req_ids=("r1", "r2"),
            model_runner_output=fake_out,
        )


def test_runtime_worker_contract_rejects_missing_input_draft() -> None:
    from vllm_dllm_plugin.runtime_worker import validate_runtime_input_draft

    with pytest.raises(ValueError, match="missing scheduled_spec_decode_tokens"):
        validate_runtime_input_draft(
            request_id="r1",
            input_draft=None,
            draft_size=DRAFT_SIZE,
        )


def test_runtime_worker_contract_rejects_malformed_input_draft_length() -> None:
    from vllm_dllm_plugin.runtime_worker import validate_runtime_input_draft

    with pytest.raises(ValueError, match="malformed scheduled_spec_decode_tokens"):
        validate_runtime_input_draft(
            request_id="r1",
            input_draft=[1, 2, 3],
            draft_size=DRAFT_SIZE,
        )


def test_runtime_worker_contract_rejects_missing_draft_handoff_coverage() -> None:
    from vllm_dllm_plugin.runtime_worker import validate_runtime_draft_handoff_coverage

    with pytest.raises(ValueError, match="missing request_id"):
        validate_runtime_draft_handoff_coverage(
            expected_req_ids={"r1", "r2"},
            produced_req_ids=["r1"],
        )


def test_runtime_worker_contract_rejects_duplicate_draft_handoff_coverage() -> None:
    from vllm_dllm_plugin.runtime_worker import validate_runtime_draft_handoff_coverage

    with pytest.raises(ValueError, match="duplicate request_id"):
        validate_runtime_draft_handoff_coverage(
            expected_req_ids={"r1"},
            produced_req_ids=["r1", "r1"],
        )
