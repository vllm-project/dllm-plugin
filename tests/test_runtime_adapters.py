# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vLLM-facing runtime adapter classes."""

from __future__ import annotations

import importlib

import pytest


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
