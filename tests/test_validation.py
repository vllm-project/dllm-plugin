# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for strict stack validation helpers (issue #4)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_dllm_plugin.validation import assert_compatible_stack


def _build_vllm_config(
    *,
    archs: tuple[str, ...] = ("DllmMockLlada2StackTest",),
    scheduler_cls: type[object] | None = None,
    worker_cls: str = "vllm_dllm_plugin.runtime_worker:DllmRuntimeWorker",
) -> object:
    class _DefaultScheduler:
        pass

    scheduler_type = scheduler_cls or _DefaultScheduler
    scheduler_config = SimpleNamespace(get_scheduler_cls=lambda: scheduler_type)
    parallel_config = SimpleNamespace(worker_cls=worker_cls)
    model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=archs))
    return SimpleNamespace(
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        model_config=model_config,
    )


def test_assert_compatible_stack_accepts_runtime_scheduler_and_worker() -> None:
    class DllmRuntimeScheduler:
        __module__ = "vllm_dllm_plugin.runtime_scheduler"

    cfg = _build_vllm_config(scheduler_cls=DllmRuntimeScheduler)
    assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_rejects_non_dllm_architecture() -> None:
    class DllmRuntimeScheduler:
        __module__ = "vllm_dllm_plugin.runtime_scheduler"

    cfg = _build_vllm_config(
        archs=("SomeOtherModel",),
        scheduler_cls=DllmRuntimeScheduler,
    )
    with pytest.raises(
        ValueError,
        match="require a dLLM-compatible model architecture",
    ):
        assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_rejects_wrong_scheduler() -> None:
    class OtherScheduler:
        __module__ = "other.scheduler"

    cfg = _build_vllm_config(scheduler_cls=OtherScheduler)
    with pytest.raises(ValueError, match="invalid scheduler class"):
        assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_rejects_wrong_worker() -> None:
    class DllmRuntimeScheduler:
        __module__ = "vllm_dllm_plugin.runtime_scheduler"

    cfg = _build_vllm_config(
        scheduler_cls=DllmRuntimeScheduler,
        worker_cls="vllm.v1.worker.gpu_worker.Worker",
    )
    with pytest.raises(ValueError, match="invalid worker class"):
        assert_compatible_stack(cfg, caller="test")
