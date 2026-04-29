# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for strict stack validation helpers (issue #4)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dllm_plugin.config import (
    DLLM_STRICT_STACK_VALIDATION_ENV_VAR,
    LLADA2_ARCHITECTURE_NAME,
)
from dllm_plugin.validation import assert_compatible_stack


def _build_vllm_config(
    *,
    archs: tuple[str, ...] = ("DllmMockLlada2StackTest",),
    scheduler_cls: type[object] | None = None,
    worker_cls: str = "dllm_plugin.runtime_worker:DllmRuntimeWorker",
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


def test_assert_compatible_stack_accepts_llada2_architecture_name() -> None:
    class DllmRuntimeScheduler:
        __module__ = "dllm_plugin.runtime_scheduler"

    cfg = _build_vllm_config(
        archs=(LLADA2_ARCHITECTURE_NAME,),
        scheduler_cls=DllmRuntimeScheduler,
    )
    assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_accepts_runtime_scheduler_and_worker() -> None:
    class DllmRuntimeScheduler:
        __module__ = "dllm_plugin.runtime_scheduler"

    cfg = _build_vllm_config(scheduler_cls=DllmRuntimeScheduler)
    assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_accepts_dllm_plugin_worker_alias() -> None:
    class DllmRuntimeScheduler:
        __module__ = "dllm_plugin.runtime_scheduler"

    cfg = _build_vllm_config(
        scheduler_cls=DllmRuntimeScheduler,
        worker_cls="dllm_plugin.Worker",
    )
    assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_rejects_non_dllm_architecture() -> None:
    class DllmRuntimeScheduler:
        __module__ = "dllm_plugin.runtime_scheduler"

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


def test_assert_compatible_stack_rejects_worker_cls_auto() -> None:
    class DllmRuntimeScheduler:
        __module__ = "dllm_plugin.runtime_scheduler"

    cfg = _build_vllm_config(
        scheduler_cls=DllmRuntimeScheduler,
        worker_cls="auto",
    )
    with pytest.raises(ValueError, match="still 'auto'"):
        assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_rejects_wrong_worker() -> None:
    class DllmRuntimeScheduler:
        __module__ = "dllm_plugin.runtime_scheduler"

    # Wrong worker type resolvable without vLLM (minimal test envs omit vllm).
    cfg = _build_vllm_config(
        scheduler_cls=DllmRuntimeScheduler,
        worker_cls="dllm_plugin.scheduler.DllmScheduler",
    )
    with pytest.raises(ValueError, match="invalid worker class"):
        assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_strict_false_skips_checks() -> None:
    class OtherScheduler:
        __module__ = "other.scheduler"

    cfg = _build_vllm_config(scheduler_cls=OtherScheduler)
    assert_compatible_stack(cfg, caller="test", strict=False)


def test_assert_compatible_stack_respects_env_disable_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OtherScheduler:
        __module__ = "other.scheduler"

    cfg = _build_vllm_config(scheduler_cls=OtherScheduler)
    monkeypatch.setenv(DLLM_STRICT_STACK_VALIDATION_ENV_VAR, "0")
    assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_missing_scheduler_config() -> None:
    parallel_config = SimpleNamespace(
        worker_cls="dllm_plugin.runtime_worker.DllmRuntimeWorker",
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=("DllmMockLlada2StackTest",)),
    )
    cfg = SimpleNamespace(parallel_config=parallel_config, model_config=model_config)
    with pytest.raises(ValueError, match="missing scheduler_config"):
        assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_missing_parallel_config() -> None:
    class DllmRuntimeScheduler:
        __module__ = "dllm_plugin.runtime_scheduler"

    scheduler_config = SimpleNamespace(get_scheduler_cls=lambda: DllmRuntimeScheduler)
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=("DllmMockLlada2StackTest",)),
    )
    cfg = SimpleNamespace(scheduler_config=scheduler_config, model_config=model_config)
    with pytest.raises(ValueError, match="missing parallel_config"):
        assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_scheduler_resolution_failure() -> None:
    def boom() -> None:
        raise ImportError("simulated import failure")

    scheduler_config = SimpleNamespace(get_scheduler_cls=boom)
    parallel_config = SimpleNamespace(
        worker_cls="dllm_plugin.runtime_worker.DllmRuntimeWorker",
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=("DllmMockLlada2StackTest",)),
    )
    cfg = SimpleNamespace(
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        model_config=model_config,
    )
    with pytest.raises(ValueError, match="failed to resolve scheduler class"):
        assert_compatible_stack(cfg, caller="test")


def test_assert_compatible_stack_includes_caller_in_errors() -> None:
    class DllmRuntimeScheduler:
        __module__ = "dllm_plugin.runtime_scheduler"

    cfg = _build_vllm_config(
        archs=("SomeOtherModel",),
        scheduler_cls=DllmRuntimeScheduler,
    )
    with pytest.raises(ValueError) as excinfo:
        assert_compatible_stack(cfg, caller="my_unit_test")
    assert "context: 'my_unit_test'" in str(excinfo.value)
