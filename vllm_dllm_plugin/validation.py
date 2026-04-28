# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict stack validation for dLLM runtime wiring (issue #4)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vllm_dllm_plugin.config import (
    DLLM_MOCK_STACK_MODEL_ID,
    DLLM_STRICT_STACK_VALIDATION_DEFAULT,
    LLADA2_ARCHITECTURE_NAME,
)

_SCHEDULER_FQCN_DOT = "vllm_dllm_plugin.runtime_scheduler.DllmRuntimeScheduler"
_SCHEDULER_FQCN_COLON = "vllm_dllm_plugin.runtime_scheduler:DllmRuntimeScheduler"
_WORKER_FQCN_DOT = "vllm_dllm_plugin.runtime_worker.DllmRuntimeWorker"
_WORKER_FQCN_COLON = "vllm_dllm_plugin.runtime_worker:DllmRuntimeWorker"


def _normalize_fqcn(value: str) -> str:
    return value.replace(":", ".")


def _get_model_architectures(vllm_config: Any) -> tuple[str, ...]:
    hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    if hf_config is None:
        return ()
    archs = getattr(hf_config, "architectures", None)
    if archs is None:
        return ()
    if isinstance(archs, str):
        return (archs,)
    if isinstance(archs, Iterable):
        return tuple(str(item) for item in archs)
    return ()


def _is_dllm_model_architecture(vllm_config: Any) -> bool:
    archs = set(_get_model_architectures(vllm_config))
    return bool(
        archs.intersection({LLADA2_ARCHITECTURE_NAME, DLLM_MOCK_STACK_MODEL_ID}),
    )


def assert_compatible_stack(
    vllm_config: Any,
    *,
    caller: str,
    strict: bool = DLLM_STRICT_STACK_VALIDATION_DEFAULT,
) -> None:
    """Fail fast when scheduler/worker/model stack is incompatible for dLLM."""

    if not strict:
        return

    archs = _get_model_architectures(vllm_config)
    if not _is_dllm_model_architecture(vllm_config):
        raise ValueError(
            "dLLM runtime adapters require a dLLM-compatible model architecture "
            f"(got architectures={archs!r}); expected one of "
            f"{(LLADA2_ARCHITECTURE_NAME, DLLM_MOCK_STACK_MODEL_ID)!r}",
        )

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is None:
        raise ValueError(
            "missing scheduler_config in vLLM config for dLLM runtime stack",
        )
    try:
        scheduler_cls = scheduler_config.get_scheduler_cls()
    except Exception as exc:
        raise ValueError(
            "failed to resolve scheduler class for dLLM runtime stack; use "
            "--scheduler-cls "
            "vllm_dllm_plugin.runtime_scheduler.DllmRuntimeScheduler "
            "(dotted qualname expected by vLLM)",
        ) from exc
    scheduler_fqcn = _normalize_fqcn(
        f"{scheduler_cls.__module__}.{scheduler_cls.__name__}",
    )
    if scheduler_fqcn != _normalize_fqcn(_SCHEDULER_FQCN_DOT):
        raise ValueError(
            "invalid scheduler class for dLLM runtime stack: "
            f"got={scheduler_fqcn!r} expected one of "
            f"{(_SCHEDULER_FQCN_DOT, _SCHEDULER_FQCN_COLON)!r}; "
            "pass --scheduler-cls "
            "vllm_dllm_plugin.runtime_scheduler.DllmRuntimeScheduler",
        )

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        raise ValueError(
            "missing parallel_config in vLLM config for dLLM runtime stack",
        )
    worker_cls = _normalize_fqcn(str(getattr(parallel_config, "worker_cls", "")))
    if worker_cls != _normalize_fqcn(_WORKER_FQCN_DOT):
        raise ValueError(
            "invalid worker class for dLLM runtime stack: "
            f"got={worker_cls!r} expected one of "
            f"{(_WORKER_FQCN_DOT, _WORKER_FQCN_COLON)!r}; "
            "pass --worker-cls "
            "vllm_dllm_plugin.runtime_worker.DllmRuntimeWorker",
        )

    del caller


__all__ = ["assert_compatible_stack"]
