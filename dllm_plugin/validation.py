# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict stack validation for dLLM runtime wiring (issue #4)."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterable
from typing import Any

from dllm_plugin.config import (
    DLLM_MOCK_STACK_MODEL_ID,
    LLADA2_ARCHITECTURE_NAME,
    resolve_strict_stack_validation,
)

_SCHEDULER_FQCN_DOT = "dllm_plugin.runtime_scheduler.DllmRuntimeScheduler"
_SCHEDULER_FQCN_COLON = "dllm_plugin.runtime_scheduler:DllmRuntimeScheduler"
_WORKER_FQCN_DOT = "dllm_plugin.runtime_worker.DllmRuntimeWorker"


def _normalize_fqcn(value: str) -> str:
    return value.replace(":", ".")


def _resolve_class_from_qualname(qualname: str, *, role: str) -> type[Any]:
    """Resolve a dotted (or colon-separated) qualname to a class object.

    Mirrors :func:`vllm.utils.import_utils.resolve_obj_by_qualname` so worker
    validation inspects the same object vLLM instantiates.
    """

    normalized = _normalize_fqcn(qualname.strip())
    if not normalized or "." not in normalized:
        raise ValueError(
            f"{role} must be a dotted qualname (e.g. dllm_plugin.Worker); "
            f"got={qualname!r}",
        )
    module_name, obj_name = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, obj_name)
    if not isinstance(obj, type):
        raise TypeError(
            f"{role} resolved to {type(obj).__name__}, expected a class",
        )
    return obj


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
    strict: bool | None = None,
) -> None:
    """Fail fast when scheduler/worker/model stack is incompatible for dLLM.

    ``caller`` is appended to raised :exc:`ValueError` messages as
    ``(context: '<caller>')`` so logs distinguish scheduler vs worker vs model
    bootstrap paths.

    When ``strict`` is ``None``, effective strictness comes from
    :func:`~dllm_plugin.config.resolve_strict_stack_validation` (see
    :data:`~dllm_plugin.config.DLLM_STRICT_STACK_VALIDATION_ENV_VAR`).

    Scheduler and worker must resolve to the concrete adapter classes
    (``DllmRuntimeScheduler``, ``DllmRuntimeWorker``); subclasses are not
    accepted unless this check is extended.
    """

    def _ctx() -> str:
        return f" (context: {caller!r})"

    if not resolve_strict_stack_validation(strict):
        return

    archs = _get_model_architectures(vllm_config)
    if not _is_dllm_model_architecture(vllm_config):
        raise ValueError(
            "dLLM runtime adapters require a dLLM-compatible model architecture "
            f"(got architectures={archs!r}); expected one of "
            f"{(LLADA2_ARCHITECTURE_NAME, DLLM_MOCK_STACK_MODEL_ID)!r}"
            f"{_ctx()}",
        )

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is None:
        raise ValueError(
            f"missing scheduler_config in vLLM config for dLLM runtime stack{_ctx()}",
        )
    try:
        scheduler_cls = scheduler_config.get_scheduler_cls()
    except Exception as exc:
        raise ValueError(
            "failed to resolve scheduler class for dLLM runtime stack; use "
            "--scheduler-cls dllm_plugin.Scheduler "
            "(or dllm_plugin.runtime_scheduler.DllmRuntimeScheduler)"
            f"{_ctx()}",
        ) from exc
    scheduler_fqcn = _normalize_fqcn(
        f"{scheduler_cls.__module__}.{scheduler_cls.__name__}",
    )
    if scheduler_fqcn != _normalize_fqcn(_SCHEDULER_FQCN_DOT):
        raise ValueError(
            "invalid scheduler class for dLLM runtime stack: "
            f"got={scheduler_fqcn!r} expected one of "
            f"{(_SCHEDULER_FQCN_DOT, _SCHEDULER_FQCN_COLON)!r}; "
            "pass --scheduler-cls dllm_plugin.Scheduler "
            "(or dllm_plugin.runtime_scheduler.DllmRuntimeScheduler)"
            f"{_ctx()}",
        )

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        raise ValueError(
            f"missing parallel_config in vLLM config for dLLM runtime stack{_ctx()}",
        )
    worker_cls_raw = getattr(parallel_config, "worker_cls", "")
    if not isinstance(worker_cls_raw, str):
        raise ValueError(
            "parallel_config.worker_cls must be a qualname string "
            f"(got type {type(worker_cls_raw).__name__}){_ctx()}",
        )
    if worker_cls_raw.strip() == "auto":
        raise ValueError(
            "parallel_config.worker_cls is still 'auto'; "
            "platform check_and_update_config should set a concrete worker "
            "before dLLM stack validation"
            f"{_ctx()}",
        )
    try:
        worker_cls = _resolve_class_from_qualname(
            worker_cls_raw,
            role="parallel_config.worker_cls",
        )
    except Exception as exc:
        raise ValueError(
            "failed to resolve worker class for dLLM runtime stack; use "
            "--worker-cls dllm_plugin.Worker "
            f"(or {_WORKER_FQCN_DOT!r})"
            f"{_ctx()}",
        ) from exc
    worker_fqcn = _normalize_fqcn(f"{worker_cls.__module__}.{worker_cls.__name__}")
    if worker_fqcn != _normalize_fqcn(_WORKER_FQCN_DOT):
        raise ValueError(
            "invalid worker class for dLLM runtime stack: "
            f"got={worker_fqcn!r} expected {_WORKER_FQCN_DOT!r}; "
            "pass --worker-cls dllm_plugin.Worker "
            f"(or {_WORKER_FQCN_DOT!r})"
            f"{_ctx()}",
        )


def assert_runtime_worker_v2_model_runner(
    *,
    use_v2_model_runner: bool,
    caller: str,
    strict: bool | None = None,
) -> None:
    """Require v2 model runner for ``DllmRuntimeWorker`` when strict.

    Issue **#10** treats ``VLLM_USE_V2_MODEL_RUNNER=1`` as first-class for the mock
    stack. When strict validation is off (env or explicit ``strict=False``), emit a
    **warning** instead of raising so monkeypatch / debug workflows keep working.
    """

    if use_v2_model_runner:
        return

    def _ctx() -> str:
        return f" (context: {caller!r})"

    if not resolve_strict_stack_validation(strict):
        warnings.warn(
            "DllmRuntimeWorker expects VLLM_USE_V2_MODEL_RUNNER=1 for the dLLM plugin "
            "mock stack; v1 model-runner paths are unsupported and may fail later."
            f"{_ctx()}",
            stacklevel=2,
        )
        return

    raise ValueError(
        "DllmRuntimeWorker requires the v2 model runner for the dLLM plugin stack; "
        "set VLLM_USE_V2_MODEL_RUNNER=1 (see docs/OPERATOR_LLaDA2.md, issue #10)"
        f"{_ctx()}",
    )


__all__ = ["assert_compatible_stack", "assert_runtime_worker_v2_model_runner"]
