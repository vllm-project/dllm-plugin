# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict stack validation for dLLM runtime wiring (issue #4)."""

from __future__ import annotations

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
_WORKER_FQCN_COLON = "dllm_plugin.runtime_worker:DllmRuntimeWorker"


def _normalize_fqcn(value: str) -> str:
    return value.replace(":", ".")


# Package-root alias ``dllm_plugin.Worker`` (vLLM resolves dotted qualnames only).
_WORKER_ACCEPT_NORMALIZED: frozenset[str] = frozenset(
    {
        _normalize_fqcn(_WORKER_FQCN_DOT),
        "dllm_plugin.Worker",
    },
)


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
    worker_cls = _normalize_fqcn(str(getattr(parallel_config, "worker_cls", "")))
    if worker_cls not in _WORKER_ACCEPT_NORMALIZED:
        raise ValueError(
            "invalid worker class for dLLM runtime stack: "
            f"got={worker_cls!r}; pass --worker-cls dllm_plugin.Worker "
            f"(or {_WORKER_FQCN_DOT!r})"
            f"{_ctx()}",
        )


__all__ = ["assert_compatible_stack"]
