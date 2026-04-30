# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM dLLM plugin: block-based diffusion LM support."""

from __future__ import annotations

import importlib.util
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dllm_plugin.scheduler import DllmScheduler
from dllm_plugin.validation import assert_compatible_stack
from dllm_plugin.worker import DllmWorker

# Do **not** import ``runtime_scheduler`` / ``runtime_worker`` at package import time.
# Their top-level ``vllm`` imports must run only after submodules such as
# ``dllm_plugin.gpu_model_runner`` have finished loading; eager imports here
# caused circular import failures for GPU integration tests.


def __getattr__(name: str):
    if name in ("DllmRuntimeScheduler", "Scheduler"):
        from dllm_plugin.runtime_scheduler import DllmRuntimeScheduler

        return DllmRuntimeScheduler
    if name in ("DllmRuntimeWorker", "Worker"):
        from dllm_plugin.runtime_worker import DllmRuntimeWorker

        return DllmRuntimeWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    __version__ = version("vllm-dllm-plugin")
except PackageNotFoundError:
    # No dist metadata (e.g. bare ``pytest`` on PYTHONPATH). Prefer ``uv sync`` /
    # editable install so ``importlib.metadata`` resolves the version.
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root=str(Path(__file__).resolve().parents[1]))
    except (ImportError, LookupError):
        __version__ = "0.0.0+unknown"


_logger = logging.getLogger(__name__)


def register_dllm() -> None:
    """Entry point for ``vllm.general_plugins`` (``dllm``).

    When ``vllm`` is importable, registers **two** architecture names with
    ``ModelRegistry``, both pointing at the same **mock** implementation for
    Phases 2–6 stack testing (issues #5 and #24):

    * :data:`~dllm_plugin.config.LLADA2_ARCHITECTURE_NAME` — placeholder
      until the real HF-mapped module ships (issue #12 / Phase 7).
    * :data:`~dllm_plugin.config.DLLM_MOCK_STACK_MODEL_ID` — explicit test id.

    Uses lazy ``"<module>:<Class>"`` registration so importing this package does
    not pull ``torch``/CUDA until the model class is needed.

    If ``vllm`` is not discoverable (``find_spec`` is ``None``), returns without
    registering. If the spec exists but importing ``ModelRegistry`` fails, logs
    DEBUG with ``exc_info`` and returns. (``find_spec`` can succeed when a full
    ``import vllm`` would still fail.)
    """
    if importlib.util.find_spec("vllm") is None:
        return

    try:
        from vllm import ModelRegistry
    except ImportError:
        _logger.debug(
            "vllm-dllm-plugin (dllm): vLLM spec found but import failed; "
            "skipping ModelRegistry registration.",
            exc_info=True,
        )
        return

    from dllm_plugin.config import (
        DLLM_MOCK_MODEL_CLASS_FQCN,
        DLLM_MOCK_STACK_MODEL_ID,
        LLADA2_ARCHITECTURE_NAME,
    )

    supported = ModelRegistry.get_supported_archs()
    for arch in (LLADA2_ARCHITECTURE_NAME, DLLM_MOCK_STACK_MODEL_ID):
        if arch in supported:
            _logger.debug(
                "dLLM plugin: architecture %r already registered, skipping",
                arch,
            )
            continue
        ModelRegistry.register_model(arch, DLLM_MOCK_MODEL_CLASS_FQCN)
        _logger.debug(
            "dLLM plugin: registered architecture %r -> %s",
            arch,
            DLLM_MOCK_MODEL_CLASS_FQCN,
        )


__all__ = [
    "DllmRuntimeScheduler",
    "DllmRuntimeWorker",
    "DllmScheduler",
    "DllmWorker",
    "Scheduler",
    "Worker",
    "assert_compatible_stack",
    "__version__",
    "register_dllm",
]
