# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM dLLM plugin: block-based diffusion LM support."""

from __future__ import annotations

import importlib.util
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dllm_plugin.scheduler import DllmScheduler
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
    if name == "assert_compatible_stack":
        from dllm_plugin.validation import assert_compatible_stack

        return assert_compatible_stack
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

_non_causal_flag_applied = False


def _apply_non_causal_flag() -> None:
    """Set attention_config.use_non_causal=True for LLaDA2 models.

    The vLLM fork reads this flag to set causal=False in
    CommonAttentionMetadata and to disable chunked prefill /
    prefix caching for non-causal models.
    """
    global _non_causal_flag_applied
    if _non_causal_flag_applied:
        return
    _non_causal_flag_applied = True

    try:
        from vllm.config import VllmConfig
    except ImportError:
        return

    from dllm_plugin.config import (
        LLADA2_ARCHITECTURE_NAME,
        LLADA2_HF_ARCHITECTURE_NAME,
    )

    _dllm_archs = {LLADA2_ARCHITECTURE_NAME, LLADA2_HF_ARCHITECTURE_NAME}
    _original_post_init = VllmConfig.__post_init__

    def _patched_post_init(self: VllmConfig) -> None:
        _original_post_init(self)

        model_config = getattr(self, "model_config", None)
        if model_config is None:
            return
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            return
        archs = set(getattr(hf_config, "architectures", []) or [])
        if not archs.intersection(_dllm_archs):
            return

        attention_config = getattr(self, "attention_config", None)
        if attention_config is not None:
            attention_config.use_non_causal = True
            _logger.info("dLLM plugin: set use_non_causal=True for LLaDA2")

    VllmConfig.__post_init__ = _patched_post_init


def register_dllm() -> None:
    """Entry point for ``vllm.general_plugins`` (``dllm``).

    **Phase 7 update:** Registers architecture names with ``ModelRegistry``:

    * :data:`~dllm_plugin.config.LLADA2_ARCHITECTURE_NAME` — Production LLaDA2.0
      model with MoE and block-style attention (Phase 7 / issue #12). Use
      ``VLLM_DLLM_USE_MOCK_MODEL=1`` to override with mock for testing.
    * :data:`~dllm_plugin.config.DLLM_MOCK_STACK_MODEL_ID` — Explicit mock model
      for Phases 2–6 stack testing (always uses mock implementation).

    Environment variables:
    * ``VLLM_DLLM_USE_MOCK_MODEL``: If set to ``1``/``true``/``yes``/``on``,
      registers LLADA2_ARCHITECTURE_NAME to the mock model instead of real model.
      Useful for testing Phases 2-6 behavior with real model disabled.

    Uses lazy ``"<module>:<Class>"`` registration so importing this package does
    not pull ``torch``/CUDA until the model class is needed.

    If ``vllm`` is not discoverable (``find_spec`` is ``None``), returns without
    registering. If the spec exists but importing ``ModelRegistry`` fails, logs
    DEBUG with ``exc_info`` and returns. (``find_spec`` can succeed when a full
    ``import vllm`` would still fail.)

    When running against the vLLM fork (dllm-fork branch), no runtime patches
    are needed — the fork handles chunked prefill, prefix caching, the draft
    hook gate, and non-causal attention natively.
    """
    _logger.debug("dLLM plugin: register_dllm() called")

    if importlib.util.find_spec("vllm") is None:
        _logger.debug("dLLM plugin: vllm not found, skipping registration")
        return

    try:
        from vllm import ModelRegistry

        _logger.debug("dLLM plugin: ModelRegistry imported successfully")
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
        LLADA2_HF_ARCHITECTURE_NAME,
        LLADA2_REAL_MODEL_CLASS_FQCN,
    )

    # Determine which model to use for LLADA2_ARCHITECTURE_NAME
    use_mock_raw = os.environ.get("VLLM_DLLM_USE_MOCK_MODEL", "").strip().lower()
    use_mock_model = use_mock_raw in {"1", "true", "yes", "on"}

    if use_mock_model:
        llada2_model_class = DLLM_MOCK_MODEL_CLASS_FQCN
        _logger.info(
            "dLLM plugin: VLLM_DLLM_USE_MOCK_MODEL=1, using mock model for %s",
            LLADA2_ARCHITECTURE_NAME,
        )
    else:
        llada2_model_class = LLADA2_REAL_MODEL_CLASS_FQCN
        _logger.info(
            "dLLM plugin: Using real LLaDA2.0 model for %s (Phase 7)",
            LLADA2_ARCHITECTURE_NAME,
        )

    supported = ModelRegistry.get_supported_archs()

    # Register LLADA2_ARCHITECTURE_NAME (real or mock based on env var)
    if LLADA2_ARCHITECTURE_NAME not in supported:
        ModelRegistry.register_model(LLADA2_ARCHITECTURE_NAME, llada2_model_class)
        _logger.debug(
            "dLLM plugin: registered architecture %r -> %s",
            LLADA2_ARCHITECTURE_NAME,
            llada2_model_class,
        )
    else:
        _logger.debug(
            "dLLM plugin: architecture %r already registered, skipping",
            LLADA2_ARCHITECTURE_NAME,
        )

    # Register LLADA2_HF_ARCHITECTURE_NAME (HuggingFace naming convention)
    # Points to same implementation as LLADA2_ARCHITECTURE_NAME
    if LLADA2_HF_ARCHITECTURE_NAME not in supported:
        ModelRegistry.register_model(LLADA2_HF_ARCHITECTURE_NAME, llada2_model_class)
        _logger.debug(
            "dLLM plugin: registered HF architecture %r -> %s",
            LLADA2_HF_ARCHITECTURE_NAME,
            llada2_model_class,
        )
    else:
        _logger.debug(
            "dLLM plugin: HF architecture %r already registered, skipping",
            LLADA2_HF_ARCHITECTURE_NAME,
        )

    # Register DLLM_MOCK_STACK_MODEL_ID (always mock)
    if DLLM_MOCK_STACK_MODEL_ID not in supported:
        ModelRegistry.register_model(
            DLLM_MOCK_STACK_MODEL_ID, DLLM_MOCK_MODEL_CLASS_FQCN
        )
        _logger.debug(
            "dLLM plugin: registered architecture %r -> %s",
            DLLM_MOCK_STACK_MODEL_ID,
            DLLM_MOCK_MODEL_CLASS_FQCN,
        )
    else:
        _logger.debug(
            "dLLM plugin: architecture %r already registered, skipping",
            DLLM_MOCK_STACK_MODEL_ID,
        )

    # Set use_non_causal=True for LLaDA2 so the fork's
    # CommonAttentionMetadata uses causal=False and per-model
    # chunked prefill / prefix caching control kicks in.
    _apply_non_causal_flag()


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
