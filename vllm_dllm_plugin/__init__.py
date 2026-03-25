# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM dLLM plugin: block-based diffusion LM support (skeleton)."""

from __future__ import annotations

import importlib.util
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

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

    Skeleton: does **not** register models, schedulers, or workers. If a ``vllm``
    **distribution is discoverable** on ``sys.path`` (``importlib.util.find_spec``),
    logs a DEBUG message and returns. If ``vllm`` is absent, returns with no log.

    ``find_spec`` can succeed without a working ``import vllm`` (e.g. broken
    install); we intentionally avoid importing vLLM here. Safe to call from tests
    without vLLM installed (no-op).
    """
    if importlib.util.find_spec("vllm") is None:
        return
    # Stub: avoid importing vllm here (expensive at load time). Add lazy/targeted
    # imports when model/scheduler/worker registration is implemented.
    _logger.debug(
        "vllm-dllm-plugin (dllm): vLLM is discoverable on sys.path but "
        "register_dllm() is still a skeleton — no models, schedulers, or workers "
        "registered yet.",
    )
