"""vLLM dLLM plugin: block-based diffusion LM support (skeleton)."""

from __future__ import annotations

import importlib.util
import logging

__version__ = "0.1.0"

_logger = logging.getLogger(__name__)


def register() -> None:
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
        "vllm-dllm-plugin (dllm): vLLM is discoverable on sys.path but register() "
        "is still a skeleton — no models, schedulers, or workers registered yet.",
    )
