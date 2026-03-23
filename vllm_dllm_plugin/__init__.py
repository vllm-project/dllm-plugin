# Assisted by: Cursor AI
"""vLLM dLLM plugin: block-based diffusion LM support (skeleton)."""

from __future__ import annotations

__version__ = "0.1.0"


def register() -> None:
    """Entry point for ``vllm.general_plugins`` (``dllm``).

    Registers plugin components when vLLM is importable. Safe to call from tests
    without vLLM (no-op).
    """
    import importlib.util

    if importlib.util.find_spec("vllm") is None:
        return
    importlib.import_module("vllm")
    # Stub: future model / scheduler / worker registration lives here.
