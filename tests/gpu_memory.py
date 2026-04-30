# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for GPU integration tests (Helm CI vs local dev)."""

from __future__ import annotations

import os


def gpu_memory_utilization() -> float:
    """KV-cache reservation fraction for ``LLM(gpu_memory_utilization=...)``.

    Helm/shared GPU nodes often have far less free VRAM than a dedicated card.
    Override with env ``DLLM_TEST_GPU_MEMORY_UTILIZATION`` (e.g. ``0.08``).
    Default ``0.9`` matches typical local GPU smoke runs.
    """

    return float(os.environ.get("DLLM_TEST_GPU_MEMORY_UTILIZATION", "0.9"))


def kv_cache_memory_bytes() -> int | None:
    """Fixed KV budget for integration tests using ``load_format='dummy'``.

    With ``load_format='dummy'``, memory profiling can disagree with real forwards;
    passing ``kv_cache_memory_bytes`` skips that path (upstream
    ``determine_available_memory``).

    Set ``DLLM_TEST_KV_CACHE_MEMORY_BYTES`` to an integer (bytes), or ``none`` to
    omit and use full profiling (needs a real-sized model forward).
    """

    raw = os.environ.get("DLLM_TEST_KV_CACHE_MEMORY_BYTES")
    if raw is None or raw == "":
        return 256 * 1024 * 1024
    if raw.strip().lower() in {"none", "null"}:
        return None
    return int(raw)
