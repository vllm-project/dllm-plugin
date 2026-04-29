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
