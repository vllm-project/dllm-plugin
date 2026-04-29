# SPDX-License-Identifier: Apache-2.0
"""Test-local alias: CI/docs historically referenced ``vllm_dllm_plugin``."""

from __future__ import annotations

import importlib
import sys


def _install_alias() -> None:
    import dllm_plugin

    sys.modules.setdefault("vllm_dllm_plugin", dllm_plugin)
    submods = (
        "config",
        "grammar_utils",
        "models.mock_llada2",
        "models",
        "remasking",
        "runtime_scheduler",
        "runtime_worker",
        "scheduler",
        "validation",
        "worker",
    )
    for name in submods:
        fq = f"dllm_plugin.{name}"
        try:
            mod = importlib.import_module(fq)
        except ImportError:
            continue
        sys.modules.setdefault(f"vllm_dllm_plugin.{name}", mod)


_install_alias()
