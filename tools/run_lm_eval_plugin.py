#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run lm-eval with the dllm-plugin eval harness.

Importing this module registers ``dllm_plugin_eval`` with lm-eval before
calling ``cli_evaluate()``.

Usage::

    python tools/run_lm_eval_plugin.py \
        --model dllm_plugin_eval \
        --model_args model_path=inclusionAI/LLaDA2.0-mini,gen_length=2048 \
        --tasks gsm8k_llada_mini \
        --include_path evaluations/tasks \
        --confirm_run_unsafe_code
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from lm_eval.__main__ import cli_evaluate  # noqa: E402

import dllm_plugin.eval_harness  # noqa: F401, E402 — triggers @register_model

if __name__ == "__main__":
    cli_evaluate()
