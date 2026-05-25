# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for lm-eval harness.

The import test runs on CPU (no GPU needed). The GSM8K sanity test
requires GPU and model weights.
"""

from __future__ import annotations

import pytest


class TestLMEvalHarness:
    def test_harness_importable(self):
        """Verify the eval harness module can be imported."""
        from dllm_plugin.eval_harness import DllmPluginEvalHarness

        assert DllmPluginEvalHarness is not None

    def test_harness_registered(self):
        """Verify the model is registered with lm-eval."""
        pytest.importorskip("lm_eval")

        from lm_eval.api.registry import MODEL_REGISTRY

        import dllm_plugin.eval_harness  # noqa: F401

        assert "dllm_plugin_eval" in MODEL_REGISTRY

    def test_loglikelihood_raises(self):
        """loglikelihood should raise NotImplementedError for diffusion models."""
        pytest.importorskip("lm_eval")

        from lm_eval.api.registry import MODEL_REGISTRY

        import dllm_plugin.eval_harness  # noqa: F401

        cls = MODEL_REGISTRY["dllm_plugin_eval"]
        assert hasattr(cls, "loglikelihood")

    def test_task_config_exists(self):
        """Verify GSM8K task config is present."""
        from pathlib import Path

        task_file = (
            Path(__file__).parent.parent
            / "evaluations"
            / "tasks"
            / "gsm8k"
            / "gsm8k-llada-mini.yaml"
        )
        assert task_file.exists(), f"Task config not found: {task_file}"
