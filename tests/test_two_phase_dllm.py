# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guardrails for two-phase v2 alignment: ``execute_model`` → ``None``, then
``sample_tokens``.

The historical gap: a worker hook that only ran when ``execute_model`` returned a
``ModelRunnerOutput`` with ``sampled_token_ids`` never saw the stock v2 path, because
``GPUModelRunner.execute_model`` returns ``None`` on the last pipeline stage and defers
sampling to ``sample_tokens(grammar_output)``.
"""

from __future__ import annotations

import inspect

import pytest


def test_upstream_v2_runner_executes_forward_then_defers_sampling() -> None:
    """Document/stabilize the core contract dLLM aligns with."""

    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu import model_runner as mr

    execute_src = inspect.getsource(mr.GPUModelRunner.execute_model)
    assert "self.execute_model_state" in execute_src
    assert "return None" in execute_src

    sample_src = inspect.getsource(mr.GPUModelRunner.sample_tokens)
    assert "grammar_output" in sample_src
    assert "self.sample(" in sample_src


def test_llada2_model_state_implements_mrv2_hooks() -> None:
    """LLaDA2ModelState provides custom_sample and take_draft_token_ids."""
    pytest.importorskip("vllm")
    from dllm_plugin.models.llada2_model_state import LLaDA2ModelState

    assert hasattr(LLaDA2ModelState, "custom_sample")
    assert hasattr(LLaDA2ModelState, "take_draft_token_ids")
    assert hasattr(LLaDA2ModelState, "before_step")
    assert hasattr(LLaDA2ModelState, "prepare_attn")
    assert hasattr(LLaDA2ModelState, "remove_request")


def test_llada2_model_provides_get_model_state_cls() -> None:
    """LLaDA2 model registers its ModelState via get_model_state_cls()."""
    pytest.importorskip("vllm")
    from dllm_plugin.models.llada2 import LLaDA2ForCausalLM
    from dllm_plugin.models.llada2_model_state import LLaDA2ModelState

    cls = LLaDA2ForCausalLM.get_model_state_cls()
    assert cls is LLaDA2ModelState


def test_dllm_runtime_worker_does_not_set_model_runner_cls() -> None:
    """Worker delegates to ModelState (no _model_runner_cls override)."""
    pytest.importorskip("vllm")
    import dllm_plugin.runtime_worker as rw

    src = inspect.getsource(rw.DllmRuntimeWorker.__init__)
    assert "_model_runner_cls" not in src


def test_dllm_runtime_worker_inherits_execute_model() -> None:
    """``execute_model`` is stock v2 worker + runner; no redundant ``super()`` shim."""

    pytest.importorskip("vllm")
    from dllm_plugin.runtime_worker import DllmRuntimeWorker

    assert "execute_model" not in DllmRuntimeWorker.__dict__
