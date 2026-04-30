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


def test_dllm_gpu_model_runner_overrides_phase_two_hooks() -> None:
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from dllm_plugin.gpu_model_runner import (
        DllmGPUModelRunner,
        _GPUModelRunnerPrepareInputsFork,
    )

    assert issubclass(DllmGPUModelRunner, _GPUModelRunnerPrepareInputsFork)
    assert (
        _GPUModelRunnerPrepareInputsFork.prepare_inputs
        is not GPUModelRunner.prepare_inputs
    )
    assert (
        DllmGPUModelRunner.prepare_inputs
        is _GPUModelRunnerPrepareInputsFork.prepare_inputs
    )
    assert DllmGPUModelRunner.get_expand_idx_mapping_block_size is not (
        _GPUModelRunnerPrepareInputsFork.get_expand_idx_mapping_block_size
    )
    assert DllmGPUModelRunner.sample is not GPUModelRunner.sample
    assert DllmGPUModelRunner.sample_tokens is not GPUModelRunner.sample_tokens
    assert DllmGPUModelRunner.execute_model is not GPUModelRunner.execute_model


def test_dllm_runtime_worker_wraps_init_device_for_dllm_runner() -> None:
    pytest.importorskip("vllm")
    import dllm_plugin.runtime_worker as rw

    src = inspect.getsource(rw.DllmRuntimeWorker.init_device)
    assert "DllmGPUModelRunner" in src
    assert "super().init_device()" in src
