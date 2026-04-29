# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU CI smoke for mock-stack vLLM wiring (catches API / qualname drift without CUDA).

The GPU-gated test in ``test_vllm_mock_integration.py`` exercises full
``LLM.generate``. This module builds a real ``VllmConfig`` via
``EngineArgs.create_engine_config`` so PR CI (vLLM extra on ubuntu-latest) still
validates scheduler/worker resolution and ``assert_compatible_stack`` against
upstream releases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("vllm")


def test_mock_stack_engine_args_resolve_paths_and_strict_validation_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from dllm_plugin.validation import assert_compatible_stack

    model_dir = Path(__file__).parent / "fixtures" / "mock_llada2_hf_config"

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    engine_args = EngineArgs(
        model=str(model_dir),
        tokenizer=str(model_dir),
        skip_tokenizer_init=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=128,
        max_num_seqs=1,
        load_format="dummy",
        scheduler_cls="dllm_plugin:Scheduler",
        worker_cls="dllm_plugin:Worker",
    )
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.LLM_CLASS,
    )

    sched_cls = vllm_config.scheduler_config.get_scheduler_cls()
    assert sched_cls.__name__ == "DllmRuntimeScheduler"

    wc = vllm_config.parallel_config.worker_cls
    assert isinstance(wc, str)
    worker_resolved = resolve_obj_by_qualname(wc)
    assert worker_resolved.__name__ == "DllmRuntimeWorker"

    assert_compatible_stack(vllm_config, caller="test_cpu_smoke")
