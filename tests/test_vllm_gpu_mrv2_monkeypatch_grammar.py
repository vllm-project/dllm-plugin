# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU integration: monkeypatch ``GPUModelRunner`` to ``DllmGPUModelRunner`` + regex SO.

Intended for Helm GPU CI (see ``tools/helm/dllm-plugin-gpu-test``).
On CPU-only runners tests skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gpu_memory import gpu_memory_utilization, kv_cache_memory_bytes

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

pytest.importorskip("transformers")


@pytest.fixture
def mock_llada2_model_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "mock_llada2_hf_config"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_injects_dllm_mrv2_via_monkeypatch_stock_worker(
    monkeypatch: pytest.MonkeyPatch,
    mock_llada2_model_dir: Path,
) -> None:
    """Patch ``GPUModelRunner`` before worker init; use stock ``Worker``.

    Disables strict stack validation so ``DllmRuntimeScheduler`` pairs with
    vLLM's ``GpuWorker`` (plan monkeypatch recipe).
    """
    import vllm.v1.worker.gpu.model_runner as gpu_mr
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from dllm_plugin.gpu_model_runner import DllmGPUModelRunner

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

    monkeypatch.setattr(gpu_mr, "GPUModelRunner", DllmGPUModelRunner)

    llm = LLM(
        model=str(mock_llada2_model_dir),
        tokenizer=str(mock_llada2_model_dir),
        skip_tokenizer_init=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization(),
        kv_cache_memory_bytes=kv_cache_memory_bytes(),
        load_format="dummy",
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="vllm.v1.worker.gpu_worker.Worker",
    )

    mr = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    assert type(mr).__name__ == "DllmGPUModelRunner"

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])],
        SamplingParams(max_tokens=1, temperature=0.0, detokenize=False),
    )
    assert outputs[0].outputs[0].token_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_dllm_stack_structured_output_regex_grammar(
    monkeypatch: pytest.MonkeyPatch,
    mock_llada2_model_dir: Path,
) -> None:
    """Production worker, tokenizer fixture, regex SO (bitmask path)."""
    from vllm import LLM, SamplingParams
    from vllm.config import StructuredOutputsConfig
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Avoid seeding ``spec_token_ids`` (length ``DRAFT_SIZE``): without a matching
    # ``speculative_config``, vLLM's structured-output bitmask buffer is sized for
    # AR-only (``max_num_seqs`` rows), but grammar fill walks one row per scheduled
    # spec token and overflows. Skipping the seed keeps this test focused on regex
    # SO + plugin wiring; full block+draft + SO remains covered elsewhere.
    monkeypatch.setenv("VLLM_DLLM_SKIP_FIRST_BLOCK_SEED", "1")

    llm = LLM(
        model=str(mock_llada2_model_dir),
        tokenizer=str(mock_llada2_model_dir),
        skip_tokenizer_init=False,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization(),
        kv_cache_memory_bytes=kv_cache_memory_bytes(),
        load_format="dummy",
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="dllm_plugin.Worker",
        structured_outputs_config=StructuredOutputsConfig(backend="auto"),
        async_scheduling=False,
    )

    sp = SamplingParams(
        max_tokens=4,
        temperature=0.0,
        detokenize=False,
        structured_outputs=StructuredOutputsParams(regex=r"[012]+"),
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])],
        sampling_params=sp,
    )
    assert outputs
    assert outputs[0].outputs
    assert outputs[0].outputs[0].token_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_runtime_adapters_strict_stack_regex_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    mock_llada2_model_dir: Path,
) -> None:
    """Full runtime adapters + regex SO with strict stack validation (default on).

    Unlike :func:`test_gpu_injects_dllm_mrv2_via_monkeypatch_stock_worker`, this does
    not set ``VLLM_DLLM_STRICT_STACK_VALIDATION=0``. Scheduler/worker use dotted FQCNs
    (not ``dllm_plugin.Scheduler`` / ``dllm_plugin.Worker`` aliases).

    Uses ``VLLM_DLLM_SKIP_FIRST_BLOCK_SEED=1`` for the same grammar-bitmask sizing
    rationale as :func:`test_gpu_dllm_stack_structured_output_regex_grammar`.
    """
    from vllm import LLM, SamplingParams
    from vllm.config import StructuredOutputsConfig
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams

    monkeypatch.delenv("VLLM_DLLM_STRICT_STACK_VALIDATION", raising=False)

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DLLM_SKIP_FIRST_BLOCK_SEED", "1")

    llm = LLM(
        model=str(mock_llada2_model_dir),
        tokenizer=str(mock_llada2_model_dir),
        skip_tokenizer_init=False,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization(),
        kv_cache_memory_bytes=kv_cache_memory_bytes(),
        load_format="dummy",
        scheduler_cls="dllm_plugin.runtime_scheduler.DllmRuntimeScheduler",
        worker_cls="dllm_plugin.runtime_worker.DllmRuntimeWorker",
        structured_outputs_config=StructuredOutputsConfig(backend="auto"),
        async_scheduling=False,
    )

    sp = SamplingParams(
        max_tokens=4,
        temperature=0.0,
        detokenize=False,
        structured_outputs=StructuredOutputsParams(regex=r"[012]+"),
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])],
        sampling_params=sp,
    )
    assert outputs[0].outputs[0].token_ids
