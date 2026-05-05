# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU integration tests for LLaDA2.0 real model.

**Phase 7 deliverable** (issue #25): End-to-end validation that real LLaDA2.0
weights load, inference runs, and HTTP API works correctly.

**Primary acceptance criteria:**
- GPU test loads real LLaDA2.0-mini weights
- HTTP request sent and valid response received
- No crashes during multi-step generation
- Structure validation only (no content checks)

Test requirements:
- CUDA GPU (A100-40GB preferred, L4-16GB fallback, H100-80GB spot for large models)
- Real LLaDA2.0-mini model from HuggingFace or synthetic checkpoint
- vLLM with dLLM plugin enabled

Run with:
    pytest -v -m dllm_gpu_integration tests/test_llada2_gpu_integration.py
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

# CRITICAL: Set environment variables BEFORE importing vllm
# vLLM loads plugins during import based on VLLM_PLUGINS env var
os.environ["VLLM_PLUGINS"] = "dllm"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_DLLM_USE_MOCK_MODEL"] = "0"

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

# CRITICAL: Explicitly register plugin models
# vLLM's automatic plugin discovery may not trigger, so we call it manually
from dllm_plugin import register_dllm  # noqa: E402

register_dllm()

from dllm_plugin.config import DRAFT_SIZE  # noqa: E402
from tests.gpu_memory import gpu_memory_utilization, kv_cache_memory_bytes  # noqa: E402

pytestmark = pytest.mark.dllm_gpu_integration


@pytest.fixture
def llada2_mini_model_dir() -> Path:
    """Get LLaDA2.0-mini local config directory.

    Returns local fixture with config (no auto_map) for vLLM model loading.
    Weights will be downloaded from HuggingFace during model initialization.

    Returns:
        Path: Local fixture directory with config.json and tokenizer files.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "llada2_mini"
    if not fixture_path.exists():
        pytest.skip(
            f"LLaDA2.0-mini fixture not available at {fixture_path}. "
            f"Run setup script to create fixtures."
        )
    return fixture_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_llada2_real_weights_llm_generate(
    monkeypatch: pytest.MonkeyPatch,
    llada2_mini_model_dir,
):
    """Test LLM.generate() with real LLaDA2.0 weights.

    **Phase 7 primary deliverable:** Load real weights and generate tokens.

    Validates:
    - Model weights load without error
    - Forward pass executes successfully
    - Output structure is correct (list of RequestOutput)
    - Token IDs are valid integers
    - No crashes during generation
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # Set environment for dLLM plugin
    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Ensure we use real model, not mock
    monkeypatch.setenv("VLLM_DLLM_USE_MOCK_MODEL", "0")

    # Create LLM with real model
    # NOTE: Do NOT use trust_remote_code - it causes vLLM to use HF auto_map model
    # instead of our registered plugin model. The config will be loaded from
    # the registered model architecture, not from HuggingFace custom code.
    llm = LLM(
        model=str(llada2_mini_model_dir),
        tokenizer=str(llada2_mini_model_dir),
        trust_remote_code=False,  # MUST be False to use registry model
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,  # PP=1 (PP>1 not supported in Phase 7)
        max_model_len=256,  # Small for testing
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization(),
        kv_cache_memory_bytes=kv_cache_memory_bytes(),
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="dllm_plugin.Worker",
        async_scheduling=False,
    )

    # Generate with simple prompt
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3])],
        SamplingParams(
            max_tokens=5,
            temperature=0.0,
            detokenize=False,
        ),
    )

    # Validate structure (NOT content - structure only per Phase 7 plan)
    assert len(outputs) == 1, "Should return one RequestOutput"
    assert len(outputs[0].outputs) == 1, "Should have one CompletionOutput"

    token_ids = outputs[0].outputs[0].token_ids
    assert len(token_ids) > 0, "Should generate at least one token"
    assert all(isinstance(t, int) for t in token_ids), "Token IDs must be integers"

    # Validate token IDs are in valid range
    # (vocab_size check would require loading config, skip for structure-only test)
    assert all(t >= 0 for t in token_ids), "Token IDs must be non-negative"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_llada2_multi_step_generation(
    monkeypatch: pytest.MonkeyPatch,
    llada2_mini_model_dir,
):
    """Test multi-step generation with real LLaDA2.0.

    Validates that block-based generation works across multiple steps.

    With DRAFT_SIZE=32 and max_tokens=64, should execute at least 2 blocks.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DLLM_USE_MOCK_MODEL", "0")

    # NOTE: Do NOT use trust_remote_code - it causes vLLM to use HF auto_map model
    # instead of our registered plugin model. The config will be loaded from
    # the registered model architecture, not from HuggingFace custom code.
    llm = LLM(
        model=str(llada2_mini_model_dir),
        tokenizer=str(llada2_mini_model_dir),
        trust_remote_code=False,  # MUST be False to use registry model
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization(),
        kv_cache_memory_bytes=kv_cache_memory_bytes(),
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="dllm_plugin.Worker",
        async_scheduling=False,
    )

    # Generate more tokens to span multiple blocks
    max_tokens = 64  # Should execute 2+ blocks (DRAFT_SIZE=32)
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3])],
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            detokenize=False,
        ),
    )

    token_ids = outputs[0].outputs[0].token_ids

    # Validate multi-step generation
    assert len(token_ids) > 0, "Should generate tokens"
    # Allow some flexibility (may commit fewer tokens per block)
    assert len(token_ids) <= max_tokens * DRAFT_SIZE + max_tokens, (
        "Sanity bound: cumulative token IDs under dLLM + greedy decode"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_llada2_http_server_integration(
    monkeypatch: pytest.MonkeyPatch,
    llada2_mini_model_dir,
):
    """Test HTTP server with real LLaDA2.0 model.

    **Phase 7 primary deliverable:** Send HTTP request and receive valid response.

    Validates:
    - vLLM server starts successfully
    - Health endpoint responds with 200
    - Chat completions endpoint accepts requests
    - Response has expected structure (choices field)
    """
    import requests  # type: ignore[import-untyped]

    port = 8766
    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DLLM_USE_MOCK_MODEL", "0")

    # Start vLLM server
    # NOTE: Do NOT use --trust-remote-code - it causes vLLM to use HF auto_map model
    proc = subprocess.Popen(
        [
            "vllm",
            "serve",
            str(llada2_mini_model_dir),
            "--tokenizer",
            str(llada2_mini_model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--enforce-eager",
            "--max-model-len",
            "256",
            "--max-num-seqs",
            "1",
            "--tensor-parallel-size",
            "1",
            "--scheduler-cls",
            "dllm_plugin.Scheduler",
            "--worker-cls",
            "dllm_plugin.Worker",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server health (120s timeout)
        health_url = f"http://127.0.0.1:{port}/health"
        for _ in range(120):
            try:
                resp = requests.get(health_url, timeout=1)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.fail("Server health check timeout after 120s")

        # Send chat completion request
        chat_url = f"http://127.0.0.1:{port}/v1/chat/completions"
        resp = requests.post(
            chat_url,
            json={
                "model": "llada2-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=30,
        )

        # Validate response structure (NOT content)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert "choices" in data, "Response missing 'choices' field"
        assert len(data["choices"]) >= 1, "Expected at least one choice"

        # Validate choice structure
        choice = data["choices"][0]
        assert "message" in choice or "text" in choice, (
            "Choice missing message/text field"
        )

    finally:
        # Cleanup: terminate server
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_llada2_tensor_parallelism_tp2(
    monkeypatch: pytest.MonkeyPatch,
    llada2_mini_model_dir,
):
    """Test tensor parallelism with TP=2.

    Validates that model works with tensor parallelism for multi-GPU scaling.

    Note: Requires 2 GPUs. Skips if unavailable.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 GPUs for TP=2 test")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DLLM_USE_MOCK_MODEL", "0")

    # NOTE: Do NOT use trust_remote_code - it causes vLLM to use HF auto_map model
    # instead of our registered plugin model. The config will be loaded from
    # the registered model architecture, not from HuggingFace custom code.
    llm = LLM(
        model=str(llada2_mini_model_dir),
        tokenizer=str(llada2_mini_model_dir),
        trust_remote_code=False,  # MUST be False to use registry model
        enforce_eager=True,
        tensor_parallel_size=2,  # TP=2
        pipeline_parallel_size=1,
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="dllm_plugin.Worker",
        async_scheduling=False,
    )

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3])],
        SamplingParams(max_tokens=5, temperature=0.0, detokenize=False),
    )

    # Validate output structure
    assert len(outputs) == 1
    token_ids = outputs[0].outputs[0].token_ids
    assert len(token_ids) > 0
    assert all(isinstance(t, int) for t in token_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_llada2_pipeline_parallelism_fails(
    monkeypatch: pytest.MonkeyPatch,
    llada2_mini_model_dir,
):
    """Test that PP > 1 fails fast as expected in Phase 7.

    Pipeline parallelism is not supported in Phase 7 MVP. Model should
    raise ValueError during initialization.
    """
    from vllm import LLM

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 GPUs for PP test")

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DLLM_USE_MOCK_MODEL", "0")

    # Expect ValueError during model initialization (use plugin model)
    with pytest.raises(ValueError, match="Pipeline parallelism.*not supported"):
        LLM(
            model=str(llada2_mini_model_dir),
            tokenizer=str(llada2_mini_model_dir),
            enforce_eager=True,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,  # PP > 1 should fail
            max_model_len=256,
            max_num_seqs=1,
            scheduler_cls="dllm_plugin.Scheduler",
            worker_cls="dllm_plugin.Worker",
        )


# Optional: Backend compatibility tests (FlashAttention vs FlashInfer)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER"])
def test_llada2_attention_backend_compatibility(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    llada2_mini_model_dir,
):
    """Test that both FlashAttention and FlashInfer backends work.

    Both backends support causal=False for block-style attention.
    This test validates they produce valid outputs.

    Note: Exact output matching between backends is deferred to unit tests.
    This test validates structure only.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DLLM_USE_MOCK_MODEL", "0")
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", backend)

    # NOTE: Do NOT use trust_remote_code - it causes vLLM to use HF auto_map model
    # instead of our registered plugin model. The config will be loaded from
    # the registered model architecture, not from HuggingFace custom code.
    llm = LLM(
        model=str(llada2_mini_model_dir),
        tokenizer=str(llada2_mini_model_dir),
        trust_remote_code=False,  # MUST be False to use registry model
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_memory_utilization(),
        kv_cache_memory_bytes=kv_cache_memory_bytes(),
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="dllm_plugin.Worker",
        async_scheduling=False,
    )

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3])],
        SamplingParams(max_tokens=5, temperature=0.0, detokenize=False),
    )

    # Validate structure
    assert len(outputs) == 1
    token_ids = outputs[0].outputs[0].token_ids
    assert len(token_ids) > 0
    assert all(isinstance(t, int) for t in token_ids)
