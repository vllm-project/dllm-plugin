# SPDX-License-Identifier: Apache-2.0
"""LLaDA2.0 benchmarking tests using GuideLLM.

Measures performance metrics for real LLaDA2.0-mini model:
- TPS (Tokens Per Second)
- TTFT (Time To First Token)
- ITL (Inter-Token Latency)
- E2E (End-to-End latency)

Tests both synchronous and streaming modes using GuideLLM CLI.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest
import requests  # type: ignore[import-untyped]

from tests.gpu_memory import gpu_memory_utilization, kv_cache_memory_bytes

pytestmark = pytest.mark.dllm_gpu_benchmark


@pytest.fixture
def llada2_mini_model_dir():
    """HuggingFace model directory for LLaDA2.0-mini."""
    model_id = "inclusionAI/LLaDA2.0-mini"
    try:
        from transformers import AutoConfig  # type: ignore[import-untyped]

        AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return model_id
    except Exception:
        # Fallback to local fixture if HF download fails
        return Path(__file__).parent / "fixtures" / "llada2_mini"


@pytest.fixture
def vllm_server(
    monkeypatch: pytest.MonkeyPatch,
    llada2_mini_model_dir,
    tmp_path: Path,
):
    """Start vLLM server with LLaDA2.0-mini for benchmarking."""
    import os

    port = 8768
    log_file = tmp_path / "vllm_server.log"

    # Build environment for subprocess
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "dllm"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_DLLM_USE_MOCK_MODEL"] = "0"

    # Start vLLM server
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [
                "vllm",
                "serve",
                str(llada2_mini_model_dir),
                "--tokenizer",
                str(llada2_mini_model_dir),
                "--trust-remote-code",
                "--model-impl",
                "dllm_plugin.models.llada2:LLaDA2ForCausalLM",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--enforce-eager",
                "--max-model-len",
                "512",
                "--max-num-seqs",
                "4",
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                str(gpu_memory_utilization()),
                "--kv-cache-memory-bytes",
                str(kv_cache_memory_bytes()),
                "--scheduler-cls",
                "dllm_plugin.Scheduler",
                "--worker-cls",
                "dllm_plugin.Worker",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )

    try:
        # Wait for server health (timeout 180s)
        healthy = False
        for _ in range(180):
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if resp.status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not healthy:
            proc.terminate()
            proc.wait()
            with open(log_file) as f:
                print(f"Server logs:\n{f.read()}")
            pytest.fail("vLLM server health check timeout")

        yield f"http://127.0.0.1:{port}"

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_llada2_guidellm_benchmark(vllm_server: str):
    """Benchmark LLaDA2.0 using GuideLLM CLI with synchronous mode and streaming.

    Measures performance metrics:
    - TTFT (Time To First Token)
    - ITL (Inter-Token Latency)
    - TPS (Tokens Per Second)
    - E2E (End-to-End latency)
    """
    # Run GuideLLM benchmark with synchronous mode and streaming enabled
    result = subprocess.run(
        [
            "guidellm",
            "benchmark",
            "--target",
            vllm_server,
            "--model",
            "llada2",
            "--rate-type",
            "synchronous",
            "--stream",  # Enable streaming
            "--max-seconds",
            "180",  # 3 minutes total
            "--data",
            "synthetic-256-64",  # 256 input tokens, 64 output tokens
        ],
        capture_output=True,
        text=True,
    )

    # Print full output
    print("\n" + "=" * 80)
    print("GuideLLM Benchmark Results (Synchronous + Streaming)")
    print("=" * 80)
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:")
        print(result.stderr)
    print("=" * 80)

    # Validate benchmark completed successfully
    assert result.returncode == 0, f"GuideLLM benchmark failed: {result.stderr}"
