# SPDX-License-Identifier: Apache-2.0
"""LLaDA2.0 benchmarking tests using GuideLLM.

Measures performance metrics for real LLaDA2.0-mini model:
- TPS (Tokens Per Second)
- TTFT (Time To First Token)
- ITL (Inter-Token Latency)
- E2E (End-to-End latency)

Tests both synchronous and streaming modes.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

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

        AutoConfig.from_pretrained(model_id)
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
                "--async-scheduling",
                "false",
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


def benchmark_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    stream: bool = False,
) -> dict[str, Any]:
    """Send a single benchmark request and measure metrics.

    Returns:
        dict with keys: ttft, itl_avg, e2e, tokens_generated, tps
    """
    start_time = time.perf_counter()
    first_token_time = None
    token_times: list[float] = []
    tokens_generated = 0
    response_text = ""

    if stream:
        # Streaming mode
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "llada2",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
            stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]  # Remove "data: " prefix
            if data_str.strip() == "[DONE]":
                break

            chunk_time = time.perf_counter()
            try:
                data = json.loads(data_str)
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "text" in choice and choice["text"]:
                        if first_token_time is None:
                            first_token_time = chunk_time
                        else:
                            token_times.append(chunk_time)
                        tokens_generated += len(choice["text"].split())
                        response_text += choice["text"]
            except json.JSONDecodeError:
                continue
    else:
        # Synchronous mode
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "llada2",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
        )
        resp.raise_for_status()
        end_time = time.perf_counter()

        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            response_text = data["choices"][0]["text"]
            if "usage" in data:
                tokens_generated = data["usage"].get("completion_tokens", 0)
            else:
                # Fallback: count tokens in response
                tokens_generated = len(response_text.split())

        # For synchronous, we don't have TTFT/ITL, only E2E
        first_token_time = end_time
        e2e = end_time - start_time
        return {
            "ttft": None,  # Not available in sync mode
            "itl_avg": None,  # Not available in sync mode
            "e2e": e2e,
            "tokens_generated": tokens_generated,
            "tps": tokens_generated / e2e if e2e > 0 else 0,
        }

    end_time = time.perf_counter()
    e2e = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else None

    # Calculate average ITL (inter-token latency)
    if len(token_times) > 1:
        itl_values = [
            token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
        ]
        itl_avg = sum(itl_values) / len(itl_values)
    else:
        itl_avg = None

    tps = tokens_generated / e2e if e2e > 0 else 0

    return {
        "ttft": ttft,
        "itl_avg": itl_avg,
        "e2e": e2e,
        "tokens_generated": tokens_generated,
        "tps": tps,
    }


@pytest.mark.skipif(
    "not config.getoption('--run-benchmarks', default=False)",
    reason="Benchmark tests only run with --run-benchmarks flag",
)
def test_llada2_benchmark_streaming(vllm_server: str):
    """Benchmark LLaDA2.0 streaming mode.

    Measures:
    - TTFT (Time To First Token)
    - ITL (Inter-Token Latency)
    - TPS (Tokens Per Second)
    - E2E (End-to-End latency)
    """
    prompt = "Once upon a time in a land far away, there lived a"
    max_tokens = 50

    # Run 5 warmup requests
    print("\n[Streaming] Running warmup requests...")
    for _ in range(5):
        benchmark_request(vllm_server, prompt, max_tokens=10, stream=True)

    # Run 10 benchmark requests
    print("[Streaming] Running benchmark requests...")
    results: list[dict[str, Any]] = []
    for i in range(10):
        result = benchmark_request(vllm_server, prompt, max_tokens, stream=True)
        results.append(result)
        print(
            f"  Request {i + 1}: TTFT={result['ttft']:.4f}s, E2E={result['e2e']:.4f}s"
        )

    # Calculate statistics
    ttft_values = [r["ttft"] for r in results if r["ttft"] is not None]
    itl_values = [r["itl_avg"] for r in results if r["itl_avg"] is not None]
    e2e_values = [r["e2e"] for r in results]
    tps_values = [r["tps"] for r in results]

    avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0
    avg_itl = sum(itl_values) / len(itl_values) if itl_values else 0
    avg_e2e = sum(e2e_values) / len(e2e_values)
    avg_tps = sum(tps_values) / len(tps_values)

    print("\n" + "=" * 60)
    print("LLaDA2.0-mini Benchmark Results (Streaming)")
    print("=" * 60)
    print(f"Average TTFT:           {avg_ttft:.4f} seconds")
    print(f"Average ITL:            {avg_itl:.4f} seconds")
    print(f"Average E2E:            {avg_e2e:.4f} seconds")
    print(f"Average TPS:            {avg_tps:.2f} tokens/second")
    print(f"Requests:               {len(results)}")
    print(f"Tokens per request:     {max_tokens}")
    print("=" * 60)

    # Basic sanity checks
    assert avg_ttft > 0, "TTFT should be positive"
    assert avg_tps > 0, "TPS should be positive"
    assert avg_e2e > 0, "E2E should be positive"


@pytest.mark.skipif(
    "not config.getoption('--run-benchmarks', default=False)",
    reason="Benchmark tests only run with --run-benchmarks flag",
)
def test_llada2_benchmark_synchronous(vllm_server: str):
    """Benchmark LLaDA2.0 synchronous mode.

    Measures:
    - TPS (Tokens Per Second)
    - E2E (End-to-End latency)

    Note: TTFT and ITL are not available in synchronous mode.
    """
    prompt = "Once upon a time in a land far away, there lived a"
    max_tokens = 50

    # Run 5 warmup requests
    print("\n[Synchronous] Running warmup requests...")
    for _ in range(5):
        benchmark_request(vllm_server, prompt, max_tokens=10, stream=False)

    # Run 10 benchmark requests
    print("[Synchronous] Running benchmark requests...")
    results: list[dict[str, Any]] = []
    for i in range(10):
        result = benchmark_request(vllm_server, prompt, max_tokens, stream=False)
        results.append(result)
        print(f"  Request {i + 1}: E2E={result['e2e']:.4f}s, TPS={result['tps']:.2f}")

    # Calculate statistics
    e2e_values = [r["e2e"] for r in results]
    tps_values = [r["tps"] for r in results]

    avg_e2e = sum(e2e_values) / len(e2e_values)
    avg_tps = sum(tps_values) / len(tps_values)

    print("\n" + "=" * 60)
    print("LLaDA2.0-mini Benchmark Results (Synchronous)")
    print("=" * 60)
    print(f"Average E2E:            {avg_e2e:.4f} seconds")
    print(f"Average TPS:            {avg_tps:.2f} tokens/second")
    print(f"Requests:               {len(results)}")
    print(f"Tokens per request:     {max_tokens}")
    print("=" * 60)

    # Basic sanity checks
    assert avg_tps > 0, "TPS should be positive"
    assert avg_e2e > 0, "E2E should be positive"
