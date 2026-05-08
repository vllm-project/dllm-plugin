#!/usr/bin/env python3
"""Simple throughput benchmark for vLLM server."""

import time

import requests


def benchmark(
    url="http://localhost:8000/v1/completions",
    prompt_tokens=256,
    output_tokens=1000,
    num_requests=10,
):
    """Run simple throughput benchmark."""
    # Generate prompt of approximate length
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)

    print(f"Running {num_requests} requests...")
    print(f"Prompt: ~{prompt_tokens} tokens, Output: {output_tokens} tokens")
    print("-" * 60)

    total_tokens = 0
    total_time = 0
    successful_requests = 0

    for i in range(num_requests):
        start = time.time()
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "inclusionAI/LLaDA2.0-mini",
                    "prompt": prompt,
                    "max_tokens": output_tokens,
                    "temperature": 0,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            # Extract token counts
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            elapsed = time.time() - start
            total_time += elapsed
            total_tokens += completion_tokens
            successful_requests += 1

            tps = completion_tokens / elapsed if elapsed > 0 else 0
            print(
                f"Request {i + 1}/{num_requests}: {completion_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)"
            )

        except Exception as e:
            print(f"Request {i + 1}/{num_requests}: FAILED - {e}")

    print("-" * 60)
    if successful_requests > 0:
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        print(f"Average throughput: {avg_tps:.1f} tokens/sec")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Successful requests: {successful_requests}/{num_requests}")
    else:
        print("All requests failed")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(benchmark())
