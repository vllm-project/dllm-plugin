#!/usr/bin/env python3
"""
Custom benchmark script for testing structured outputs with guidellm.

This script uses guidellm as a Python library to pass the extras parameter
which contains structured output configuration (guided_regex).
"""

import asyncio
import json
from pathlib import Path

from guidellm.benchmark import (
    BenchmarkGenerativeTextArgs,
    GenerativeConsoleBenchmarkerProgress,
    benchmark_generative_text,
)
from guidellm.logger import logger


class PathEncoder(json.JSONEncoder):
    """JSON encoder that handles Path objects."""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


async def run_benchmark(
    target: str,
    model: str,
    data_path: str,
    profile: str,
    rate: float | None,
    output_path: str,
    extras: dict | None = None,
    max_tokens: int | None = None,
):
    """
    Run a single benchmark with optional structured output configuration.

    :param target: vLLM server URL
    :param model: Model name
    :param data_path: Path to JSONL data file
    :param profile: Benchmark profile (synchronous, constant, etc.)
    :param rate: Requests per second (None for synchronous profile)
    :param output_path: Where to save results
    :param extras: Extra parameters to pass to API (e.g., guided_regex)
    :param max_tokens: Default max_tokens if not in data
    """
    # Configure backend_kwargs with all parameters
    backend_kwargs = {
        "target": target,
        "model": model,
        "max_tokens": max_tokens,
        "request_format": "/v1/completions",  # Use completions API, not chat
    }

    if extras:
        backend_kwargs["extras"] = extras

    # Configure benchmark (don't pass rate for synchronous profile)
    args_dict = {
        "backend": "openai_http",
        "backend_kwargs": backend_kwargs,
        "data": [data_path],
        "profile": profile,
    }

    if rate is not None:
        args_dict["rate"] = rate

    args = BenchmarkGenerativeTextArgs(**args_dict)

    # Run benchmark
    logger.info(f"Starting benchmark: {profile} @ {rate}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Extras: {extras}")

    results = await benchmark_generative_text(
        args=args,
        progress=GenerativeConsoleBenchmarkerProgress(),
    )

    # benchmark_generative_text returns results in various formats
    # Save each result
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Handle different result types
    if isinstance(results, (tuple, list)):
        # Multiple results
        results_data = []
        for r in results:
            if isinstance(r, dict):
                results_data.append(r)
            elif hasattr(r, 'model_dump'):
                results_data.append(r.model_dump())
            else:
                results_data.append(str(r))
    elif isinstance(results, dict):
        results_data = results
    elif hasattr(results, 'model_dump'):
        results_data = results.model_dump()
    else:
        results_data = str(results)

    with output.open("w") as f:
        json.dump(results_data, f, indent=2, cls=PathEncoder)

    logger.info(f"Results saved to {output_path}")

    return results


async def main():
    """Run all 4 benchmark scenarios."""

    TARGET = "http://localhost:8000"
    MODEL = "inclusionAI/LLaDA2.0-mini"
    PATTERN = r"^([a-z]{6}[A-Z]{5}[0-9]{4}[a-z]{3}[A-Z]{2}[0-9]{1})+$"

    scenarios = [
        {
            "name": "1_synchronous_freeform",
            "data": "/tmp/structured-output-data-freeform.jsonl",
            "profile": "synchronous",
            "rate": None,  # Synchronous doesn't use rate
            "extras": None,
            "description": "Synchronous (concurrency=1) with free-form outputs",
        },
        {
            "name": "2_synchronous_structured",
            "data": "/tmp/structured-output-data-structured.jsonl",
            "profile": "synchronous",
            "rate": None,  # Synchronous doesn't use rate
            "extras": {"body": {"guided_regex": PATTERN}},
            "description": "Synchronous (concurrency=1) with structured outputs",
        },
        {
            "name": "3_constant_freeform",
            "data": "/tmp/structured-output-data-freeform.jsonl",
            "profile": "constant",
            "rate": 100.0,
            "extras": None,
            "description": "Constant rate (100 RPS) with free-form outputs",
        },
        {
            "name": "4_constant_structured",
            "data": "/tmp/structured-output-data-structured.jsonl",
            "profile": "constant",
            "rate": 100.0,
            "extras": {"body": {"guided_regex": PATTERN}},
            "description": "Constant rate (100 RPS) with structured outputs",
        },
    ]

    results = {}

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'='*70}\n")

        try:
            result = await run_benchmark(
                target=TARGET,
                model=MODEL,
                data_path=scenario["data"],
                profile=scenario["profile"],
                rate=scenario["rate"],
                output_path=f"/tmp/benchmark_{scenario['name']}.json",
                extras=scenario["extras"],
                max_tokens=500,  # Default if not in data (reduced to fit in 2048 context)
            )

            results[scenario["name"]] = {
                "success": True,
                "output_path": f"/tmp/benchmark_{scenario['name']}.json",
            }

        except Exception as e:
            logger.error(f"Scenario {scenario['name']} failed: {e}")
            results[scenario["name"]] = {
                "success": False,
                "error": str(e),
            }

    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    for name, result in results.items():
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        print(f"{status}: {name}")
        if not result["success"]:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Results: {result['output_path']}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
