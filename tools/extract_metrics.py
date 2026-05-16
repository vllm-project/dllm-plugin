#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the dLLM project
"""Extract key metrics from GuideLLM benchmark JSON.

Usage:
    python3 tools/extract_metrics.py benchmarks/baseline.json

Output:
    TTFT (median): 1720.5ms
    ITL (median): 3.92ms
    TPS (generation): 178.4 tok/s
    TPS (total): 188.2 tok/s
    Requests completed: 25
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def extract_metrics(json_path: str | Path) -> dict[str, float | int]:
    """Extract key performance metrics from GuideLLM benchmark JSON.

    Args:
        json_path: Path to GuideLLM benchmark JSON file

    Returns:
        Dictionary with extracted metrics:
        - ttft_ms_median: Time to first token (median)
        - itl_ms_median: Inter-token latency (median)
        - output_tokens_per_sec: Generation throughput
        - total_tokens_per_sec: Total throughput (prompt + generation)
        - completed_requests: Number of successful requests

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        KeyError: If expected benchmark structure is missing
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(json_path) as f:
        data = json.load(f)

    # Extract metrics from GuideLLM structure
    benchmark = data["benchmarks"][0]
    stats = benchmark["request_latency_stats"]
    throughput = benchmark["server_throughput_stats"]

    return {
        "ttft_ms_median": stats["ttft_ms_median"],
        "itl_ms_median": stats["itl_ms_median"],
        "output_tokens_per_sec": throughput["output_tokens_per_sec"],
        "total_tokens_per_sec": throughput["total_tokens_per_sec"],
        "completed_requests": benchmark["completed_requests"],
    }


def format_metrics(metrics: dict[str, float | int]) -> str:
    """Format metrics for human-readable output.

    Args:
        metrics: Dictionary of metrics from extract_metrics()

    Returns:
        Formatted string with metrics
    """
    return "\n".join(
        [
            f"TTFT (median): {metrics['ttft_ms_median']:.1f}ms",
            f"ITL (median): {metrics['itl_ms_median']:.2f}ms",
            f"TPS (generation): {metrics['output_tokens_per_sec']:.1f} tok/s",
            f"TPS (total): {metrics['total_tokens_per_sec']:.1f} tok/s",
            f"Requests completed: {metrics['completed_requests']}",
        ]
    )


def compare_metrics(
    baseline: dict[str, float | int],
    optimized: dict[str, float | int],
) -> str:
    """Compare baseline vs optimized metrics and calculate improvements.

    Args:
        baseline: Metrics from baseline run
        optimized: Metrics from optimized run

    Returns:
        Formatted comparison string with % improvements
    """
    lines = ["Baseline → Optimized (% change):"]

    # TTFT: Lower is better
    ttft_change = (
        (optimized["ttft_ms_median"] - baseline["ttft_ms_median"])
        / baseline["ttft_ms_median"]
        * 100
    )
    lines.append(
        f"TTFT: {baseline['ttft_ms_median']:.1f}ms → "
        f"{optimized['ttft_ms_median']:.1f}ms ({ttft_change:+.1f}%)"
    )

    # ITL: Lower is better
    itl_change = (
        (optimized["itl_ms_median"] - baseline["itl_ms_median"])
        / baseline["itl_ms_median"]
        * 100
    )
    lines.append(
        f"ITL: {baseline['itl_ms_median']:.2f}ms → "
        f"{optimized['itl_ms_median']:.2f}ms ({itl_change:+.1f}%)"
    )

    # TPS: Higher is better
    tps_change = (
        (optimized["output_tokens_per_sec"] - baseline["output_tokens_per_sec"])
        / baseline["output_tokens_per_sec"]
        * 100
    )
    lines.append(
        f"TPS (gen): {baseline['output_tokens_per_sec']:.1f} tok/s → "
        f"{optimized['output_tokens_per_sec']:.1f} tok/s ({tps_change:+.1f}%)"
    )

    return "\n".join(lines)


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python3 tools/extract_metrics.py <benchmark.json>", file=sys.stderr
        )
        print(
            "   or: python3 tools/extract_metrics.py <baseline.json> <optimized.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    json_path = sys.argv[1]

    try:
        metrics = extract_metrics(json_path)
        print(format_metrics(metrics))

        # If second file provided, compare
        if len(sys.argv) >= 3:
            optimized_path = sys.argv[2]
            baseline_metrics = metrics
            optimized_metrics = extract_metrics(optimized_path)

            print()
            print("=" * 50)
            print(compare_metrics(baseline_metrics, optimized_metrics))

    except FileNotFoundError:
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, IndexError) as e:
        print(f"Error: Invalid benchmark JSON structure: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
