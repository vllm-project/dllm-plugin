#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare lm-eval results between dInfer and dllm-plugin.

Usage::

    python tools/compare_lm_eval_results.py \
        results/dinfer.json results/dllm_plugin.json
"""

from __future__ import annotations

import json
import sys


def load_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", {})
    out = {}
    for task_name, metrics in results.items():
        for metric_name, value in metrics.items():
            if metric_name.startswith("alias"):
                continue
            if isinstance(value, (int, float)):
                out[f"{task_name}/{metric_name}"] = value
    return out


def compare(baseline_path: str, plugin_path: str) -> None:
    baseline = load_results(baseline_path)
    plugin = load_results(plugin_path)

    all_keys = sorted(set(baseline) | set(plugin))

    print(f"{'Metric':<50} {'dInfer':>10} {'Plugin':>10} {'Delta':>10}")
    print("-" * 82)

    for key in all_keys:
        b = baseline.get(key)
        p = plugin.get(key)
        b_str = f"{b:.4f}" if b is not None else "N/A"
        p_str = f"{p:.4f}" if p is not None else "N/A"
        if b is not None and p is not None:
            delta = p - b
            d_str = f"{delta:+.4f}"
        else:
            d_str = "N/A"
        print(f"{key:<50} {b_str:>10} {p_str:>10} {d_str:>10}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <dinfer_results.json> <plugin_results.json>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
