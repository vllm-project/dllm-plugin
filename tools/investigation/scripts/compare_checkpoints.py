#!/usr/bin/env python3
"""Compare captured tensors between dInfer and dllm-plugin implementations.

CPU-only — no GPU required. Reads .pt files from both capture directories
and produces a comprehensive divergence report.

Usage:
    python3 compare_checkpoints.py --captures-dir ./captures --scenario first_block
    python3 compare_checkpoints.py --captures-dir ./captures --all-scenarios
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import CaptureRegistry, assert_config_match, normalize_for_comparison

TOLERANCE_FP32 = {"atol": 1e-5, "rtol": 1e-4, "name": "FP32"}
TOLERANCE_BF16 = {"atol": 1e-3, "rtol": 1e-2, "name": "BF16"}
TOLERANCE_BF16_LOOSE = {"atol": 1e-2, "rtol": 1e-2, "name": "BF16_LOOSE"}
TOLERANCE_ACCUMULATED = {"atol": 5e-2, "rtol": 5e-2, "name": "ACCUMULATED"}

ALL_TOLERANCES = [
    TOLERANCE_FP32,
    TOLERANCE_BF16,
    TOLERANCE_BF16_LOOSE,
    TOLERANCE_ACCUMULATED,
]


def try_align_shapes(a: torch.Tensor, e: torch.Tensor, name: str):
    """Try to align tensor shapes between frameworks.

    Handles the known transposition: dInfer captures Q/K norm as
    [num_heads, seq_len, head_dim] while vLLM captures as [seq_len, num_heads, head_dim].
    When shapes are identical but axes may be swapped (e.g. [4, 4, 128]),
    tries both orderings and picks the one with higher cosine similarity.
    """
    if a.shape != e.shape:
        # Different shapes — try permuting first two dims
        if a.dim() == e.dim() == 3 and a.shape[2] == e.shape[2]:
            e_perm = e.permute(1, 0, 2)
            if a.shape == e_perm.shape:
                return a, e_perm
            a_perm = a.permute(1, 0, 2)
            if a_perm.shape == e.shape:
                return a_perm, e
        return a, e

    # Same shape — but may need transposition (e.g. both [4, 4, 128])
    if a.dim() == 3 and a.shape[0] == a.shape[1] and "norm" in name:
        cos_orig = F.cosine_similarity(
            a.flatten().unsqueeze(0), e.flatten().unsqueeze(0)
        ).item()
        e_perm = e.permute(1, 0, 2)
        cos_perm = F.cosine_similarity(
            a.flatten().unsqueeze(0), e_perm.flatten().unsqueeze(0)
        ).item()
        if cos_perm > cos_orig + 0.01:
            return a, e_perm

    return a, e


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor, name: str) -> dict:
    """Compute comprehensive comparison metrics between two tensors."""
    a = normalize_for_comparison(actual.float(), "vllm")
    e = normalize_for_comparison(expected.float(), "dinfer")
    a, e = try_align_shapes(a, e, name)

    if a.shape != e.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }

    abs_diff = torch.abs(a - e)
    rel_diff = abs_diff / (torch.abs(e) + 1e-10)

    # Core metrics
    result = {
        "name": name,
        "shape": list(a.shape),
        "dtype": str(actual.dtype),
        "exact_match": torch.equal(actual, expected),
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "median_abs_diff": abs_diff.median().item(),
        "p95_abs_diff": torch.quantile(abs_diff, 0.95).item(),
        "p99_abs_diff": torch.quantile(abs_diff, 0.99).item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
    }

    # Cosine similarity (flatten to 1D)
    if a.numel() > 1:
        cos_sim = F.cosine_similarity(
            a.flatten().unsqueeze(0), e.flatten().unsqueeze(0)
        ).item()
        result["cosine_similarity"] = cos_sim

    # Check all tolerance levels
    for tol in ALL_TOLERANCES:
        passed = torch.allclose(a, e, atol=tol["atol"], rtol=tol["rtol"])
        result[f"pass_{tol['name']}"] = passed

    # Determine overall status based on which tolerance passes
    if result.get("pass_FP32"):
        result["status"] = "EXACT"
    elif result.get("pass_BF16"):
        result["status"] = "PASS_BF16"
    elif result.get("pass_BF16_LOOSE"):
        result["status"] = "PASS_LOOSE"
    elif result.get("pass_ACCUMULATED"):
        result["status"] = "PASS_ACCUMULATED"
    else:
        result["status"] = "FAIL"

    # Per-position analysis (for sequence-dimension tensors)
    if a.dim() >= 2:
        per_pos_max = abs_diff.view(a.shape[0], -1).max(dim=-1).values
        result["per_position_max_diff"] = per_pos_max.tolist()

    # NaN/Inf checks
    result["actual_has_nan"] = torch.isnan(actual).any().item()
    result["actual_has_inf"] = torch.isinf(actual).any().item()
    result["expected_has_nan"] = torch.isnan(expected).any().item()
    result["expected_has_inf"] = torch.isinf(expected).any().item()

    return result


def compute_kl_divergence(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10
) -> float:
    """KL divergence between probability distributions."""
    p = torch.clamp(p.float(), min=eps)
    q = torch.clamp(q.float(), min=eps)
    return (p * torch.log(p / q)).sum().item()


def compare_scenario(captures_dir: Path, scenario: str) -> dict:
    """Compare all checkpoints for a scenario."""
    print(f"\n{'=' * 70}")
    print(f"Comparing scenario: {scenario}")
    print(f"{'=' * 70}")

    dinfer_reg = CaptureRegistry.load(captures_dir, "dinfer", scenario)
    vllm_reg = CaptureRegistry.load(captures_dir, "vllm", scenario)

    # Validate config match
    try:
        assert_config_match(dinfer_reg.model_config, vllm_reg.model_config)
        print("[CONFIG] Model configs match")
    except ValueError as e:
        print(f"[CONFIG WARNING] {e}")

    # Validate input alignment
    if dinfer_reg.input_ids == vllm_reg.input_ids:
        print(f"[INPUT] Token IDs match: {dinfer_reg.input_ids}")
    else:
        print("[INPUT WARNING] Token IDs differ!")
        print(f"  dInfer: {dinfer_reg.input_ids}")
        print(f"  vLLM:   {vllm_reg.input_ids}")

    # Find matching checkpoints
    common_keys = sorted(
        set(dinfer_reg.captures.keys()) & set(vllm_reg.captures.keys())
    )
    dinfer_only = sorted(
        set(dinfer_reg.captures.keys()) - set(vllm_reg.captures.keys())
    )
    vllm_only = sorted(set(vllm_reg.captures.keys()) - set(dinfer_reg.captures.keys()))

    print(
        f"\n[COVERAGE] Common: {len(common_keys)}, dInfer-only: {len(dinfer_only)}, vLLM-only: {len(vllm_only)}"
    )
    if dinfer_only:
        print(
            f"  dInfer-only: {dinfer_only[:10]}{'...' if len(dinfer_only) > 10 else ''}"
        )
    if vllm_only:
        print(f"  vLLM-only:   {vllm_only[:10]}{'...' if len(vllm_only) > 10 else ''}")

    # Compare each checkpoint
    results = {}
    status_counts = defaultdict(int)

    for key in common_keys:
        result = compare_tensors(vllm_reg.captures[key], dinfer_reg.captures[key], key)
        results[key] = result
        status_counts[result["status"]] += 1

        status_icon = {
            "EXACT": "=",
            "PASS_BF16": "+",
            "PASS_LOOSE": "~",
            "PASS_ACCUMULATED": "~",
            "FAIL": "X",
            "SHAPE_MISMATCH": "!",
        }
        icon = status_icon.get(result["status"], "?")
        max_diff = result.get("max_abs_diff", "N/A")
        cos_sim = result.get("cosine_similarity", "N/A")
        if isinstance(max_diff, float):
            max_diff = f"{max_diff:.2e}"
        if isinstance(cos_sim, float):
            cos_sim = f"{cos_sim:.6f}"
        print(f"  [{icon}] {key:50s}  max_diff={max_diff:>10s}  cos_sim={cos_sim}")

    # KL divergence for probability checkpoints
    kl_results = {}
    for key in ["L2_probs.global.softmax_probs"]:
        if key in dinfer_reg.captures and key in vllm_reg.captures:
            p = normalize_for_comparison(dinfer_reg.captures[key], "dinfer")
            q = normalize_for_comparison(vllm_reg.captures[key], "vllm")
            if p.shape == q.shape:
                kl = compute_kl_divergence(p, q)
                kl_results[key] = kl
                print(f"\n  [KL] {key}: {kl:.6e}")

    # Summary
    print(f"\n{'─' * 70}")
    print(f"Summary: {len(common_keys)} checkpoints compared")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    return {
        "scenario": scenario,
        "num_compared": len(common_keys),
        "num_dinfer_only": len(dinfer_only),
        "num_vllm_only": len(vllm_only),
        "status_counts": dict(status_counts),
        "kl_divergence": kl_results,
        "results": results,
        "dinfer_only_keys": dinfer_only,
        "vllm_only_keys": vllm_only,
    }


def generate_divergence_matrix(report: dict) -> str:
    """Generate a layer-by-layer divergence matrix from comparison results."""
    lines = ["\n## Divergence Matrix (max absolute difference)\n"]

    # Group by level and layer
    levels = defaultdict(lambda: defaultdict(float))
    for key, result in report["results"].items():
        parts = key.split(".")
        if len(parts) >= 3:
            level = parts[0]
            layer = parts[1]
            op = ".".join(parts[2:])
            max_diff = result.get("max_abs_diff", 0)
            levels[f"{level}.{op}"][layer] = max_diff

    # Find all layers
    all_layers = sorted(set(layer for ops in levels.values() for layer in ops.keys()))

    # Header
    header = f"{'Operation':40s} | " + " | ".join(f"{l:>8s}" for l in all_layers)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for op_key in sorted(levels.keys()):
        row = f"{op_key:40s} | "
        row += " | ".join(
            f"{levels[op_key].get(l, 0):.2e}"
            if levels[op_key].get(l, 0) > 0
            else f"{'---':>8s}"
            for l in all_layers
        )
        lines.append(row)

    return "\n".join(lines)


def generate_error_flow(report: dict) -> str:
    """Generate per-layer error flow analysis."""
    lines = ["\n## Per-Layer Error Flow\n"]

    # Find layers from L5 results
    layer_ops = defaultdict(dict)
    for key, result in report["results"].items():
        if key.startswith("L5_sublayer.layer"):
            parts = key.split(".")
            layer = parts[1]
            op = parts[2]
            layer_ops[layer][op] = result.get("max_abs_diff", 0)

    for layer in sorted(layer_ops.keys()):
        ops = layer_ops[layer]
        flow = " → ".join(f"{op}={diff:.2e}" for op, diff in sorted(ops.items()))
        lines.append(f"  {layer}: {flow}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare captured tensors")
    parser.add_argument(
        "--captures-dir", type=str, required=True, help="Root captures directory"
    )
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for reports"
    )
    args = parser.parse_args()

    captures_dir = Path(args.captures_dir)

    # Determine scenarios
    if args.all_scenarios:
        scenarios = [d.name for d in captures_dir.iterdir() if d.is_dir()]
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        scenarios = ["first_block"]

    output_dir = (
        Path(args.output_dir) if args.output_dir else captures_dir / "comparisons"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    for scenario in scenarios:
        report = compare_scenario(captures_dir, scenario)
        all_reports[scenario] = report

        # Save per-scenario report
        report_path = output_dir / f"{scenario}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[SAVED] {report_path}")

        # Generate divergence matrix
        matrix = generate_divergence_matrix(report)
        print(matrix)

        # Generate error flow
        flow = generate_error_flow(report)
        print(flow)

    # Generate consolidated summary
    summary_lines = [
        "# dInfer vs dllm-plugin Numerical Comparison Report\n",
        f"## Scenarios: {', '.join(scenarios)}\n",
    ]

    summary_lines.append(
        "| Scenario | Compared | EXACT | PASS_BF16 | PASS_LOOSE | FAIL |"
    )
    summary_lines.append(
        "|----------|----------|-------|-----------|------------|------|"
    )
    for scenario, report in all_reports.items():
        sc = report["status_counts"]
        summary_lines.append(
            f"| {scenario} | {report['num_compared']} "
            f"| {sc.get('EXACT', 0)} | {sc.get('PASS_BF16', 0)} "
            f"| {sc.get('PASS_LOOSE', 0) + sc.get('PASS_ACCUMULATED', 0)} "
            f"| {sc.get('FAIL', 0) + sc.get('SHAPE_MISMATCH', 0)} |"
        )

    summary = "\n".join(summary_lines)
    summary_path = output_dir / "COMPARISON_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\n[SAVED] {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
