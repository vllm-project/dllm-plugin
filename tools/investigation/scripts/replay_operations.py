#!/usr/bin/env python3
"""Replay isolation: feed dInfer-captured inputs through vLLM operations.

For each operation, loads the dInfer-captured INPUT tensor, runs it through
the vLLM implementation with identical weights, and compares against the
dInfer-captured OUTPUT tensor.

This isolates whether an operation itself diverges (intrinsic) or merely
propagates upstream error (propagated).

Usage:
    python3 replay_operations.py --captures-dir ./captures --scenario first_block
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import CaptureRegistry, normalize_for_comparison


def load_rmsnorm_weights(
    model_path: str, layer_idx: int, norm_name: str = "input_layernorm"
):
    """Load RMSNorm weights from HuggingFace checkpoint."""
    import glob

    from safetensors import safe_open

    safetensors_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    weight_key = f"model.layers.{layer_idx}.{norm_name}.weight"

    for f in safetensors_files:
        with safe_open(f, framework="pt") as sf:
            if weight_key in sf.keys():
                return sf.get_tensor(weight_key)

    raise KeyError(f"Weight {weight_key} not found in checkpoint")


def replay_rmsnorm(
    dinfer_input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Run RMSNorm using dllm-plugin's LLaDA2RMSNorm implementation."""
    orig_dtype = dinfer_input.dtype
    x = dinfer_input.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype)
    x = x * weight.to(orig_dtype)
    return x


def replay_rmsnorm_stock_vllm(
    dinfer_input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Run RMSNorm using stock vLLM behavior (FP32 weight multiply)."""
    orig_dtype = dinfer_input.dtype
    x = dinfer_input.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    # Stock vLLM: multiply weight in FP32 (the bug)
    x = x * weight.to(torch.float32)
    return x.to(orig_dtype)


def compare_result(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Compare replay result against reference."""
    a = actual.float()
    e = expected.float()
    if a.shape != e.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "actual_shape": list(a.shape),
            "expected_shape": list(e.shape),
        }

    abs_diff = torch.abs(a - e)
    return {
        "name": name,
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "exact_match": torch.equal(actual, expected),
        "pass_BF16": torch.allclose(a, e, atol=1e-3, rtol=1e-2),
    }


def run_replay(captures_dir: Path, model_path: str, scenario: str):
    """Run replay isolation for all operations across all layers."""
    print(f"\n{'=' * 70}")
    print(f"Replay Isolation: {scenario}")
    print(f"{'=' * 70}")

    dinfer_reg = CaptureRegistry.load(captures_dir, "dinfer", scenario)
    vllm_reg = CaptureRegistry.load(captures_dir, "vllm", scenario)

    results = {"scenario": scenario, "operations": {}}

    # Replay RMSNorm for each layer
    print("\n[RMSNORM] Replaying input_layernorm for all layers...")
    for layer_idx in range(100):
        input_key = f"L7_atomic.layer{layer_idx:02d}.rmsnorm_normalized"
        output_key = f"L7_atomic.layer{layer_idx:02d}.rmsnorm_scaled"
        dinfer_input_key = f"L5_sublayer.layer{layer_idx:02d}.input_norm_out"

        if dinfer_input_key not in dinfer_reg.captures:
            break

        # Try to get the pre-normalization input
        # The input to RMSNorm is the hidden state entering the layer
        layer_input_key = f"L4_layer.layer{layer_idx:02d}.hidden_out"
        prev_layer_key = (
            f"L4_layer.layer{(layer_idx - 1):02d}.hidden_out"
            if layer_idx > 0
            else "L4_layer.global.embedding"
        )

        if prev_layer_key in dinfer_reg.captures:
            dinfer_input = normalize_for_comparison(
                dinfer_reg.captures[prev_layer_key], "dinfer"
            )
            dinfer_output = normalize_for_comparison(
                dinfer_reg.captures[dinfer_input_key], "dinfer"
            )
            vllm_output = normalize_for_comparison(
                vllm_reg.captures.get(dinfer_input_key, torch.zeros(1)), "vllm"
            )

            try:
                weight = load_rmsnorm_weights(model_path, layer_idx)
                # Replay with custom RMSNorm (dInfer-compatible)
                replayed_custom = replay_rmsnorm(dinfer_input, weight)
                custom_result = compare_result(
                    f"rmsnorm_custom.layer{layer_idx:02d}",
                    replayed_custom,
                    dinfer_output,
                )

                # Replay with stock vLLM RMSNorm (known-buggy)
                replayed_stock = replay_rmsnorm_stock_vllm(dinfer_input, weight)
                stock_result = compare_result(
                    f"rmsnorm_stock.layer{layer_idx:02d}", replayed_stock, dinfer_output
                )

                intrinsic = "YES" if not custom_result["pass_BF16"] else "no"
                print(
                    f"  Layer {layer_idx:2d}: custom_diff={custom_result['max_abs_diff']:.2e}"
                    f"  stock_diff={stock_result['max_abs_diff']:.2e}"
                    f"  intrinsic_divergence={intrinsic}"
                )

                results["operations"][f"rmsnorm.layer{layer_idx:02d}"] = {
                    "custom": custom_result,
                    "stock": stock_result,
                    "intrinsic_divergence": not custom_result["pass_BF16"],
                }
            except Exception as e:
                print(f"  Layer {layer_idx:2d}: SKIPPED ({e})")

    # Summary
    print(f"\n{'─' * 70}")
    intrinsic_count = sum(
        1
        for v in results["operations"].values()
        if v.get("intrinsic_divergence", False)
    )
    print(
        f"Intrinsic divergences found: {intrinsic_count} / {len(results['operations'])}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Replay isolation tests")
    parser.add_argument("--captures-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="/workspace/llada2-mini")
    parser.add_argument("--scenario", type=str, default="first_block")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    captures_dir = Path(args.captures_dir)
    results = run_replay(captures_dir, args.model_path, args.scenario)

    output_dir = (
        Path(args.output_dir) if args.output_dir else captures_dir / "comparisons"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.scenario}_replay.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] {report_path}")


if __name__ == "__main__":
    main()
