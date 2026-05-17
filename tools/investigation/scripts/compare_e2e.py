#!/usr/bin/env python3
"""Systematically compare dInfer vs vLLM E2E captures at every sub-operation.

Loads captures from both frameworks and compares at each checkpoint to
identify exactly where numerical divergence first appears.

Usage:
    python3 compare_e2e.py --captures-dir /workspace/captures/e2e
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import CaptureRegistry


def normalize(tensor: torch.Tensor, framework: str) -> torch.Tensor:
    """Normalize shapes: dInfer [1, seq, ...] -> [seq, ...], vLLM [tokens, ...]."""
    if framework == "dinfer" and tensor.dim() >= 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def compare(name: str, dinfer_t: torch.Tensor, vllm_t: torch.Tensor) -> dict:
    """Compare two tensors and return detailed metrics."""
    d = normalize(dinfer_t, "dinfer").float()
    v = normalize(vllm_t, "vllm").float()

    result = {
        "name": name,
        "dinfer_shape": list(dinfer_t.shape),
        "vllm_shape": list(vllm_t.shape),
    }

    if d.shape != v.shape:
        result["status"] = "SHAPE_MISMATCH"
        # Try to find common subsequence for partial comparison
        if d.dim() == v.dim() and d.dim() >= 1:
            min_seq = min(d.shape[0], v.shape[0])
            if min_seq > 0 and d.shape[1:] == v.shape[1:]:
                d_sub = d[:min_seq]
                v_sub = v[:min_seq]
                abs_diff = torch.abs(d_sub - v_sub)
                result["partial_max_diff"] = abs_diff.max().item()
                result["partial_mean_diff"] = abs_diff.mean().item()
                result["partial_tokens"] = min_seq
        return result

    abs_diff = torch.abs(d - v)
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Cosine similarity
    d_flat = d.reshape(-1)
    v_flat = v.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        d_flat.unsqueeze(0), v_flat.unsqueeze(0)
    ).item()

    # Per-token metrics (first dim is tokens/sequence)
    if d.dim() >= 2:
        per_token_max = abs_diff.reshape(d.shape[0], -1).max(dim=-1)[0]
        result["per_token_max_diff"] = per_token_max.tolist()[:8]  # first 8 tokens

    result.update({
        "status": "EXACT" if torch.equal(dinfer_t, vllm_t) else "DIVERGED",
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "cosine_sim": cos_sim,
        "exact_match": torch.equal(dinfer_t, vllm_t),
        "close_bf16": torch.allclose(d, v, atol=1e-3, rtol=1e-2),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures-dir", type=str, default="/workspace/captures/e2e")
    args = parser.parse_args()

    cap_dir = Path(args.captures_dir)
    scenario = "e2e_first_denoise"

    dinfer = CaptureRegistry.load(cap_dir, "dinfer", scenario)
    vllm_reg = CaptureRegistry.load(cap_dir, "vllm", scenario)

    print(f"\ndInfer captures: {len(dinfer.captures)}")
    print(f"vLLM captures: {len(vllm_reg.captures)}")

    # Find common checkpoint names
    common = sorted(set(dinfer.captures.keys()) & set(vllm_reg.captures.keys()))
    dinfer_only = sorted(set(dinfer.captures.keys()) - set(vllm_reg.captures.keys()))
    vllm_only = sorted(set(vllm_reg.captures.keys()) - set(dinfer.captures.keys()))

    if dinfer_only:
        print(f"\ndInfer-only ({len(dinfer_only)}): {dinfer_only[:10]}")
    if vllm_only:
        print(f"vLLM-only ({len(vllm_only)}): {vllm_only[:10]}")

    print(f"\nCommon checkpoints: {len(common)}")

    # Compare model inputs first
    print(f"\n{'=' * 70}")
    print("MODEL INPUTS")
    print(f"{'=' * 70}")

    if "model_input_ids" in common:
        d_ids = dinfer.captures["model_input_ids"]
        v_ids = vllm_reg.captures["model_input_ids"]
        d_ids_n = normalize(d_ids, "dinfer")
        v_ids_n = normalize(v_ids, "vllm")
        print(f"  dInfer input_ids: shape={list(d_ids.shape)} values={d_ids_n[:8].tolist()}")
        print(f"  vLLM   input_ids: shape={list(v_ids.shape)} values={v_ids_n[:8].tolist()}")
        if d_ids_n.shape == v_ids_n.shape:
            match = torch.equal(d_ids_n, v_ids_n)
            print(f"  Input IDs match: {match}")
            if not match:
                diff_mask = d_ids_n != v_ids_n
                diff_pos = diff_mask.nonzero(as_tuple=True)[0]
                print(f"  Differ at positions: {diff_pos.tolist()[:20]}")

    if "model_positions" in common:
        d_pos = dinfer.captures["model_positions"]
        v_pos = vllm_reg.captures["model_positions"]
        d_pos_n = normalize(d_pos, "dinfer")
        v_pos_n = normalize(v_pos, "vllm")
        print(f"  dInfer positions: shape={list(d_pos.shape)} values={d_pos_n[:8].tolist()}")
        print(f"  vLLM   positions: shape={list(v_pos.shape)} values={v_pos_n[:8].tolist()}")

    # Compare all operations in order: embedding -> layers -> final norm -> lm_head
    print(f"\n{'=' * 70}")
    print("SYSTEMATIC COMPARISON (ordered by computation flow)")
    print(f"{'=' * 70}")

    # Define the expected order
    ordered_keys = []

    # Embedding
    if "embedding" in common:
        ordered_keys.append("embedding")

    # Per-layer operations
    num_layers = 20  # LLaDA2-mini
    layer_ops = [
        "input", "norm1_input", "norm1_output",
        "attn_input", "qkv_proj", "q_norm", "k_norm", "attn_output", "o_proj",
        "residual1", "norm2_output",
        "moe_input", "gate_logits", "moe_output",
        "output",
    ]
    for i in range(num_layers):
        for op in layer_ops:
            key = f"layer{i:02d}.{op}"
            if key in common:
                ordered_keys.append(key)

    # Final operations
    for key in ["final_norm_input", "final_norm_output", "lm_head_output"]:
        if key in common:
            ordered_keys.append(key)

    # Add any remaining common keys not in ordered list
    remaining = [k for k in common if k not in ordered_keys and k not in ("model_input_ids", "model_positions")]
    ordered_keys.extend(remaining)

    results = []
    first_divergence = None
    first_significant_divergence = None

    for name in ordered_keys:
        r = compare(name, dinfer.captures[name], vllm_reg.captures[name])
        results.append(r)

        status = r["status"]
        if status == "SHAPE_MISMATCH":
            symbol = "!!"
            detail = f"shapes {r['dinfer_shape']} vs {r['vllm_shape']}"
            if "partial_max_diff" in r:
                detail += f" (partial max_diff={r['partial_max_diff']:.2e})"
        elif status == "EXACT":
            symbol = "=="
            detail = f"cos={r['cosine_sim']:.6f}"
        else:
            max_d = r["max_abs_diff"]
            if max_d < 1e-6:
                symbol = "~="
            elif max_d < 1e-3:
                symbol = "~"
            else:
                symbol = "!!"
            detail = f"max_diff={max_d:.2e} mean={r['mean_abs_diff']:.2e} cos={r['cosine_sim']:.6f}"

            if first_divergence is None and not r["exact_match"]:
                first_divergence = name
            if first_significant_divergence is None and max_d > 1e-3:
                first_significant_divergence = name

        print(f"  {symbol:3s} {name:50s} {detail}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    exact = sum(1 for r in results if r.get("exact_match"))
    close = sum(1 for r in results if r.get("close_bf16"))
    total = len(results)

    print(f"  Total checkpoints compared: {total}")
    print(f"  Exact matches: {exact}/{total}")
    print(f"  BF16-close (atol=1e-3): {close}/{total}")
    print(f"  First divergence: {first_divergence}")
    print(f"  First significant divergence (>1e-3): {first_significant_divergence}")

    if first_significant_divergence:
        r = next(r for r in results if r["name"] == first_significant_divergence)
        print(f"\n  >>> FIRST SIGNIFICANT DIVERGENCE at: {first_significant_divergence}")
        print(f"  >>> max_abs_diff = {r['max_abs_diff']:.6e}")
        print(f"  >>> cosine_sim = {r['cosine_sim']:.6f}")
        if "per_token_max_diff" in r:
            print(f"  >>> per-token max diff: {[f'{x:.2e}' for x in r['per_token_max_diff']]}")

    # Save results
    out_path = cap_dir / scenario / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "total": total,
            "exact": exact,
            "close_bf16": close,
            "first_divergence": first_divergence,
            "first_significant_divergence": first_significant_divergence,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
