#!/usr/bin/env python3
"""Compare deep attention captures between dInfer and vLLM.

Loads A1-A9 checkpoints from both frameworks and identifies the exact
point where divergence is introduced.

Usage:
    python3 compare_attention_deep.py --captures-dir ./captures [--layer 0]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import CaptureRegistry

CHECKPOINTS = [
    ("A1_q_after_norm", "Q after norm (pre-RoPE)"),
    ("A2_k_after_norm", "K after norm (pre-RoPE)"),
    ("A3_rope_cos", "RoPE cos values"),
    ("A4_rope_sin", "RoPE sin values"),
    ("A5_q_after_rope", "Q after RoPE"),
    ("A6_k_after_rope", "K after RoPE"),
    ("A6b_v_states", "V states"),
    ("A7_k_after_gqa", "K after GQA repeat"),
    ("A8_attn_output_pre_oproj", "Attention output (pre-o_proj)"),
    ("A9_oproj_output", "O-proj output"),
]


def try_align(a: torch.Tensor, e: torch.Tensor, name: str):
    """Try shape alignments: batch dim strip, head/seq transpose."""
    # Strip batch dim
    if a.dim() > e.dim() and a.shape[0] == 1:
        a = a.squeeze(0)
    if e.dim() > a.dim() and e.shape[0] == 1:
        e = e.squeeze(0)

    if a.shape == e.shape:
        return a, e

    # Try transpose of first two dims (head/seq swap)
    if a.dim() == e.dim() >= 3:
        if a.dim() == 3 and a.shape[2] == e.shape[2]:
            e_p = e.permute(1, 0, 2)
            if a.shape == e_p.shape:
                cos_orig = F.cosine_similarity(
                    a.flatten().float().unsqueeze(0), e.flatten().float().unsqueeze(0)
                ).item()
                cos_perm = F.cosine_similarity(
                    a.flatten().float().unsqueeze(0), e_p.flatten().float().unsqueeze(0)
                ).item()
                return (a, e_p) if cos_perm > cos_orig else (a, e)

        if a.dim() == 4 and e.dim() == 3:
            # dInfer [bsz, heads, seq, dim] -> [heads, seq, dim] -> try [seq, heads, dim]
            a_sq = a.squeeze(0)
            if a_sq.dim() == 3:
                a_t = a_sq.permute(1, 0, 2)
                if a_t.shape == e.shape:
                    return a_t, e
                if a_sq.shape == e.shape:
                    cos_orig = F.cosine_similarity(
                        a_sq.flatten().float().unsqueeze(0),
                        e.flatten().float().unsqueeze(0),
                    ).item()
                    cos_perm = F.cosine_similarity(
                        a_t.flatten().float().unsqueeze(0),
                        e.flatten().float().unsqueeze(0),
                    ).item()
                    return (a_t, e) if cos_perm > cos_orig else (a_sq, e)

    return a, e


def compare(a: torch.Tensor, e: torch.Tensor, name: str) -> dict:
    """Compare two tensors with metrics."""
    a, e = try_align(a, e, name)

    if a.shape != e.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "a_shape": list(a.shape),
            "e_shape": list(e.shape),
        }

    af, ef = a.float(), e.float()
    diff = torch.abs(af - ef)

    result = {
        "name": name,
        "shape": list(a.shape),
        "exact_match": torch.equal(a, e),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "elements_differ": (a != e).sum().item(),
        "total_elements": a.numel(),
    }

    if a.numel() > 1:
        result["cosine_sim"] = F.cosine_similarity(
            af.flatten().unsqueeze(0), ef.flatten().unsqueeze(0)
        ).item()

    # Per-position analysis (if 2D or 3D)
    if diff.dim() >= 2:
        per_pos = diff.reshape(diff.shape[0], -1).max(dim=-1).values
        result["per_position_max"] = per_pos.tolist()

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare deep attention captures")
    parser.add_argument("--captures-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--all-layers", action="store_true")
    args = parser.parse_args()

    captures_dir = Path(args.captures_dir)
    layers = list(range(20)) if args.all_layers else [args.layer]

    for layer_idx in layers:
        scenario = f"attn_deep_layer{layer_idx:02d}"
        print(f"\n{'=' * 70}")
        print(f"Deep Attention Comparison: Layer {layer_idx}")
        print(f"{'=' * 70}")

        dinfer = CaptureRegistry.load(captures_dir, "dinfer", scenario)
        vllm = CaptureRegistry.load(captures_dir, "vllm", scenario)

        if not dinfer.captures or not vllm.captures:
            print(
                f"  SKIP: missing captures (dinfer={len(dinfer.captures)}, vllm={len(vllm.captures)})"
            )
            continue

        # Compare each checkpoint in order
        first_divergence = None
        results = {}

        for key, description in CHECKPOINTS:
            if key not in dinfer.captures and key not in vllm.captures:
                print(f"  [--] {key:35s}  {description:30s}  MISSING (both)")
                continue
            if key not in dinfer.captures:
                print(f"  [--] {key:35s}  {description:30s}  MISSING (dinfer)")
                continue
            if key not in vllm.captures:
                print(f"  [--] {key:35s}  {description:30s}  MISSING (vllm)")
                continue

            result = compare(dinfer.captures[key], vllm.captures[key], key)
            results[key] = result

            if result.get("status") == "SHAPE_MISMATCH":
                icon = "!!"
                detail = f"shapes: {result['a_shape']} vs {result['e_shape']}"
            elif result["exact_match"]:
                icon = "=="
                detail = "EXACT"
            elif result["max_abs_diff"] < 0.01:
                icon = "~="
                detail = f"max_diff={result['max_abs_diff']:.2e}, cos={result.get('cosine_sim', 'N/A'):.6f}"
            else:
                icon = "XX"
                detail = f"max_diff={result['max_abs_diff']:.2e}, cos={result.get('cosine_sim', 'N/A'):.6f}"
                if first_divergence is None:
                    first_divergence = key

            differ_str = (
                f"({result.get('elements_differ', '?')}/{result.get('total_elements', '?')} differ)"
                if not result.get("exact_match")
                and result.get("status") != "SHAPE_MISMATCH"
                else ""
            )
            print(f"  [{icon}] {key:35s}  {detail:50s}  {differ_str}")

        # Summary
        print(f"\n  {'─' * 60}")
        if first_divergence:
            print(f"  FIRST DIVERGENCE: {first_divergence}")
            # Determine the cause
            a5_ok = (
                results.get("A5_q_after_rope", {}).get("exact_match")
                or results.get("A5_q_after_rope", {}).get("max_abs_diff", 999) < 0.001
            )
            a8_ok = (
                results.get("A8_attn_output_pre_oproj", {}).get("max_abs_diff", 999)
                < 0.01
            )

            if not a5_ok:
                print("  CAUSE: RoPE implementation difference (A5 diverges)")
            elif not a8_ok:
                print(
                    "  CAUSE: Attention kernel difference (A8 diverges despite A5/A6 matching)"
                )
            else:
                print("  CAUSE: O-proj difference (A9 diverges despite A8 matching)")
        else:
            print("  NO DIVERGENCE FOUND")

        # Save report
        output_dir = captures_dir / "comparisons"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"attn_deep_layer{layer_idx:02d}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
