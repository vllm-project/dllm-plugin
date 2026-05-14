#!/usr/bin/env python3
"""E2E comparison: logits, probabilities, KL divergence, token match."""

import os
import sys

import torch

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

from dllm_plugin import register_dllm

register_dllm()
import torch.nn.functional as F
from vllm import LLM, SamplingParams

llm = LLM(
    model="/workspace/llada2-mini",
    trust_remote_code=True,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=256,
    max_num_seqs=1,
    gpu_memory_utilization=0.9,
    dtype="bfloat16",
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
)

model = llm.llm_engine.model_executor.driver_worker.model_runner.model
base = model.model if hasattr(model, "model") else model

# Capture final hidden states and compute logits
caps = {}


def hook_final_norm(module, input, output):
    if isinstance(output, tuple):
        caps["final_norm"] = output[0].detach().cpu().clone()
    else:
        caps["final_norm"] = output.detach().cpu().clone()


base.norm.register_forward_hook(hook_final_norm)

# Capture logits from logits_processor (or lm_head)
for name in ["logits_processor", "lm_head"]:
    mod = getattr(model, name, None)
    if mod is not None:

        def make_hook(n):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    t = output[0]
                else:
                    t = output
                if isinstance(t, torch.Tensor) and t.dim() >= 2:
                    caps[f"logits_{n}"] = t.detach().cpu().clone()

            return hook

        mod.register_forward_hook(make_hook(name))

out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))

# Save captured tensors
output_dir = "/workspace/captures/first_block"
os.makedirs(output_dir, exist_ok=True)

logits_key = next((k for k in caps if k.startswith("logits_")), None)
if logits_key:
    logits = caps[logits_key]
    print(f"Captured logits from: {logits_key}")
    torch.save(logits, f"{output_dir}/vllm.first_block.L3_logits.global.raw_logits.pt")
    probs = torch.softmax(logits.float(), dim=-1)
    torch.save(probs, f"{output_dir}/vllm.first_block.L2_probs.global.softmax_probs.pt")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")

    # Top-5 predictions per position
    print("\n=== vLLM Top-5 Predictions ===")
    for pos in range(min(4, probs.shape[0])):
        topk = torch.topk(probs[pos], 5)
        tokens = topk.indices.tolist()
        p = topk.values.tolist()
        print(f"  pos {pos}: {[(t, round(v, 4)) for t, v in zip(tokens, p)]}")

    # Load dInfer logits/probs for comparison
    dinfer_probs_path = (
        f"{output_dir}/dinfer.first_block.L2_probs.global.softmax_probs.pt"
    )
    dinfer_logits_path = (
        f"{output_dir}/dinfer.first_block.L3_logits.global.raw_logits.pt"
    )

    if os.path.exists(dinfer_probs_path):
        dp = torch.load(
            dinfer_probs_path, map_location="cpu", weights_only=True
        ).squeeze(0)
        vp = probs

        # Align shapes
        if dp.shape != vp.shape:
            min_vocab = min(dp.shape[-1], vp.shape[-1])
            dp = dp[..., :min_vocab]
            vp = vp[..., :min_vocab]

        print("\n=== KL Divergence (dInfer || vLLM) ===")
        eps = 1e-10
        for pos in range(min(4, dp.shape[0])):
            p = torch.clamp(dp[pos].float(), min=eps)
            q = torch.clamp(vp[pos].float(), min=eps)
            kl = (p * torch.log(p / q)).sum().item()
            print(f"  pos {pos}: KL={kl:.6f}")

        # Overall KL
        p_all = torch.clamp(dp.float(), min=eps)
        q_all = torch.clamp(vp.float(), min=eps)
        kl_total = (p_all * torch.log(p_all / q_all)).sum().item() / dp.shape[0]
        print(f"  avg:   KL={kl_total:.6f}")

        # Top-1 token agreement
        print("\n=== Top-1 Token Agreement ===")
        d_top1 = dp.argmax(dim=-1)
        v_top1 = vp.argmax(dim=-1)
        for pos in range(min(4, dp.shape[0])):
            match = "MATCH" if d_top1[pos] == v_top1[pos] else "MISMATCH"
            print(
                f"  pos {pos}: dInfer={d_top1[pos].item()} vLLM={v_top1[pos].item()} {match}"
            )
        agreement = (d_top1 == v_top1).float().mean().item()
        print(f"  overall: {agreement * 100:.1f}% agreement")

        # Top-5 agreement
        d_top5 = torch.topk(dp, 5, dim=-1).indices
        v_top5 = torch.topk(vp, 5, dim=-1).indices
        top5_match = 0
        for pos in range(dp.shape[0]):
            d_set = set(d_top5[pos].tolist())
            v_set = set(v_top5[pos].tolist())
            top5_match += len(d_set & v_set) / 5
        top5_match /= dp.shape[0]
        print(f"  top-5 overlap: {top5_match * 100:.1f}%")

    if os.path.exists(dinfer_logits_path):
        dl = torch.load(
            dinfer_logits_path, map_location="cpu", weights_only=True
        ).squeeze(0)
        vl = caps["logits"]
        if dl.shape != vl.shape:
            min_vocab = min(dl.shape[-1], vl.shape[-1])
            dl = dl[..., :min_vocab]
            vl = vl[..., :min_vocab]
        diff = torch.abs(dl.float() - vl.float())
        cos = F.cosine_similarity(
            dl.float().flatten().unsqueeze(0), vl.float().flatten().unsqueeze(0)
        ).item()
        print("\n=== Logits Comparison ===")
        print(f"  max_diff: {diff.max():.2e}")
        print(f"  mean_diff: {diff.mean():.2e}")
        print(f"  cos_sim: {cos:.6f}")
else:
    print("Logits not captured!")

# Generated text
if out:
    print("\n=== Generated Output ===")
    print(f"  vLLM: {out[0].outputs[0].text!r}")
    print(f"  tokens: {out[0].outputs[0].token_ids}")
