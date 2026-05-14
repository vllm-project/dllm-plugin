#!/usr/bin/env python3
"""E2E comparison v2: capture full logits via model forward hook."""

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

# Capture hidden states after final norm, compute logits manually
caps = {}
orig_forward = type(model).forward


def cap_forward(self, input_ids=None, positions=None, **kwargs):
    result = orig_forward(self, input_ids=input_ids, positions=positions, **kwargs)
    # After the model forward, capture final norm output
    # The model.forward computes: embed -> layers -> norm -> logits
    # We need the hidden states after norm to compute logits
    # The result IS the logits (or IntermediateTensors)
    if isinstance(result, torch.Tensor) and result.dim() >= 2:
        caps["model_output"] = result.detach().cpu().clone()
    return result


type(model).forward = cap_forward
_active = [False]


# Also hook the final norm to get pre-logits hidden states
def hook_norm(module, input, output):
    if _active[0]:
        if isinstance(output, tuple):
            caps["final_hidden"] = output[0].detach().cpu().clone()
        else:
            caps["final_hidden"] = output.detach().cpu().clone()


base.norm.register_forward_hook(hook_norm)


# Hook lm_head
def hook_lm(module, input, output):
    if _active[0]:
        if isinstance(output, tuple):
            caps["lm_head_out"] = output[0].detach().cpu().clone()
        else:
            caps["lm_head_out"] = output.detach().cpu().clone()


model.lm_head.register_forward_hook(hook_lm)

_active[0] = True
out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
_active[0] = False
type(model).forward = orig_forward

print(f"Captured keys: {list(caps.keys())}")
for k, v in caps.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# Compute logits from final_hidden if lm_head wasn't captured
if "final_hidden" in caps and "lm_head_out" not in caps:
    # Manually compute: logits = hidden @ lm_head_weight.T
    weight = model.lm_head.weight.detach().cpu()
    hidden = caps["final_hidden"].float()
    logits = hidden @ weight.float().T
    print(f"\nManual logits: {logits.shape}")
elif "lm_head_out" in caps:
    logits = caps["lm_head_out"].float()
    print(f"\nCaptured logits: {logits.shape}")
else:
    print("No logits available!")
    sys.exit(1)

# Compute probabilities
probs = torch.softmax(logits, dim=-1)

# Load dInfer logits/probs
c = "/workspace/captures/first_block"
dp = torch.load(
    f"{c}/dinfer.first_block.L2_probs.global.softmax_probs.pt",
    map_location="cpu",
    weights_only=True,
).squeeze(0)
dl = torch.load(
    f"{c}/dinfer.first_block.L3_logits.global.raw_logits.pt",
    map_location="cpu",
    weights_only=True,
).squeeze(0)

print(f"\ndInfer probs: {dp.shape}")
print(f"vLLM probs: {probs.shape}")

# Align shapes
num_pos = min(probs.shape[0], dp.shape[0])
vocab = min(probs.shape[-1], dp.shape[-1])
dp = dp[:num_pos, :vocab].float()
vp = probs[:num_pos, :vocab].float()
dl = dl[:num_pos, :vocab].float()
vl = logits[:num_pos, :vocab].float()

print(f"Comparing {num_pos} positions, {vocab} vocab size")

# === Top predictions ===
print("\n=== Top-5 Predictions ===")
print(f"{'Pos':>3} | {'dInfer top-1 (prob)':>25} | {'vLLM top-1 (prob)':>25} | Match")
for pos in range(num_pos):
    d_top = dp[pos].argmax().item()
    v_top = vp[pos].argmax().item()
    d_prob = dp[pos, d_top].item()
    v_prob = vp[pos, v_top].item()
    match = "YES" if d_top == v_top else "NO"
    print(
        f"{pos:>3} | {d_top:>10} ({d_prob:.4f}) | {v_top:>10} ({v_prob:.4f}) | {match}"
    )

# === KL Divergence ===
print("\n=== KL Divergence (per position) ===")
eps = 1e-10
for pos in range(num_pos):
    p = torch.clamp(dp[pos], min=eps)
    q = torch.clamp(vp[pos], min=eps)
    kl = (p * torch.log(p / q)).sum().item()
    print(f"  pos {pos}: KL = {kl:.6f}")

kl_avg = (
    sum(
        (
            torch.clamp(dp[i], min=eps)
            * torch.log(torch.clamp(dp[i], min=eps) / torch.clamp(vp[i], min=eps))
        )
        .sum()
        .item()
        for i in range(num_pos)
    )
    / num_pos
)
print(f"  avg:   KL = {kl_avg:.6f}")

# === Logits comparison ===
print("\n=== Logits Comparison ===")
diff = torch.abs(dl - vl)
cos = F.cosine_similarity(dl.flatten().unsqueeze(0), vl.flatten().unsqueeze(0)).item()
print(f"  max_diff: {diff.max():.2e}")
print(f"  mean_diff: {diff.mean():.2e}")
print(f"  cos_sim: {cos:.6f}")

# Per-position logits cosine similarity
print("\n=== Per-Position Logits Cosine Similarity ===")
for pos in range(num_pos):
    c = F.cosine_similarity(dl[pos].unsqueeze(0), vl[pos].unsqueeze(0)).item()
    print(f"  pos {pos}: cos_sim = {c:.6f}")

# === Top-k agreement ===
for k in [1, 5, 10, 100]:
    d_topk = set()
    v_topk = set()
    for pos in range(num_pos):
        d_topk.update(torch.topk(dp[pos], k).indices.tolist())
        v_topk.update(torch.topk(vp[pos], k).indices.tolist())
    overlap = len(d_topk & v_topk) / len(d_topk | v_topk)
    print(f"  top-{k:>3} Jaccard overlap: {overlap * 100:.1f}%")

print("\n=== Generated text ===")
print(f"  vLLM: {out[0].outputs[0].text!r}")
