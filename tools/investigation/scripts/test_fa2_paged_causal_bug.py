#!/usr/bin/env python3
"""Definitive test: does FA2 paged attention respect causal=False
with the EXACT shapes/params used by the real model?"""

import os
import sys

import torch

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

from dllm_plugin import register_dllm

register_dllm()
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
)

runner = llm.llm_engine.model_executor.driver_worker.model_runner
model = runner.model
base = model.model if hasattr(model, "model") else model
attn = base.layers[0].self_attn.attn

# Get the actual KV cache tensor
kv_cache = attn.kv_cache
print(
    f"KV cache shape: {kv_cache.shape}"
)  # [2, num_blocks, block_size, num_kv_heads, head_size]

# Run a forward pass to populate the cache
captures = {}
original_forward = base.layers[0].self_attn.attn.forward


def cap_forward(query, key, value, **kwargs):
    captures["q"] = query.detach().clone()
    captures["k"] = key.detach().clone() if key is not None else None
    captures["v"] = value.detach().clone() if value is not None else None
    result = original_forward(query, key, value, **kwargs)
    captures["out"] = result.detach().clone()
    return result


base.layers[0].self_attn.attn.forward = cap_forward

out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
base.layers[0].self_attn.attn.forward = original_forward

q = captures["q"]
k = captures["k"]
v = captures["v"]
vllm_out = captures["out"]
print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}")
print(f"vLLM output: {vllm_out.shape}")

# Now test with flash_attn_varlen_func directly using the EXACT Q/K/V
# First, read the block_table and slot_mapping from the forward context
# We can't get those easily, so let's use a simpler approach:
# Write K/V to a fresh cache page, then read with FA2

from vllm.vllm_flash_attn import flash_attn_varlen_func

num_kv_heads = 4
head_dim = 128
seq_len = q.shape[0]

# Reshape Q for FA2: [seq, num_heads, head_dim]
q_3d = q.view(seq_len, -1, head_dim)  # [4, 16, 128]
num_heads = q_3d.shape[1]

# Reshape K/V: [seq, num_kv_heads, head_dim]
k_3d = k.view(seq_len, num_kv_heads, head_dim)
v_3d = v.view(seq_len, num_kv_heads, head_dim)

# Create a small paged cache and write K/V into it
block_size = 16
cache_k = torch.zeros(
    1, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
)
cache_v = torch.zeros(
    1, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
)

# Write K/V into positions 0-3 of block 0
cache_k[0, :seq_len] = k_3d
cache_v[0, :seq_len] = v_3d

block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
seqused_k = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

out_bidir = torch.empty(
    seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
)
out_causal = torch.empty_like(out_bidir)

# Test causal=False with paged cache
flash_attn_varlen_func(
    q=q_3d.cuda(),
    k=cache_k,
    v=cache_v,
    out=out_bidir,
    cu_seqlens_q=cu_seqlens,
    max_seqlen_q=seq_len,
    seqused_k=seqused_k,
    max_seqlen_k=seq_len,
    softmax_scale=1.0 / head_dim**0.5,
    causal=False,
    block_table=block_table,
    fa_version=2,
)

# Test causal=True with paged cache
flash_attn_varlen_func(
    q=q_3d.cuda(),
    k=cache_k,
    v=cache_v,
    out=out_causal,
    cu_seqlens_q=cu_seqlens,
    max_seqlen_q=seq_len,
    seqused_k=seqused_k,
    max_seqlen_k=seq_len,
    softmax_scale=1.0 / head_dim**0.5,
    causal=True,
    block_table=block_table,
    fa_version=2,
)

# Also test SDPA (no paging)
import torch.nn.functional as F

n_rep = num_heads // num_kv_heads
q4d = q_3d.unsqueeze(0).permute(0, 2, 1, 3).float()
k4d = (
    k_3d.unsqueeze(0).permute(0, 2, 1, 3).repeat_interleave(n_rep, dim=1).float().cuda()
)
v4d = (
    v_3d.unsqueeze(0).permute(0, 2, 1, 3).repeat_interleave(n_rep, dim=1).float().cuda()
)
sdpa_b = F.scaled_dot_product_attention(q4d.cuda(), k4d, v4d, is_causal=False)
sdpa_c = F.scaled_dot_product_attention(q4d.cuda(), k4d, v4d, is_causal=True)

# Flatten for comparison
ob = out_bidir.view(seq_len, -1).cpu()
oc = out_causal.view(seq_len, -1).cpu()
sb = sdpa_b.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1).cpu()
sc = sdpa_c.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1).cpu()
vo = vllm_out.view(seq_len, -1).cpu()

print("\n=== Results ===")
print(
    f"FA2 paged causal=F vs causal=T: max_diff={torch.abs(ob.float() - oc.float()).max():.6e}"
)
print(
    f"  -> FA2 paged DOES{'' if torch.abs(ob.float() - oc.float()).max() > 0.001 else ' NOT'} respect causal flag"
)

print(
    f"\nvLLM actual vs FA2 paged causal=F: max_diff={torch.abs(vo.float() - ob.float()).max():.6e}"
)
print(
    f"vLLM actual vs FA2 paged causal=T: max_diff={torch.abs(vo.float() - oc.float()).max():.6e}"
)
print(
    f"vLLM actual vs SDPA causal=F:      max_diff={torch.abs(vo.float() - sb.float()).max():.6e}"
)
print(
    f"vLLM actual vs SDPA causal=T:      max_diff={torch.abs(vo.float() - sc.float()).max():.6e}"
)

print("\nvLLM actual matches: ", end="")
diffs = {
    "FA2_bidir": torch.abs(vo.float() - ob.float()).max().item(),
    "FA2_causal": torch.abs(vo.float() - oc.float()).max().item(),
    "SDPA_bidir": torch.abs(vo.float() - sb.float()).max().item(),
    "SDPA_causal": torch.abs(vo.float() - sc.float()).max().item(),
}
best = min(diffs, key=diffs.get)  # type: ignore[arg-type]
print(f"{best} (max_diff={diffs[best]:.6e})")
