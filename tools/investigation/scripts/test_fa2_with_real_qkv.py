#!/usr/bin/env python3
"""Test FA2 with REAL model Q/K/V — does causal flag work with actual values?"""

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
from vllm.vllm_flash_attn import flash_attn_varlen_func

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
attn_mod = base.layers[0].self_attn

# Capture Q/K/V after RoPE
caps = {}
original_forward = attn_mod.forward


def cap_forward(positions, hidden_states):
    qkv, _ = attn_mod.qkv_proj(hidden_states)
    num_heads = attn_mod.num_heads
    num_kv_heads = attn_mod.num_kv_heads
    head_size = attn_mod.head_size
    q, k, v = qkv.split(
        [num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size],
        dim=-1,
    )
    nt = q.shape[0]
    q = q.view(nt, num_heads, head_size)
    k = k.view(nt, num_kv_heads, head_size)
    q = attn_mod.q_norm(q)
    k = attn_mod.k_norm(k)
    q = q.reshape(nt, num_heads * head_size)
    k = k.reshape(nt, num_kv_heads * head_size)
    q, k = attn_mod.rotary_emb(positions, q, k)

    caps["q"] = q.detach().clone()
    caps["k"] = k.detach().clone()
    caps["v"] = v.detach().clone()

    # Call original to get actual output
    return original_forward(positions, hidden_states)


attn_mod.forward = cap_forward
llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
attn_mod.forward = original_forward

q = caps["q"]
k = caps["k"]
v = caps["v"]

seq_len = q.shape[0]
num_heads = 16
num_kv_heads = 4
head_dim = 128

# Reshape Q/K/V for FA2: [seq, heads, dim]
q3 = q.view(seq_len, num_heads, head_dim)
k3 = k.view(seq_len, num_kv_heads, head_dim)
v3 = v.view(seq_len, num_kv_heads, head_dim)

# Call FA2 DIRECTLY (not through paged cache) with same Q/K/V
# Use cu_seqlens, NOT block_table — NO paged cache
cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
sl = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

out_f = torch.empty(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
out_t = torch.empty_like(out_f)

# Test WITHOUT block_table (contiguous K/V)
flash_attn_varlen_func(
    q=q3,
    k=k3,
    v=v3,
    out=out_f,
    cu_seqlens_q=cu,
    max_seqlen_q=seq_len,
    seqused_k=sl,
    max_seqlen_k=seq_len,
    softmax_scale=1.0 / head_dim**0.5,
    causal=False,
    fa_version=2,
)

flash_attn_varlen_func(
    q=q3,
    k=k3,
    v=v3,
    out=out_t,
    cu_seqlens_q=cu,
    max_seqlen_q=seq_len,
    seqused_k=sl,
    max_seqlen_k=seq_len,
    softmax_scale=1.0 / head_dim**0.5,
    causal=True,
    fa_version=2,
)

d = torch.abs(out_f.float() - out_t.float()).max().item()
print(f"FA2 no-paging, real Q/K/V: bidir vs causal max_diff={d:.6e}")
print(f"  -> {'RESPECTS' if d > 0.001 else 'IGNORES'} causal flag")

# Also compare with PyTorch SDPA
import torch.nn.functional as F

n_rep = num_heads // num_kv_heads
q4 = q3.unsqueeze(0).permute(0, 2, 1, 3).float()
k4 = k3.unsqueeze(0).permute(0, 2, 1, 3).repeat_interleave(n_rep, dim=1).float()
v4 = v3.unsqueeze(0).permute(0, 2, 1, 3).repeat_interleave(n_rep, dim=1).float()

sdpa_b = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
sdpa_c = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
sb = sdpa_b.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1).cpu()
sc = sdpa_c.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1).cpu()

of = out_f.view(seq_len, -1).cpu()
ot = out_t.view(seq_len, -1).cpu()

print(
    f"\nFA2 no-paging causal=F vs SDPA bidir: {torch.abs(of.float() - sb.float()).max():.6e}"
)
print(
    f"FA2 no-paging causal=T vs SDPA causal: {torch.abs(ot.float() - sc.float()).max():.6e}"
)
print(
    f"FA2 no-paging causal=F vs SDPA causal: {torch.abs(of.float() - sc.float()).max():.6e}"
)
