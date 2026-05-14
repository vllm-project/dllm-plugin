#!/usr/bin/env python3
"""Compare direct FA2 call vs engine attention path — same Q/K/V."""

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
attn_layer = base.layers[0].self_attn

caps = {}
orig = attn_layer.forward
_active = [False]  # Only activate during generate


def cap(positions, hidden_states):
    if not _active[0]:
        return orig(positions, hidden_states)
    nh = attn_layer.num_heads
    nkv = attn_layer.num_kv_heads
    hd = attn_layer.head_size

    qkv, _ = attn_layer.qkv_proj(hidden_states)
    q, k, v = qkv.split([nh * hd, nkv * hd, nkv * hd], dim=-1)
    nt = q.shape[0]
    q = q.view(nt, nh, hd)
    k = k.view(nt, nkv, hd)
    q = attn_layer.q_norm(q)
    k = attn_layer.k_norm(k)
    q = q.reshape(nt, nh * hd)
    k = k.reshape(nt, nkv * hd)
    q, k = attn_layer.rotary_emb(positions, q, k)

    caps["q"] = q.detach().cpu().clone()
    caps["k"] = k.detach().cpu().clone()
    caps["v"] = v.detach().cpu().clone()

    # Direct FA2 (no paging, no custom op)
    q3 = q.view(nt, nh, hd)
    k3 = k.view(nt, nkv, hd)
    v3 = v.view(nt, nkv, hd)
    out_direct = torch.empty(nt, nh, hd, dtype=q.dtype, device=q.device)
    cu = torch.tensor([0, nt], dtype=torch.int32, device=q.device)
    sl = torch.tensor([nt], dtype=torch.int32, device=q.device)
    flash_attn_varlen_func(
        q=q3,
        k=k3,
        v=v3,
        out=out_direct,
        cu_seqlens_q=cu,
        max_seqlen_q=nt,
        seqused_k=sl,
        max_seqlen_k=nt,
        softmax_scale=1.0 / hd**0.5,
        causal=False,
        fa_version=2,
    )
    caps["direct"] = out_direct.view(nt, -1).detach().cpu().clone()

    # Engine path (through self.attn custom op)
    engine_out = attn_layer.attn(query=q, key=k, value=v)
    caps["engine"] = engine_out.detach().cpu().clone()

    # Use direct output for the model forward
    output, _ = attn_layer.o_proj(out_direct.view(nt, -1))
    return output


attn_layer.forward = cap
_active[0] = True
llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
_active[0] = False
attn_layer.forward = orig

q, k, v = caps["q"], caps["k"], caps["v"]
direct = caps["direct"]
engine = caps["engine"]
nt = q.shape[0]

# SDPA reference
n_rep = 16 // 4
q4 = q.view(nt, 16, 128).unsqueeze(0).permute(0, 2, 1, 3).float().cuda()
k4 = (
    k.view(nt, 4, 128)
    .unsqueeze(0)
    .permute(0, 2, 1, 3)
    .repeat_interleave(n_rep, 1)
    .float()
    .cuda()
)
v4 = (
    v.view(nt, 4, 128)
    .unsqueeze(0)
    .permute(0, 2, 1, 3)
    .repeat_interleave(n_rep, 1)
    .float()
    .cuda()
)
sb = (
    F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    .squeeze(0)
    .permute(1, 0, 2)
    .reshape(nt, -1)
    .cpu()
)
sc = (
    F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
    .squeeze(0)
    .permute(1, 0, 2)
    .reshape(nt, -1)
    .cpu()
)

diff_de = torch.abs(direct.float() - engine.float()).max().item()
diff_db = torch.abs(direct.float() - sb.float()).max().item()
diff_dc = torch.abs(direct.float() - sc.float()).max().item()
diff_eb = torch.abs(engine.float() - sb.float()).max().item()
diff_ec = torch.abs(engine.float() - sc.float()).max().item()

print(f"Direct FA2 vs Engine attn: max_diff={diff_de:.6e}")
print(f"Direct FA2 vs SDPA bidir:  max_diff={diff_db:.6e}")
print(f"Direct FA2 vs SDPA causal: max_diff={diff_dc:.6e}")
print(f"Engine attn vs SDPA bidir: max_diff={diff_eb:.6e}")
print(f"Engine attn vs SDPA causal:max_diff={diff_ec:.6e}")

d_match = "BIDIR" if diff_db < diff_dc else "CAUSAL"
e_match = "BIDIR" if diff_eb < diff_ec else "CAUSAL"
print(f"\nDirect FA2 matches: {d_match}")
print(f"Engine attn matches: {e_match}")
