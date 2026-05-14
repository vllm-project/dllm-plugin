#!/usr/bin/env python3
"""Self-consistency test: does vLLM's FlashAttention output match
torch SDPA bidirectional on the SAME Q/K/V?

This test captures Q/K/V AFTER RoPE (right before the attention call)
and the attention output, then computes SDPA bidirectional and causal
manually to compare.
"""

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
attn = base.layers[0].self_attn

# Monkey-patch the attention forward to capture Q/K/V after RoPE
original_forward = attn.forward
caps = {}


def capturing_forward(positions, hidden_states):
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    head_size = attn.head_size

    # QKV projection
    qkv, _ = attn.qkv_proj(hidden_states)
    q, k, v = qkv.split(
        [num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size],
        dim=-1,
    )

    num_tokens = q.shape[0]
    q = q.view(num_tokens, num_heads, head_size)
    k = k.view(num_tokens, num_kv_heads, head_size)

    # Q/K normalization
    q = attn.q_norm(q)
    k = attn.k_norm(k)

    q = q.reshape(num_tokens, num_heads * head_size)
    k = k.reshape(num_tokens, num_kv_heads * head_size)

    # RoPE
    q, k = attn.rotary_emb(positions, q, k)

    # Capture post-RoPE Q, K, V
    caps["q"] = q.detach().cpu().clone()
    caps["k"] = k.detach().cpu().clone()
    caps["v"] = v.detach().cpu().clone()

    # Call the real attention
    from vllm.forward_context import get_forward_context

    from dllm_plugin.forward_context import get_num_prefix_tokens_list

    num_prefix_tokens_list = get_num_prefix_tokens_list()
    if num_prefix_tokens_list is not None:
        context = get_forward_context()
        attn_metadata_dict = context.attn_metadata
        if isinstance(attn_metadata_dict, dict):
            layer_name = getattr(attn, "layer_name", None) or attn.attn.layer_name
            attn_metadata = attn_metadata_dict.get(layer_name)
        else:
            attn_metadata = attn_metadata_dict
        attn_output = attn._forward_concatenated(
            query=q,
            key=k,
            value=v,
            attn_metadata=attn_metadata,
            num_prefix_tokens_list=num_prefix_tokens_list,
        )
    else:
        attn_output = attn.attn(query=q, key=k, value=v)

    caps["attn_out"] = attn_output.detach().cpu().clone()

    output, _ = attn.o_proj(attn_output)
    return output


attn.forward = capturing_forward

out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
attn.forward = original_forward

q = caps["q"]
k = caps["k"]
v = caps["v"]
vllm_out = caps["attn_out"]

print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}")
print(f"Attn out: {vllm_out.shape}")

# Reshape for SDPA: [batch=1, heads, seq, dim]
seq_len = q.shape[0]
num_heads = 16
num_kv_heads = 4
head_dim = 128
n_rep = num_heads // num_kv_heads

q_3d = q.view(seq_len, num_heads, head_dim)
k_3d = k.view(seq_len, num_kv_heads, head_dim)
v_3d = v.view(seq_len, num_kv_heads, head_dim)

q4d = q_3d.unsqueeze(0).permute(0, 2, 1, 3).float().cuda()
k4d = (
    k_3d.unsqueeze(0).permute(0, 2, 1, 3).repeat_interleave(n_rep, dim=1).float().cuda()
)
v4d = (
    v_3d.unsqueeze(0).permute(0, 2, 1, 3).repeat_interleave(n_rep, dim=1).float().cuda()
)

sdpa_bidir = F.scaled_dot_product_attention(q4d, k4d, v4d, is_causal=False)
sdpa_causal = F.scaled_dot_product_attention(q4d, k4d, v4d, is_causal=True)

sb = sdpa_bidir.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1).cpu()
sc = sdpa_causal.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1).cpu()

diff_b = torch.abs(sb.float() - vllm_out.float()).max().item()
diff_c = torch.abs(sc.float() - vllm_out.float()).max().item()
cos_b = F.cosine_similarity(
    sb.float().flatten().unsqueeze(0), vllm_out.float().flatten().unsqueeze(0)
).item()
cos_c = F.cosine_similarity(
    sc.float().flatten().unsqueeze(0), vllm_out.float().flatten().unsqueeze(0)
).item()

print(f"\nvLLM attn vs SDPA bidirectional: max_diff={diff_b:.6e}, cos_sim={cos_b:.6f}")
print(f"vLLM attn vs SDPA causal:        max_diff={diff_c:.6e}, cos_sim={cos_c:.6f}")

# Per position
print("\nPer-position max_diff vs bidirectional:")
for pos in range(seq_len):
    db = torch.abs(sb[pos].float() - vllm_out[pos].float()).max().item()
    dc = torch.abs(sc[pos].float() - vllm_out[pos].float()).max().item()
    print(f"  pos {pos}: bidir={db:.6e}, causal={dc:.6e}")

print(f"\nvLLM matches: {'BIDIRECTIONAL' if diff_b < diff_c else 'CAUSAL'}")
