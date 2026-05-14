#!/usr/bin/env python3
"""Check: are the K/V values in the paged cache correct for the 4-token pass?

Intercepts both the KV cache write and the attention read to verify
the cache contains the correct values.
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

# Get layer 0's attention
attn_layer = base.layers[0].self_attn
attn = attn_layer.attn  # The vLLM Attention object

# Capture what K/V are written to cache, and what the attention output is
captures = {}

# Hook on the Attention module's forward to capture K and V BEFORE cache write
original_attn_forward = attn.forward


def patched_attn_forward(query, key, value, **kwargs):
    captures["q_input"] = query.detach().cpu().clone()
    captures["k_input"] = key.detach().cpu().clone() if key is not None else None
    captures["v_input"] = value.detach().cpu().clone() if value is not None else None
    result = original_attn_forward(query, key, value, **kwargs)
    captures["attn_output"] = result.detach().cpu().clone()
    return result


attn.forward = patched_attn_forward

out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))

# The captures will have the LAST forward pass's values (the 4-token pass)
if "k_input" in captures and captures["k_input"] is not None:
    k = captures["k_input"]
    v = captures["v_input"]
    q = captures["q_input"]
    a = captures["attn_output"]

    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"V shape: {v.shape}")
    print(f"Attn output shape: {a.shape}")

    import torch.nn.functional as F

    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    n_rep = num_heads // num_kv_heads
    seq_len = q.shape[0]

    # K/V may be 2D [seq, hidden] or 3D [seq, heads, dim]
    if q.dim() == 2:
        q = q.view(seq_len, num_heads, head_dim)
    if k.dim() == 2:
        k = k.view(seq_len, num_kv_heads, head_dim)
    if v.dim() == 2:
        v = v.view(seq_len, num_kv_heads, head_dim)

    q4d = q.unsqueeze(0).permute(0, 2, 1, 3).float()
    k4d = k.unsqueeze(0).permute(0, 2, 1, 3).repeat_interleave(n_rep, dim=1).float()
    v4d = v.unsqueeze(0).permute(0, 2, 1, 3).repeat_interleave(n_rep, dim=1).float()

    sdpa_bidir = F.scaled_dot_product_attention(q4d, k4d, v4d, is_causal=False)
    sdpa_causal = F.scaled_dot_product_attention(q4d, k4d, v4d, is_causal=True)

    sb = sdpa_bidir.squeeze(0).permute(1, 0, 2).reshape(q.shape[0], -1)
    sc = sdpa_causal.squeeze(0).permute(1, 0, 2).reshape(q.shape[0], -1)
    a_flat = a.reshape(q.shape[0], -1)

    db = torch.abs(sb - a_flat.float()).max().item()
    dc = torch.abs(sc - a_flat.float()).max().item()

    print(f"\nAttn output vs SDPA bidirectional: max_diff={db:.6e}")
    print(f"Attn output vs SDPA causal:        max_diff={dc:.6e}")
    print(f"Matches: {'BIDIRECTIONAL' if db < dc else 'CAUSAL'}")

    # Also check: are K/V the same for all 4 positions?
    print("\nK values per position (head 0, dim 0):")
    for pos in range(min(4, k.shape[0])):
        print(f"  pos {pos}: K[0,0]={k[pos, 0, 0].item():.6f}")
else:
    print("No K/V captured!")
    print(f"Captures: {list(captures.keys())}")
