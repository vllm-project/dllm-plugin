#!/usr/bin/env python3
"""Call the C extension directly with causal=False to verify behavior."""

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
from vllm.forward_context import get_forward_context

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

model = llm.llm_engine.model_executor.driver_worker.model_runner.model
base = model.model if hasattr(model, "model") else model
attn = base.layers[0].self_attn.attn

# Capture Q, KV cache, block_table from real forward
caps = {}
orig_attn_fwd = attn.forward


def cap(query, key, value, **kw):
    result = orig_attn_fwd(query, key, value, **kw)
    caps["q"] = query.detach().clone()
    caps["k"] = key.detach().clone() if key is not None else None
    caps["out"] = result.detach().clone()
    caps["kv_cache"] = attn.kv_cache.detach().clone()
    ctx = get_forward_context()
    md = ctx.attn_metadata
    if isinstance(md, dict):
        m = md[attn.layer_name]
        caps["block_table"] = m.block_table.detach().clone()
        caps["query_start_loc"] = m.query_start_loc.detach().clone()
        caps["seq_lens"] = m.seq_lens.detach().clone()
        caps["max_query_len"] = m.max_query_len
        caps["max_seq_len"] = m.max_seq_len
        caps["slot_mapping"] = (
            m.slot_mapping.detach().clone()
            if hasattr(m, "slot_mapping") and m.slot_mapping is not None
            else None
        )
    return result


attn.forward = cap
llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
attn.forward = orig_attn_fwd

q = caps["q"]
kv = caps["kv_cache"]
k_cache = kv[0]  # [num_blocks, block_size, num_kv_heads, head_size]
v_cache = kv[1]
bt = caps["block_table"]
cu = caps["query_start_loc"]
sl = caps["seq_lens"]

seq_len = q.shape[0]
num_heads = 16
head_dim = 128
q_3d = q.view(seq_len, num_heads, head_dim)

print(f"q: {q_3d.shape}, k_cache: {k_cache.shape}, block_table: {bt.shape}")
print(f"cu_seqlens: {cu.tolist()}, seq_lens: {sl.tolist()}")

# Call C extension DIRECTLY
out_false = torch.empty_like(q_3d)
out_true = torch.empty_like(q_3d)

dummy_cu_k = torch.zeros(2, dtype=torch.int32, device="cuda")

print("\nCalling torch.ops._vllm_fa2_C.varlen_fwd with causal=False...")
torch.ops._vllm_fa2_C.varlen_fwd(
    q_3d,
    k_cache,
    v_cache,
    out_false,
    cu,
    dummy_cu_k,
    sl,
    None,
    bt,
    None,
    caps["max_query_len"],
    caps["max_seq_len"],
    0.0,
    1.0 / head_dim**0.5,
    False,
    False,  # is_bf16=False, causal=False
    -1,
    -1,  # window
    0.0,  # softcap
    False,  # return_softmax
    0,
    None,  # num_splits, generator
)

print("Calling torch.ops._vllm_fa2_C.varlen_fwd with causal=True...")
torch.ops._vllm_fa2_C.varlen_fwd(
    q_3d,
    k_cache,
    v_cache,
    out_true,
    cu,
    dummy_cu_k,
    sl,
    None,
    bt,
    None,
    caps["max_query_len"],
    caps["max_seq_len"],
    0.0,
    1.0 / head_dim**0.5,
    False,
    True,  # is_bf16=False, causal=True
    -1,
    -1,
    0.0,
    False,
    0,
    None,
)

vllm_out = caps["out"].view(seq_len, -1)
of = out_false.view(seq_len, -1)
ot = out_true.view(seq_len, -1)

df = torch.abs(of.float() - vllm_out.float()).max().item()
dt = torch.abs(ot.float() - vllm_out.float()).max().item()

print(f"\nC ext causal=False vs vLLM: max_diff={df:.6e}")
print(f"C ext causal=True  vs vLLM: max_diff={dt:.6e}")
print(f"C ext bidir vs causal: max_diff={torch.abs(of.float() - ot.float()).max():.6e}")
print(f"vLLM matches: {'BIDIRECTIONAL' if df < dt else 'CAUSAL'}")
