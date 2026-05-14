#!/usr/bin/env python3
"""Call FA2 directly with model's Q/K/V and compare with vLLM output."""

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
)

runner = llm.llm_engine.model_executor.driver_worker.model_runner
model = runner.model
base = model.model if hasattr(model, "model") else model

# Capture EVERYTHING from layer 0 attention: Q, K, V, kv_cache, attn_metadata, and output
caps = {}

# Can't monkey-patch get_attention_context due to custom op closure.
# Instead, capture inside the attention layer itself
attn_layer = base.layers[0].self_attn
original_attn_call = attn_layer.attn.forward
_n = [0]


def capture_attn(query, key, value, **kwargs):
    _n[0] += 1
    result = original_attn_call(query, key, value, **kwargs)
    # Only capture the LAST call (real inference, not warmup)
    caps["q"] = query.detach().cpu().clone()
    caps["k"] = key.detach().cpu().clone() if key is not None else None
    caps["v"] = value.detach().cpu().clone() if value is not None else None
    caps["out"] = result.detach().cpu().clone()
    # Also capture kv_cache and metadata
    kv = attn_layer.attn.kv_cache
    if kv is not None:
        caps["kv_cache_shape"] = kv.shape
        caps["kv_cache_k"] = kv[0].detach().cpu().clone()  # key cache
        caps["kv_cache_v"] = kv[1].detach().cpu().clone()  # value cache
    # Get metadata via forward context
    ctx = get_forward_context()
    if ctx is not None and ctx.attn_metadata is not None:
        md = ctx.attn_metadata
        if isinstance(md, dict):
            layer_name = attn_layer.attn.layer_name
            m = md.get(layer_name)
            if m is not None:
                caps["meta_causal"] = m.causal
                caps["meta_num_actual_tokens"] = m.num_actual_tokens
                caps["meta_max_query_len"] = m.max_query_len
                caps["meta_seq_lens"] = (
                    m.seq_lens.cpu().tolist()
                    if hasattr(m.seq_lens, "tolist")
                    else str(m.seq_lens)
                )
                caps["meta_block_table"] = (
                    m.block_table.cpu().clone() if m.block_table is not None else None
                )
                caps["meta_query_start_loc"] = (
                    m.query_start_loc.cpu().tolist()
                    if m.query_start_loc is not None
                    else None
                )
                caps["meta_slot_mapping"] = (
                    m.slot_mapping.cpu().clone()
                    if hasattr(m, "slot_mapping") and m.slot_mapping is not None
                    else None
                )
    return result


attn_layer.attn.forward = capture_attn
out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
attn_layer.attn.forward = original_attn_call

print(f"Captured {len(caps)} items in {_n[0]} calls")

# Now replay with FA2 directly using the captured parameters
q = caps["q"].cuda()
k_cache = caps["kv_cache_k"].cuda()
v_cache = caps["kv_cache_v"].cuda()
vllm_out = caps["out"]
bt = caps.get("meta_block_table")

print(f"Q: {q.shape}")
print(f"KV cache: {k_cache.shape}")
print(f"meta_causal: {caps.get('meta_causal')}")
print(f"meta_num_actual_tokens: {caps.get('meta_num_actual_tokens')}")
print(f"meta_seq_lens: {caps.get('meta_seq_lens')}")
print(f"meta_block_table: {bt.shape if bt is not None else None}")
print(f"meta_query_start_loc: {caps.get('meta_query_start_loc')}")

if bt is not None:
    seq_len = q.shape[0]
    num_heads = 16
    head_dim = 128

    # Q is [seq, hidden] -> reshape to [seq, heads, dim]
    q_3d = q.view(seq_len, num_heads, head_dim)

    cu_seqlens = torch.tensor(
        caps["meta_query_start_loc"], dtype=torch.int32, device="cuda"
    )
    seqused_k = torch.tensor(
        caps["meta_seq_lens"][:1], dtype=torch.int32, device="cuda"
    )
    max_q = caps["meta_max_query_len"]

    out_test = torch.empty(
        seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )

    # Call FA2 with causal=False using EXACT same cache and block_table
    flash_attn_varlen_func(
        q=q_3d,
        k=k_cache,
        v=v_cache,
        out=out_test,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_q,
        seqused_k=seqused_k,
        max_seqlen_k=seqused_k[0].item(),
        softmax_scale=1.0 / head_dim**0.5,
        causal=False,
        block_table=bt.cuda(),
        fa_version=2,
    )

    out_flat = out_test.view(seq_len, -1).cpu()
    vllm_flat = vllm_out.view(seq_len, -1).cpu()

    diff = torch.abs(out_flat.float() - vllm_flat.float())
    print(f"\nDirect FA2(causal=False) vs vLLM actual: max_diff={diff.max():.6e}")

    # Also test causal=True
    out_causal = torch.empty_like(out_test)
    flash_attn_varlen_func(
        q=q_3d,
        k=k_cache,
        v=v_cache,
        out=out_causal,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_q,
        seqused_k=seqused_k,
        max_seqlen_k=seqused_k[0].item(),
        softmax_scale=1.0 / head_dim**0.5,
        causal=True,
        block_table=bt.cuda(),
        fa_version=2,
    )

    oc_flat = out_causal.view(seq_len, -1).cpu()
    diff_c = torch.abs(oc_flat.float() - vllm_flat.float())
    print(f"Direct FA2(causal=True)  vs vLLM actual: max_diff={diff_c.max():.6e}")
    print(
        f"Direct FA2 bidir vs causal: max_diff={torch.abs(out_flat.float() - oc_flat.float()).max():.6e}"
    )
    print(
        f"\nvLLM matches: {'BIDIRECTIONAL' if diff.max() < diff_c.max() else 'CAUSAL'}"
    )
