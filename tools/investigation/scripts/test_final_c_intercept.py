#!/usr/bin/env python3
"""FINAL TEST: intercept torch.ops._vllm_fa2_C.varlen_fwd directly."""

import os
import sys

import torch

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

# Intercept the actual C extension call
_orig_varlen_fwd = torch.ops._vllm_fa2_C.varlen_fwd
_calls = []


def _intercepted_varlen_fwd(
    q,
    k,
    v,
    out,
    cu_sq,
    cu_sk,
    seqused_k,
    gen,
    block_table,
    alibi,
    max_sq,
    max_sk,
    dropout,
    scale,
    is_bf16,
    causal,
    wl,
    wr,
    softcap,
    ret_sm,
    num_splits,
    generator,
):
    _calls.append(
        {
            "causal": causal,
            "nq": q.shape[0],
            "max_sq": max_sq,
            "has_bt": block_table is not None,
        }
    )
    return _orig_varlen_fwd(
        q,
        k,
        v,
        out,
        cu_sq,
        cu_sk,
        seqused_k,
        gen,
        block_table,
        alibi,
        max_sq,
        max_sk,
        dropout,
        scale,
        is_bf16,
        causal,
        wl,
        wr,
        softcap,
        ret_sm,
        num_splits,
        generator,
    )


torch.ops._vllm_fa2_C.varlen_fwd = _intercepted_varlen_fwd

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
llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))

from collections import Counter

patterns = Counter()
for c in _calls:
    patterns[f"causal={c['causal']} nq={c['nq']} max_sq={c['max_sq']}"] += 1

print(f"\nC extension calls: {len(_calls)}")
for p, n in patterns.most_common():
    print(f"  {p}  x{n}")
