#!/usr/bin/env python3
"""Test: intercept flash_attn_varlen_func to verify causal flag at kernel level."""

import os
import sys

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

import vllm.v1.attention.backends.flash_attn as fm

_orig = fm.flash_attn_varlen_func
_calls = []


def _intercept(*args, **kwargs):
    causal = kwargs.get("causal", "NOT_SET")
    q = kwargs.get("q", args[0] if args else None)
    nq = q.shape[0] if q is not None else "?"
    bt = kwargs.get("block_table")
    _calls.append({"causal": causal, "nq": nq, "paged": bt is not None})
    return _orig(*args, **kwargs)


fm.flash_attn_varlen_func = _intercept

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
    patterns[f"causal={c['causal']} nq={c['nq']} paged={c['paged']}"] += 1

print(f"\nTotal flash_attn_varlen_func calls: {len(_calls)}")
for p, n in patterns.most_common():
    print(f"  {p}  x{n}")

if not _calls:
    print("  NO CALLS INTERCEPTED - module reference not captured")
