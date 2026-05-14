#!/usr/bin/env python3
"""Trace every model forward pass to determine which are warmup vs real."""

import os
import sys
import traceback

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

from dllm_plugin import register_dllm

register_dllm()

# Patch the model's forward to log every call with a stack trace snippet
import vllm.v1.attention.backends.flash_attn as fm

_orig = fm.flash_attn_varlen_func
_n = [0]


def _trace(*args, **kwargs):
    q = kwargs.get("q")
    nq = q.shape[0] if q is not None else "?"
    causal = kwargs.get("causal", "?")

    if _n[0] % 20 == 0:  # Only log first call of each 20-layer pass
        # Get caller stack to identify warmup vs real
        stack = traceback.extract_stack()
        callers = [
            f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}" for f in stack[-8:-1]
        ]
        is_dummy = any(
            "dummy" in c.lower() or "profile" in c.lower() or "warmup" in c.lower()
            for c in callers
        )
        print(
            f"[PASS {_n[0] // 20}] nq={nq} causal={causal} dummy={is_dummy} stack={' <- '.join(callers[-4:])}",
            flush=True,
        )
    _n[0] += 1
    return _orig(*args, **kwargs)


fm.flash_attn_varlen_func = _trace

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

print("--- LLM created ---", flush=True)
print(f"Calls during init: {_n[0]}", flush=True)

out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
print(f"Total calls: {_n[0]}", flush=True)
