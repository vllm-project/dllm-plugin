#!/usr/bin/env python3
"""Force causal=False at the kernel level and verify output changes."""

import os
import sys

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

# Register custom op that wraps flash_attn_varlen_func with forced causal=False
import vllm.v1.attention.backends.flash_attn as fa_mod

_orig_impl_forward = (
    fa_mod.FlashAttentionImpl.forward.__wrapped__
    if hasattr(fa_mod.FlashAttentionImpl.forward, "__wrapped__")
    else None
)

# Patch at the flash_attn_varlen_func level
_real_flash = fa_mod.flash_attn_varlen_func
_call_n = [0]


def _forced_noncausal(*args, **kwargs):
    orig_causal = kwargs.get("causal")
    kwargs["causal"] = False  # FORCE non-causal
    q = kwargs.get("q")
    nq = q.shape[0] if q is not None else "?"
    if _call_n[0] < 100 and orig_causal is not None:
        print(f"[FORCED] causal: {orig_causal} -> False, nq={nq}", flush=True)
        _call_n[0] += 1
    return _real_flash(*args, **kwargs)


# Replace at module level
fa_mod.flash_attn_varlen_func = _forced_noncausal

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

# Capture layer 0 attention output
runner = llm.llm_engine.model_executor.driver_worker.model_runner
model = runner.model
base = model.model if hasattr(model, "model") else model

captures = {}


def hook(module, input, output):
    captures["attn_out"] = output.detach().cpu().clone()


base.layers[0].self_attn.register_forward_hook(hook)

out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))

print(f"\nForced calls: {_call_n[0]}")
print(f"Attn_out captured: {'attn_out' in captures}")
if "attn_out" in captures:
    va = captures["attn_out"]
    print(
        f"Shape: {va.shape}, mean: {va.float().mean():.6f}, std: {va.float().std():.6f}"
    )
    print(f"Non-zero: {(va != 0).sum().item()} / {va.numel()}")
    # Check per-position variance (causal has pos0 different from bidir)
    for i in range(min(4, va.shape[0])):
        print(
            f"  pos {i}: mean={va[i].float().mean():.6f}, std={va[i].float().std():.6f}"
        )
