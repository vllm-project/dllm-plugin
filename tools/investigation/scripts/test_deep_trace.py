#!/usr/bin/env python3
"""Deep trace: full stack + slot_mapping for every attention forward pass."""

import os
import sys
import traceback

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

from dllm_plugin import register_dllm

register_dllm()

# Intercept unified_kv_cache_update to check slot_mapping
import vllm.model_executor.layers.attention.attention as attn_mod

_orig_kv_update = attn_mod.unified_kv_cache_update
_kv_updates = []


def _traced_kv_update(key, value, layer_name):
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    slot_mapping = None
    if ctx and isinstance(ctx.slot_mapping, dict):
        sm = ctx.slot_mapping.get(layer_name)
        if sm is not None:
            slot_mapping = sm.cpu().tolist()

    if len(_kv_updates) < 5:
        nk = key.shape[0] if key is not None else "?"
        sm_preview = slot_mapping[:8] if slot_mapping else "N/A"
        all_neg1 = all(s == -1 for s in slot_mapping) if slot_mapping else "N/A"
        print(
            f"[KV_UPDATE] layer={layer_name.split('.')[-2]}.{layer_name.split('.')[-1]} nk={nk} slot_mapping={sm_preview}... all_neg1={all_neg1}",
            flush=True,
        )

    _kv_updates.append(
        {
            "layer": layer_name,
            "nk": key.shape[0] if key is not None else 0,
            "slot_mapping": slot_mapping,
        }
    )
    return _orig_kv_update(key, value, layer_name)


attn_mod.unified_kv_cache_update = _traced_kv_update

# Also intercept flash_attn to get full stack traces
import vllm.v1.attention.backends.flash_attn as fm

_orig_fa = fm.flash_attn_varlen_func
_fa_calls = []


def _traced_fa(*args, **kwargs):
    nq = kwargs.get("q").shape[0]  # type: ignore[union-attr]
    causal = kwargs.get("causal", "?")

    # Get stack trace to identify caller
    stack = traceback.extract_stack()
    callers = [f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}" for f in stack]

    # Identify key callers
    is_dummy = any("dummy" in c.lower() for c in callers)
    is_profile = any("profile" in c.lower() for c in callers)
    is_warmup = any(
        "warmup" in c.lower() or "kernel_warmup" in c.lower() for c in callers
    )
    is_generate = any("generate" in c.lower() for c in callers)

    entry = {
        "nq": nq,
        "causal": causal,
        "dummy": is_dummy,
        "profile": is_profile,
        "warmup": is_warmup,
        "generate": is_generate,
    }
    _fa_calls.append(entry)

    if len(_fa_calls) % 20 == 1:  # Log first call of each 20-layer pass
        # Show the 8 most relevant callers
        relevant = [c for c in callers if "vllm" in c or "dllm" in c][-8:]
        print(
            f"\n[FA PASS {len(_fa_calls) // 20}] nq={nq} causal={causal} dummy={is_dummy} profile={is_profile} warmup={is_warmup} generate={is_generate}",
            flush=True,
        )
        for c in relevant:
            print(f"  {c}", flush=True)

    return _orig_fa(*args, **kwargs)


fm.flash_attn_varlen_func = _traced_fa

from vllm import LLM, SamplingParams

print("=== Creating LLM ===", flush=True)
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
init_fa_calls = len(_fa_calls)
init_kv_updates = len(_kv_updates)
print(
    f"\n=== LLM created: {init_fa_calls} FA calls, {init_kv_updates} KV updates during init ===",
    flush=True,
)

# Summarize init passes
from collections import Counter

init_patterns = Counter()
for c in _fa_calls[:init_fa_calls]:
    k = f"nq={c['nq']} dummy={c['dummy']} warmup={c['warmup']}"
    init_patterns[k] += 1
print("Init passes:")
for p, n in init_patterns.most_common():
    print(f"  {p}  x{n}")

# Summarize KV updates during init
kv_slot_summary = Counter()
for u in _kv_updates[:init_kv_updates]:
    sm = u["slot_mapping"]
    all_neg1 = all(s == -1 for s in sm) if sm else True
    k = f"nk={u['nk']} all_neg1={all_neg1}"
    kv_slot_summary[k] += 1
print("\nKV updates during init:")
for p, n in kv_slot_summary.most_common():
    print(f"  {p}  x{n}")

print("\n=== Running generate ===", flush=True)
out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
gen_fa_calls = len(_fa_calls) - init_fa_calls
gen_kv_updates = len(_kv_updates) - init_kv_updates
print(
    f"\n=== Generate: {gen_fa_calls} FA calls, {gen_kv_updates} KV updates ===",
    flush=True,
)

# Summarize generate KV updates
gen_kv_summary = Counter()
for u in _kv_updates[init_kv_updates:]:
    sm = u["slot_mapping"]
    all_neg1 = all(s == -1 for s in sm) if sm else True
    k = f"nk={u['nk']} all_neg1={all_neg1}"
    gen_kv_summary[k] += 1
print("KV updates during generate:")
for p, n in gen_kv_summary.most_common():
    print(f"  {p}  x{n}")
