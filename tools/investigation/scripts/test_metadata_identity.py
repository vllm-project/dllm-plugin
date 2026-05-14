#!/usr/bin/env python3
"""Check: is the metadata object identity preserved through the dispatch chain?"""

import os
import sys

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

# Patch get_attention_context to log the causal value it returns
import vllm.model_executor.layers.attention.attention as attn_mod

_orig_gac = attn_mod.get_attention_context
_logged = {}


def _patched_gac(layer_name):
    result = _orig_gac(layer_name)
    md = result[0]
    # Only log first call per layer
    key = (id(md), layer_name)
    if md is not None and key not in _logged and len(_logged) < 5:
        _logged[key] = True
        print(f"[GAC] {layer_name}: causal={md.causal} id={id(md):#x}", flush=True)
    return result


attn_mod.get_attention_context = _patched_gac

# CRITICAL: Also patch the REGISTERED custom op function
# The custom op captured a reference to get_attention_context at registration time.
# We need to patch the function that the custom op calls.

# Check: is unified_attention_with_output using a module-level reference or a closure?
uao = attn_mod.unified_attention_with_output
print(f"unified_attention_with_output: {uao}")
print(f"  __module__: {getattr(uao, '__module__', 'N/A')}")

# The custom op dispatch doesn't go through Python at all for torch.ops calls.
# But the use_direct_call path DOES use Python.
# Let me check which path is used.

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

model = llm.llm_engine.model_executor.driver_worker.model_runner.model
base = model.model if hasattr(model, "model") else model
attn = base.layers[0].self_attn.attn

# Check if use_direct_call is True
print(f"use_direct_call: {attn.use_direct_call}")

llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
print(f"GAC calls logged: {len(_logged)}")
