#!/usr/bin/env python3
"""Verify the fork's non-causal config is correctly set."""

import os
import sys

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

from dllm_plugin import register_dllm

register_dllm()
from vllm import LLM

llm = LLM(
    model="/workspace/llada2-mini",
    trust_remote_code=True,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_model_len=256,
    max_num_seqs=1,
    gpu_memory_utilization=0.9,
    dtype="bfloat16",
)

vc = llm.llm_engine.vllm_config
ac = vc.attention_config
mc = vc.model_config
sc = vc.scheduler_config
cc = vc.cache_config

print(f"attention_config.use_non_causal: {getattr(ac, 'use_non_causal', 'N/A')}")
print(f"model_config._use_non_causal: {getattr(mc, '_use_non_causal', 'N/A')}")
print(f"enable_chunked_prefill: {sc.enable_chunked_prefill}")
print(f"enable_prefix_caching: {cc.enable_prefix_caching}")

# Also check the CommonAttentionMetadata causal value
from vllm.forward_context import get_forward_context

model = llm.llm_engine.model_executor.driver_worker.model_runner.model
orig = type(model).forward
checked = [False]


def dbg(self, input_ids=None, positions=None, **kwargs):
    if not checked[0]:
        checked[0] = True
        ctx = get_forward_context()
        if ctx and ctx.attn_metadata:
            md = ctx.attn_metadata
            if isinstance(md, dict):
                k = next(iter(md))
                m = md[k]
                print(f"CommonAttentionMetadata.causal: {m.causal}")
    return orig(self, input_ids=input_ids, positions=positions, **kwargs)


type(model).forward = dbg

from vllm import SamplingParams

llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
print("DONE")
