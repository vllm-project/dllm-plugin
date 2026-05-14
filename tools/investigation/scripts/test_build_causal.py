#!/usr/bin/env python3
"""Check causal value at build() input and output."""

import os
import sys

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

import vllm.v1.attention.backends.flash_attn as fa

orig = fa.FlashAttentionMetadataBuilder.build
bc = []


def logged_build(self, cpl, cam, fb=False):
    bc.append({"input_causal": cam.causal, "tokens": cam.num_actual_tokens})
    result = orig(self, cpl, cam, fb)
    bc[-1]["output_causal"] = result.causal
    return result


fa.FlashAttentionMetadataBuilder.build = logged_build

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

print(f"\nBuild calls: {len(bc)}")
for i, c in enumerate(bc):
    print(
        f"  build #{i}: input_causal={c['input_causal']} output_causal={c['output_causal']} tokens={c['tokens']}"
    )
