#!/usr/bin/env python3
"""Simple test: does the fork produce bidirectional attention output?"""

import os
import sys

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

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
    enable_prefix_caching=False,
)

model = llm.llm_engine.model_executor.driver_worker.model_runner.model
base = model.model if hasattr(model, "model") else model

# Capture layer 0 attention output via hook
caps = {}


def hook(module, input, output):
    caps["out"] = output.detach().cpu().clone()


base.layers[0].self_attn.register_forward_hook(hook)

out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))

if "out" in caps:
    va = caps["out"]
    print(f"Layer 0 output shape: {va.shape}")
    print("Per-position stats:")
    for i in range(min(4, va.shape[0])):
        print(
            f"  pos {i}: mean={va[i].float().mean():.6f}, std={va[i].float().std():.6f}"
        )

    # In causal attention, pos 0 sees only 1 token -> lower variance
    # In bidirectional, pos 0 sees all tokens -> higher variance
    # pos 3 should be the same either way (sees all tokens in both modes)
    if va.shape[0] >= 4:
        ratio = va[0].float().std() / va[3].float().std()
        print(f"\n  pos0_std / pos3_std = {ratio:.4f}")
        if ratio < 0.9:
            print("  -> Likely CAUSAL (pos 0 sees fewer tokens)")
        else:
            print("  -> Likely BIDIRECTIONAL (all positions see all tokens)")
else:
    print("No capture!")
