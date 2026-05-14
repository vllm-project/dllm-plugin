#!/usr/bin/env python3
"""Check scheduler_metadata field in attention metadata."""

import os
import sys

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

from dllm_plugin import register_dllm

register_dllm()
from vllm import LLM, SamplingParams
from vllm.forward_context import get_forward_context

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
orig = type(model).forward
checked = [False]


def debug_forward(self, input_ids=None, positions=None, **kwargs):
    if not checked[0]:
        checked[0] = True
        ctx = get_forward_context()
        if ctx is not None and ctx.attn_metadata is not None:
            md = ctx.attn_metadata
            if isinstance(md, dict):
                key = next(iter(md))
                m = md[key]
                if m is not None:
                    print(f"causal: {m.causal}", flush=True)
                    sm = m.scheduler_metadata
                    print(f"scheduler_metadata: {sm}", flush=True)
                    print(f"scheduler_metadata is None: {sm is None}", flush=True)
                    if sm is not None:
                        print(f"scheduler_metadata shape: {sm.shape}", flush=True)
                        print(f"scheduler_metadata dtype: {sm.dtype}", flush=True)
    return orig(self, input_ids=input_ids, positions=positions, **kwargs)


type(model).forward = debug_forward
llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))
