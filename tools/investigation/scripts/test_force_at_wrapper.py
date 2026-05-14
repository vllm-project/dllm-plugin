#!/usr/bin/env python3
"""Force causal=False at the flash_attn.py MODULE level wrapper and verify output."""

import os
import sys

import torch

sys.path.insert(0, "/workspace/dllm-plugin")
os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

# Patch flash_attn_varlen_func at the flash_attn.py module level
import vllm.v1.attention.backends.flash_attn as flash_mod

_orig = flash_mod.flash_attn_varlen_func


def _forced_false(**kwargs):
    kwargs["causal"] = False
    return _orig(**kwargs)


flash_mod.flash_attn_varlen_func = _forced_false

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
caps = {}


def hook(module, input, output):
    caps["out"] = output.detach().cpu().clone()


base.layers[0].self_attn.register_forward_hook(hook)

out = llm.generate(["The quick brown fox"], SamplingParams(temperature=0, max_tokens=1))

if "out" in caps:
    va = caps["out"]
    import torch.nn.functional as F

    # Compare with SDPA bidirectional and causal
    vq = torch.load(
        "investigation_captures/captures/attn_deep_layer00/vllm.attn_deep_layer00.A5_q_after_rope.pt",
        map_location="cpu",
        weights_only=True,
    )
    vk = torch.load(
        "investigation_captures/captures/attn_deep_layer00/vllm.attn_deep_layer00.A6_k_after_rope.pt",
        map_location="cpu",
        weights_only=True,
    )
    dv = torch.load(
        "investigation_captures/captures/attn_deep_layer00/dinfer.attn_deep_layer00.A6b_v_states.pt",
        map_location="cpu",
        weights_only=True,
    )

    print(f"Output shape: {va.shape}")
    print(f"Output mean: {va.float().mean():.6f}, std: {va.float().std():.6f}")

    # Compare FORCED output vs previous causal output
    da = torch.load(
        "investigation_captures/captures/attn_deep_layer00/dinfer.attn_deep_layer00.A9_oproj_output.pt",
        map_location="cpu",
        weights_only=True,
    ).squeeze(0)
    diff = torch.abs(va.float() - da.float())
    cos = F.cosine_similarity(
        va.float().flatten().unsqueeze(0), da.float().flatten().unsqueeze(0)
    ).item()
    print(f"FORCED non-causal vs dInfer: max_diff={diff.max():.6e}, cos_sim={cos:.6f}")
else:
    print("No capture!")
