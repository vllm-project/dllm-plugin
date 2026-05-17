#!/usr/bin/env python3
"""Capture every sub-operation during dllm-plugin's FIRST denoising iteration.

Uses a forward-pass counter on the model to distinguish:
  FWD 0 = prefill
  FWD 1 = first denoise (bootstrap / all-mask)
  FWD 2 = first actual denoise with committed tokens

Captures FWD 1 by default — the first time the model sees the full draft
block after prefill.

Usage:
    python3 capture_e2e_vllm.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_USE_MOCK_MODEL", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import torch

torch.compiler.disable(recursive=True)

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import CaptureRegistry, extract_model_config

MODEL_PATH = "/workspace/llada2-mini"
PROMPT = "The quick brown fox"
CAPTURE_DIR = "/workspace/captures/e2e"
TARGET_FWD = 1  # 0=prefill, 1=first denoise


def find_module(model, *candidates):
    for path in candidates:
        parts = path.split(".")
        obj = model
        try:
            for part in parts:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            return obj
        except (AttributeError, IndexError, TypeError):
            continue
    return None


class FwdCounter:
    def __init__(self, target_fwd: int):
        self.count = 0
        self.target = target_fwd
        self.active = False

    def on_fwd_start(self):
        if self.count == self.target:
            self.active = True
        else:
            self.active = False
        self.count += 1


def register_gated_hooks(model, registry, counter):
    """Register hooks that only fire during the target forward pass."""
    hooks = []
    base = None
    for attr in ["model", "transformer"]:
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if hasattr(candidate, "layers"):
                base = candidate
                break
    if base is None:
        base = model

    def make_hook(name):
        def hook(module, input, output):
            if not counter.active:
                return
            try:
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor):
                    registry.register(name, tensor)
            except Exception as e:
                print(f"[HOOK ERR] {name}: {e}")

        return hook

    def make_pre_hook(name, idx=0):
        def hook(module, input):
            if not counter.active:
                return
            try:
                tensor = input[idx] if isinstance(input, tuple) else input
                if isinstance(tensor, torch.Tensor):
                    registry.register(name, tensor)
            except Exception as e:
                print(f"[PRE-HOOK ERR] {name}: {e}")

        return hook

    # Embedding
    embed = find_module(base, "embed_tokens", "word_embeddings")
    if embed:
        hooks.append(embed.register_forward_hook(make_hook("embedding")))

    # Per-layer
    for i, layer in enumerate(base.layers):
        p = f"layer{i:02d}"

        hooks.append(layer.register_forward_pre_hook(make_pre_hook(f"{p}.input")))

        norm1 = find_module(layer, "input_layernorm")
        if norm1:
            hooks.append(norm1.register_forward_pre_hook(make_pre_hook(f"{p}.norm1_input")))
            hooks.append(norm1.register_forward_hook(make_hook(f"{p}.norm1_output")))

        attn = find_module(layer, "self_attn")
        if attn:
            hooks.append(attn.register_forward_pre_hook(make_pre_hook(f"{p}.attn_input")))
            hooks.append(attn.register_forward_hook(make_hook(f"{p}.attn_output")))

            qkv = find_module(attn, "qkv_proj")
            if qkv:
                hooks.append(qkv.register_forward_hook(make_hook(f"{p}.qkv_proj")))
            q_norm = find_module(attn, "q_norm")
            k_norm = find_module(attn, "k_norm")
            if q_norm:
                hooks.append(q_norm.register_forward_hook(make_hook(f"{p}.q_norm")))
            if k_norm:
                hooks.append(k_norm.register_forward_hook(make_hook(f"{p}.k_norm")))
            o_proj = find_module(attn, "o_proj")
            if o_proj:
                hooks.append(o_proj.register_forward_hook(make_hook(f"{p}.o_proj")))

        norm2 = find_module(layer, "post_attention_layernorm")
        if norm2:
            hooks.append(norm2.register_forward_pre_hook(make_pre_hook(f"{p}.residual1")))
            hooks.append(norm2.register_forward_hook(make_hook(f"{p}.norm2_output")))

        mlp = find_module(layer, "mlp")
        if mlp:
            hooks.append(mlp.register_forward_pre_hook(make_pre_hook(f"{p}.moe_input")))
            hooks.append(mlp.register_forward_hook(make_hook(f"{p}.moe_output")))

            gate = find_module(mlp, "gate")
            if gate:
                hooks.append(gate.register_forward_hook(make_hook(f"{p}.gate_logits")))

        hooks.append(layer.register_forward_hook(make_hook(f"{p}.output")))

    final_norm = find_module(base, "norm", "final_layernorm")
    if final_norm:
        hooks.append(final_norm.register_forward_pre_hook(make_pre_hook("final_norm_input")))
        hooks.append(final_norm.register_forward_hook(make_hook("final_norm_output")))

    lm_head = find_module(model, "lm_head")
    if lm_head:
        hooks.append(lm_head.register_forward_hook(make_hook("lm_head_output")))

    print(f"  Registered {len(hooks)} gated hooks (target FWD={counter.target})")
    return hooks


def main():
    from dllm_plugin import register_dllm

    register_dllm()

    from vllm import LLM, SamplingParams

    print("Loading vLLM engine with dllm-plugin...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="dllm_plugin.Worker",
        async_scheduling=False,
        dtype="bfloat16",
    )

    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    vllm_model = runner.model
    print(f"  Model type: {type(vllm_model).__name__}")

    # Set up capture
    registry = CaptureRegistry(framework="vllm", scenario="e2e_first_denoise")

    hf_config = None
    for attr in ["config", "model_config"]:
        if hasattr(vllm_model, attr):
            hf_config = getattr(vllm_model, attr)
            break
    base = None
    for attr_name in ["model", "transformer"]:
        if hasattr(vllm_model, attr_name):
            candidate = getattr(vllm_model, attr_name)
            if hasattr(candidate, "layers"):
                base = candidate
                break
    if hf_config is None and base and hasattr(base, "config"):
        hf_config = base.config
    if hf_config:
        registry.model_config = extract_model_config(hf_config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    registry.input_ids = input_ids[0].tolist()
    registry.positions = list(range(input_ids.shape[1]))
    print(f"  Prompt: '{PROMPT}' -> {registry.input_ids} ({len(registry.input_ids)} tokens)")

    # Register gated hooks on the model
    counter = FwdCounter(target_fwd=TARGET_FWD)
    hooks = register_gated_hooks(vllm_model, registry, counter)

    # Intercept the model's forward to count and capture input state.
    # vLLM calls model(**model_inputs) so everything arrives as kwargs.
    original_forward = vllm_model.forward

    def instrumented_forward(**kwargs):
        counter.on_fwd_start()
        fwd_num = counter.count - 1
        ids = kwargs.get("input_ids")
        pos = kwargs.get("positions")
        seq_len = ids.shape[0] if isinstance(ids, torch.Tensor) else "?"
        if counter.active:
            if isinstance(ids, torch.Tensor):
                registry.register("model_input_ids", ids)
            if isinstance(pos, torch.Tensor):
                registry.register("model_positions", pos)
            print(f"  [FWD {fwd_num}] CAPTURING (seq_len={seq_len})")
        else:
            print(f"  [FWD {fwd_num}] skipping (seq_len={seq_len})")
        return original_forward(**kwargs)

    vllm_model.forward = instrumented_forward

    # Run generation
    print("\nRunning llm.generate()...")
    outputs = llm.generate(
        [PROMPT],
        SamplingParams(max_tokens=32, temperature=0),
    )
    gen_text = outputs[0].outputs[0].text
    gen_ids = list(outputs[0].outputs[0].token_ids)
    registry.extra["generated_text"] = gen_text
    registry.extra["generated_ids"] = gen_ids
    print(f"  Generated: {gen_text[:100]!r} ({len(gen_ids)} tokens)")

    # Clean up
    vllm_model.forward = original_forward
    for h in hooks:
        h.remove()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Captures: {len(registry.captures)}")
    for name in sorted(registry.captures.keys()):
        t = registry.captures[name]
        print(f"  {name}: shape={list(t.shape)} dtype={t.dtype}")
    print(f"{'=' * 60}")

    out_dir = registry.save(CAPTURE_DIR)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
