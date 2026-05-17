#!/usr/bin/env python3
"""Capture every sub-operation during dInfer's FIRST denoising iteration.

Uses a forward-pass counter to distinguish prefill (FWD 0) from the first
denoise step (FWD 1). Only captures on FWD 1 so we get the exact state
the model sees when processing the first block of [prompt + mask] tokens.

Usage:
    python3 capture_e2e_dinfer.py
"""

import json
import os
import sys
from pathlib import Path

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch

torch.compiler.disable(recursive=True)

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import CaptureRegistry, extract_model_config

MODEL_PATH = "/workspace/llada2-mini"
PROMPT = "The quick brown fox"
CAPTURE_DIR = "/workspace/captures/e2e"
TARGET_FWD = 1  # 0=prefill, 1=first denoise


def init_distributed():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "45611")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from vllm import distributed
    from vllm.config import ParallelConfig

    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    return ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=1,
    )


def find_module(model, *candidates):
    for path in candidates:
        parts = path.split(".")
        obj = model
        try:
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


class FwdCounter:
    """Tracks forward passes and gates capture to a specific one."""

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
    base = model.model if hasattr(model, "model") else model

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
    embed = find_module(base, "word_embeddings", "embed_tokens")
    if embed:
        hooks.append(embed.register_forward_hook(make_hook("embedding")))

    # Per-layer hooks
    for i, layer in enumerate(base.layers):
        p = f"layer{i:02d}"

        # Layer input (pre-hook on layer itself captures hidden_states)
        hooks.append(layer.register_forward_pre_hook(make_pre_hook(f"{p}.input")))

        # input_layernorm
        norm1 = find_module(layer, "input_layernorm")
        if norm1:
            hooks.append(norm1.register_forward_pre_hook(make_pre_hook(f"{p}.norm1_input")))
            hooks.append(norm1.register_forward_hook(make_hook(f"{p}.norm1_output")))

        # attention
        attn = find_module(layer, "attention", "self_attn")
        if attn:
            hooks.append(attn.register_forward_pre_hook(make_pre_hook(f"{p}.attn_input")))
            hooks.append(attn.register_forward_hook(make_hook(f"{p}.attn_output")))

            # QKV projection
            qkv = find_module(attn, "query_key_value", "qkv_proj")
            if qkv:
                hooks.append(qkv.register_forward_hook(make_hook(f"{p}.qkv_proj")))

            # Q/K norms
            q_norm = find_module(attn, "query_layernorm", "q_norm")
            k_norm = find_module(attn, "key_layernorm", "k_norm")
            if q_norm:
                hooks.append(q_norm.register_forward_hook(make_hook(f"{p}.q_norm")))
            if k_norm:
                hooks.append(k_norm.register_forward_hook(make_hook(f"{p}.k_norm")))

            # O projection
            o_proj = find_module(attn, "dense", "o_proj")
            if o_proj:
                hooks.append(o_proj.register_forward_hook(make_hook(f"{p}.o_proj")))

        # Post-attention residual (input to post_attention_layernorm)
        norm2 = find_module(layer, "post_attention_layernorm")
        if norm2:
            hooks.append(norm2.register_forward_pre_hook(make_pre_hook(f"{p}.residual1")))
            hooks.append(norm2.register_forward_hook(make_hook(f"{p}.norm2_output")))

        # MLP/MoE
        mlp = find_module(layer, "mlp")
        if mlp:
            hooks.append(mlp.register_forward_pre_hook(make_pre_hook(f"{p}.moe_input")))
            hooks.append(mlp.register_forward_hook(make_hook(f"{p}.moe_output")))

            # Gate
            gate = find_module(mlp, "gate", "router")
            if gate:
                hooks.append(gate.register_forward_hook(make_hook(f"{p}.gate_logits")))

        # Layer output
        hooks.append(layer.register_forward_hook(make_hook(f"{p}.output")))

    # Final norm
    final_norm = find_module(base, "norm", "final_layernorm")
    if final_norm:
        hooks.append(final_norm.register_forward_pre_hook(make_pre_hook("final_norm_input")))
        hooks.append(final_norm.register_forward_hook(make_hook("final_norm_output")))

    # LM head
    lm_head = find_module(model, "lm_head", "output")
    if lm_head:
        hooks.append(lm_head.register_forward_hook(make_hook("lm_head_output")))

    print(f"  Registered {len(hooks)} gated hooks (target FWD={counter.target})")
    return hooks


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("Initializing distributed...")
    parallel_config = init_distributed()

    print("Loading model...")
    from dinfer import (
        BlockDiffusionLLM,
        BlockIteratorFactory,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeModelLM
    from transformers import AutoConfig, AutoTokenizer
    from vllm.config import VllmConfig, set_current_vllm_config

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = LLaDA2MoeModelLM(config=model_config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

    mask_id = 156895
    eos_id = tokenizer.eos_token_id

    decoder = ThresholdParallelDecoder(
        temperature=0, threshold=0.9, mask_id=mask_id, eos_id=eos_id,
    )

    prompt_len = tokenizer(PROMPT, return_tensors="pt")["input_ids"].shape[1]
    gen_length = 32
    total_len = ((prompt_len + gen_length + 31) // 32) * 32
    max_len = max(total_len, 64)

    dllm = BlockDiffusionLLM(
        model, decoder,
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True, max_length=max_len),
        early_stop=True,
    )
    print(f"  BlockDiffusionLLM created (max_length={max_len})")

    # Set up capture
    registry = CaptureRegistry(framework="dinfer", scenario="e2e_first_denoise")
    if hasattr(model, "config"):
        registry.model_config = extract_model_config(model.config)
    elif hasattr(model.model, "config"):
        registry.model_config = extract_model_config(model.model.config)

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)
    registry.input_ids = input_ids[0].tolist()
    registry.positions = list(range(input_ids.shape[1]))
    print(f"  Prompt: '{PROMPT}' -> {registry.input_ids} ({len(registry.input_ids)} tokens)")

    # Gated hooks: only capture on the target forward pass
    counter = FwdCounter(target_fwd=TARGET_FWD)
    hooks = register_gated_hooks(model, registry, counter)

    # Intercept the model's forward to track forward pass count AND
    # capture input_ids/positions
    base_model = model.model if hasattr(model, "model") else model
    original_forward = model.forward

    def instrumented_forward(*args, **kwargs):
        counter.on_fwd_start()
        fwd_num = counter.count - 1
        if counter.active:
            # Capture the actual input_ids and positions
            if args:
                ids = args[0]
                if isinstance(ids, torch.Tensor):
                    registry.register("model_input_ids", ids)
            if "input_ids" in kwargs and isinstance(kwargs["input_ids"], torch.Tensor):
                registry.register("model_input_ids", kwargs["input_ids"])
            for kw in ["positions", "position_ids"]:
                if kw in kwargs and isinstance(kwargs[kw], torch.Tensor):
                    registry.register("model_positions", kwargs[kw])

            print(f"  [FWD {fwd_num}] CAPTURING (active=True)")
        else:
            ids = args[0] if args else kwargs.get("input_ids")
            seq_len = ids.shape[-1] if isinstance(ids, torch.Tensor) else "?"
            print(f"  [FWD {fwd_num}] skipping (seq_len={seq_len})")
        return original_forward(*args, **kwargs)

    model.forward = instrumented_forward

    # Run generation
    print("\nRunning BlockDiffusionLLM.generate()...")
    from vllm.config import get_current_vllm_config, set_current_vllm_config
    from vllm.forward_context import set_forward_context

    try:
        vllm_config = get_current_vllm_config()
    except (AssertionError, RuntimeError):
        vllm_config = VllmConfig(parallel_config=parallel_config)

    with set_current_vllm_config(vllm_config):
        with set_forward_context(None, vllm_config):
            outputs = dllm.generate(input_ids, gen_length=32, block_length=32)

    if outputs is not None:
        gen_ids = outputs[0, input_ids.shape[1]:].cpu().tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        registry.extra["generated_text"] = gen_text
        registry.extra["generated_ids"] = gen_ids
        print(f"  Generated: {gen_text[:100]!r}")

    # Clean up
    model.forward = original_forward
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
