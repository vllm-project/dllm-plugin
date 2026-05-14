#!/usr/bin/env python3
"""Capture tensors from dllm-plugin via vLLM PREFILL path (no block diffusion).

Loads the model via vLLM LLM engine WITHOUT the dllm scheduler/worker, so
the engine processes raw prompt tokens at sequential positions [0,1,2,...].
This matches how the dInfer reference captures work (AutoModelForCausalLM
with raw tokens and bidirectional attention).

Based on the established pattern from tools/validation/capture_vllm_prefill.py.

Usage:
    PYTHONPATH=/workspace/dllm-plugin python3 capture_vllm_prefill.py --scenario first_block
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_USE_MOCK_MODEL", "0")
os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import (
    CaptureRegistry,
    extract_model_config,
    load_config,
    make_capture_hook,
    make_capture_pre_hook,
)


def load_vllm_prefill(model_path: str):
    """Load model via LLM engine WITHOUT dllm scheduler (standard prefill path).

    This is the key insight from previous validation scripts: by NOT passing
    scheduler_cls/worker_cls, vLLM processes raw tokens through its standard
    prefill path, matching how dInfer's AutoModelForCausalLM processes tokens.
    """
    from dllm_plugin import register_dllm

    register_dllm()

    from vllm import LLM

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        # NO scheduler_cls, NO worker_cls — standard vLLM prefill
    )

    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    model = runner.model
    hf_config = llm.llm_engine.model_config.hf_config
    return llm, model, hf_config


def find_module(model, *candidates):
    for path in candidates:
        parts = path.split(".")
        obj = model
        try:
            for part in parts:
                obj = getattr(obj, part) if not part.isdigit() else obj[int(part)]
            return obj
        except (AttributeError, IndexError, TypeError):
            continue
    return None


def get_model_base(model):
    for attr in ["model", "transformer"]:
        if hasattr(model, attr):
            base = getattr(model, attr)
            if hasattr(base, "layers"):
                return base
    return model


def register_all_hooks(model, registry):
    """Register capture hooks. Returns hook handles."""
    hooks = []
    base = get_model_base(model)

    # L4: Layer hidden states
    embed = find_module(base, "embed_tokens")
    if embed:
        hooks.append(
            embed.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.embedding")
            )
        )

    for i, layer in enumerate(base.layers):
        hooks.append(
            layer.register_forward_hook(
                make_capture_hook(registry, f"L4_layer.layer{i:02d}.hidden_out")
            )
        )

    final_norm = find_module(base, "norm")
    if final_norm:
        hooks.append(
            final_norm.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.final_norm")
            )
        )

    lm_head = find_module(model, "lm_head")
    if lm_head:
        hooks.append(
            lm_head.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.lm_head")
            )
        )

    # L5: Sub-layer
    for i, layer in enumerate(base.layers):
        prefix = f"L5_sublayer.layer{i:02d}"
        for attr, name in [
            ("input_layernorm", "input_norm_out"),
            ("self_attn", "attn_out"),
            ("post_attention_layernorm", "post_attn_norm_out"),
        ]:
            mod = find_module(layer, attr)
            if mod:
                hooks.append(
                    mod.register_forward_hook(
                        make_capture_hook(registry, f"{prefix}.{name}")
                    )
                )
                if attr == "post_attention_layernorm":
                    hooks.append(
                        mod.register_forward_pre_hook(
                            make_capture_pre_hook(registry, f"{prefix}.residual1")
                        )
                    )
        mlp = find_module(layer, "mlp")
        if mlp:
            hooks.append(
                mlp.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.moe_out")
                )
            )
        hooks.append(
            layer.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.residual2")
            )
        )

    # L6: Sub-attention
    for i, layer in enumerate(base.layers):
        prefix = f"L6_subattn.layer{i:02d}"
        attn = find_module(layer, "self_attn")
        if not attn:
            continue
        for attr, name in [
            ("qkv_proj", "qkv_proj_out"),
            ("q_norm", "q_norm"),
            ("k_norm", "k_norm"),
            ("o_proj", "o_proj_out"),
        ]:
            mod = find_module(attn, attr)
            if mod:
                hooks.append(
                    mod.register_forward_hook(
                        make_capture_hook(registry, f"{prefix}.{name}")
                    )
                )

    # L7: Atomic (RMSNorm + MoE gate + shared expert)
    for i, layer in enumerate(base.layers):
        prefix = f"L7_atomic.layer{i:02d}"
        norm1 = find_module(layer, "input_layernorm")
        if norm1 and hasattr(norm1, "weight"):
            original_forward = norm1.forward
            eps = getattr(norm1, "variance_epsilon", getattr(norm1, "eps", 1e-6))
            w = norm1.weight

            def make_patched_norm(orig_fwd, eps_val, weight, pfx, reg):
                def patched(x, *args, **kwargs):
                    orig_dtype = x.dtype
                    x_fp32 = x.to(torch.float32)
                    if args and args[0] is not None:
                        x_fp32 = x_fp32 + args[0]
                    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
                    reg.register(f"{pfx}.rmsnorm_variance", variance)
                    rsqrt_val = torch.rsqrt(variance + eps_val)
                    reg.register(f"{pfx}.rmsnorm_rsqrt", rsqrt_val)
                    normalized = x_fp32 * rsqrt_val
                    reg.register(f"{pfx}.rmsnorm_normalized", normalized.to(orig_dtype))
                    scaled = normalized.to(orig_dtype) * weight
                    reg.register(f"{pfx}.rmsnorm_scaled", scaled)
                    return orig_fwd(x, *args, **kwargs)

                return patched

            norm1.forward = make_patched_norm(
                original_forward, eps, w, prefix, registry
            )
            norm1._original_forward = original_forward

        mlp = find_module(layer, "mlp")
        if mlp:
            gate = find_module(mlp, "gate")
            if gate:

                def make_gate_hook(pfx, reg):
                    def hook(module, input, output):
                        logits = output[0] if isinstance(output, tuple) else output
                        reg.register(f"{pfx}.gate_logits", logits)
                        reg.register(
                            f"{pfx}.gate_sigmoid", torch.sigmoid(logits.float())
                        )

                    return hook

                hooks.append(
                    gate.register_forward_hook(make_gate_hook(prefix, registry))
                )

            for attr_name, cap_name in [
                ("shared_expert_gate", "shared_gate"),
                ("shared_expert_up", "shared_up"),
                ("shared_expert_down", "shared_output"),
            ]:
                mod = find_module(mlp, attr_name)
                if mod:
                    hooks.append(
                        mod.register_forward_hook(
                            make_capture_hook(registry, f"{prefix}.{cap_name}")
                        )
                    )

    print(f"  Registered {len(hooks)} hooks")
    return hooks


def cleanup_hooks(model, hooks):
    for h in hooks:
        h.remove()
    base = get_model_base(model)
    for layer in base.layers:
        norm1 = find_module(layer, "input_layernorm")
        if norm1 and hasattr(norm1, "_original_forward"):
            norm1.forward = norm1._original_forward
            del norm1._original_forward


def run_scenario(llm, model, config, scenario_name):
    """Capture via llm.generate() in standard prefill mode."""
    scenario = config["scenarios"][scenario_name]
    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario_name} — {scenario['description']}")
    print(f"{'=' * 70}")

    registry = CaptureRegistry(framework="vllm", scenario=scenario_name)
    hf_config = llm.llm_engine.model_config.hf_config
    registry.model_config = extract_model_config(hf_config)

    # Load input_ids from dInfer capture for alignment
    dinfer_meta_path = (
        Path(config["capture_root"])
        / scenario_name
        / f"dinfer.{scenario_name}.metadata.json"
    )
    if dinfer_meta_path.exists():
        with open(dinfer_meta_path) as f:
            dinfer_meta = json.load(f)
        token_ids = dinfer_meta["input_ids"]
        print(f"  Loaded input_ids from dInfer: {token_ids}")
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], trust_remote_code=True
        )
        token_ids = tokenizer(config["test_prompt"])["input_ids"]
        print(f"  Tokenized: {token_ids}")

    registry.input_ids = token_ids
    registry.positions = list(range(len(token_ids)))

    # Register hooks
    hooks = register_all_hooks(model, registry)

    # Trigger forward pass via generate (standard prefill, no block diffusion)
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    # Use the same prompt text
    prompt_text = config["test_prompt"]
    print(f"  Running prefill for: '{prompt_text}'")
    outputs = llm.generate([prompt_text], sampling_params)

    if outputs:
        output_ids = outputs[0].outputs[0].token_ids
        registry.register_non_tensor("L1_output.token_ids", list(output_ids))
        print(f"  Generated token: {output_ids}")

    cleanup_hooks(model, hooks)

    for level in ["L2", "L3", "L4", "L5", "L6", "L7"]:
        count = sum(1 for k in registry.captures if k.startswith(f"{level}_"))
        print(f"  {level}: {count} checkpoints")

    out_dir = registry.save(config["capture_root"])
    print(
        f"\n[DONE] {scenario_name}: {len(registry.captures)} checkpoints -> {out_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="vLLM prefill capture")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    print("Loading vLLM (prefill mode, no block diffusion)...")
    llm, model, hf_config = load_vllm_prefill(config["model_path"])

    print("\n[STRUCTURE] Model module tree (depth 3):")
    for name, mod in model.named_modules():
        depth = name.count(".")
        if depth <= 3:
            print(f"  {'  ' * depth}{name}: {type(mod).__name__}")

    scenarios = (
        list(config["scenarios"].keys())
        if args.all_scenarios
        else [args.scenario or "first_block"]
    )

    for scenario_name in scenarios:
        run_scenario(llm, model, config, scenario_name)

    print(f"\n{'=' * 70}")
    print("ALL CAPTURES COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
