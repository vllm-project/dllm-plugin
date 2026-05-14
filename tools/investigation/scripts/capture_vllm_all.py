#!/usr/bin/env python3
"""Exhaustive tensor capture from dllm-plugin (vLLM 0.20.1) LLaDA2.0-mini.

Captures ALL layers (0-19) at ALL levels (L1-L7) for all scenarios.
Run on the investigation-vllm pod with vLLM 0.20.1 + dllm-plugin installed.

Usage:
    python3 capture_vllm_all.py [--scenario first_block]
    python3 capture_vllm_all.py --all-scenarios
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_DLLM_USE_MOCK_MODEL", "0")

import torch

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import (
    CaptureRegistry,
    extract_model_config,
    load_config,
    make_capture_hook,
    make_capture_pre_hook,
)


def load_vllm_model(model_path: str):
    """Load LLaDA2ForCausalLM via vLLM LLM engine."""
    from dllm_plugin import register_dllm

    register_dllm()

    from vllm import LLM

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        model_impl="dllm_plugin.models.llada2:LLaDA2ForCausalLM",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="dllm_plugin.Worker",
        dtype="bfloat16",
    )

    # Access internal model for hook registration
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    vllm_model = runner.model
    return llm, vllm_model, runner


def find_module(model, *candidates):
    """Find a module by trying multiple attribute paths."""
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


def discover_model_structure(model):
    """Print model module hierarchy for hook registration."""
    print("\n[STRUCTURE] Model module tree (depth 3):")
    for name, mod in model.named_modules():
        depth = name.count(".")
        if depth <= 3:
            print(f"  {'  ' * depth}{name}: {type(mod).__name__}")


def get_model_base(model):
    """Get the base transformer model (handles vLLM wrapping)."""
    for attr in ["model", "transformer", "gpt_neox"]:
        if hasattr(model, attr):
            base = getattr(model, attr)
            if hasattr(base, "layers"):
                return base
    return model


def capture_L4(model, registry, input_ids, positions, device):
    """Capture Level 4: all layer hidden states."""
    hooks = []
    base = get_model_base(model)

    # Embedding
    embed = find_module(base, "embed_tokens", "word_embeddings")
    if embed:
        hooks.append(
            embed.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.embedding")
            )
        )

    # Each decoder layer
    for i, layer in enumerate(base.layers):
        hooks.append(
            layer.register_forward_hook(
                make_capture_hook(registry, f"L4_layer.layer{i:02d}.hidden_out")
            )
        )

    # Final norm
    final_norm = find_module(base, "norm", "final_layernorm")
    if final_norm:
        hooks.append(
            final_norm.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.final_norm")
            )
        )

    # LM head
    lm_head = find_module(model, "lm_head")
    if lm_head:
        hooks.append(
            lm_head.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.lm_head")
            )
        )

    # Forward pass using vLLM's internal forward
    with torch.no_grad():
        ids = (
            input_ids.to(device).squeeze(0)
            if input_ids.dim() > 1
            else input_ids.to(device)
        )
        pos = (
            positions.to(device).squeeze(0)
            if positions.dim() > 1
            else positions.to(device)
        )
        output = model(ids, positions=pos)

    # Capture logits
    if isinstance(output, tuple):
        logits = output[0]
    elif hasattr(output, "logits"):
        logits = output.logits
    else:
        logits = output
    if logits is not None:
        registry.register("L3_logits.global.raw_logits", logits)
        probs = torch.softmax(logits.float(), dim=-1)
        registry.register("L2_probs.global.softmax_probs", probs)

    for h in hooks:
        h.remove()
    return logits


def capture_L5_layer(model, registry, layer_idx, input_ids, positions, device):
    """Capture Level 5: sub-layer operations for a specific layer."""
    hooks = []
    base = get_model_base(model)
    layer = base.layers[layer_idx]
    prefix = f"L5_sublayer.layer{layer_idx:02d}"

    # input_layernorm
    norm1 = find_module(layer, "input_layernorm")
    if norm1:
        hooks.append(
            norm1.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.input_norm_out")
            )
        )

    # attention
    attn = find_module(layer, "self_attn")
    if attn:
        hooks.append(
            attn.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.attn_out")
            )
        )

    # post_attention_layernorm
    norm2 = find_module(layer, "post_attention_layernorm")
    if norm2:
        hooks.append(
            norm2.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.post_attn_norm_out")
            )
        )
        hooks.append(
            norm2.register_forward_pre_hook(
                make_capture_pre_hook(registry, f"{prefix}.residual1")
            )
        )

    # MoE
    mlp = find_module(layer, "mlp")
    if mlp:
        hooks.append(
            mlp.register_forward_hook(make_capture_hook(registry, f"{prefix}.moe_out"))
        )

    # Layer output = residual2
    hooks.append(
        layer.register_forward_hook(make_capture_hook(registry, f"{prefix}.residual2"))
    )

    with torch.no_grad():
        ids = (
            input_ids.to(device).squeeze(0)
            if input_ids.dim() > 1
            else input_ids.to(device)
        )
        pos = (
            positions.to(device).squeeze(0)
            if positions.dim() > 1
            else positions.to(device)
        )
        model(ids, positions=pos)

    for h in hooks:
        h.remove()


def capture_L6_layer(model, registry, layer_idx, input_ids, positions, device):
    """Capture Level 6: sub-attention operations for a specific layer."""
    hooks = []
    base = get_model_base(model)
    layer = base.layers[layer_idx]
    prefix = f"L6_subattn.layer{layer_idx:02d}"

    attn = find_module(layer, "self_attn")
    if attn is None:
        print(f"[WARN] Could not find self_attn for layer {layer_idx}")
        return

    # QKV projection
    qkv = find_module(attn, "qkv_proj")
    if qkv:
        hooks.append(
            qkv.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.qkv_proj_out")
            )
        )

    # Q/K normalization
    q_norm = find_module(attn, "q_norm")
    k_norm = find_module(attn, "k_norm")
    if q_norm:
        hooks.append(
            q_norm.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.q_norm")
            )
        )
    if k_norm:
        hooks.append(
            k_norm.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.k_norm")
            )
        )

    # Output projection
    o_proj = find_module(attn, "o_proj")
    if o_proj:
        hooks.append(
            o_proj.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.o_proj_out")
            )
        )

    with torch.no_grad():
        ids = (
            input_ids.to(device).squeeze(0)
            if input_ids.dim() > 1
            else input_ids.to(device)
        )
        pos = (
            positions.to(device).squeeze(0)
            if positions.dim() > 1
            else positions.to(device)
        )
        model(ids, positions=pos)

    for h in hooks:
        h.remove()


def capture_L7_layer(model, registry, layer_idx, input_ids, positions, device):
    """Capture Level 7: atomic operations for a specific layer."""
    base = get_model_base(model)
    layer = base.layers[layer_idx]
    prefix = f"L7_atomic.layer{layer_idx:02d}"
    hooks = []

    # RMSNorm internals
    norm1 = find_module(layer, "input_layernorm")
    if norm1 and hasattr(norm1, "weight"):
        original_forward = norm1.forward
        eps = getattr(norm1, "variance_epsilon", getattr(norm1, "eps", 1e-6))
        weight = norm1.weight

        def patched_norm_forward(x, *args, **kwargs):
            orig_dtype = x.dtype
            x_fp32 = x.to(torch.float32)
            variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
            registry.register(f"{prefix}.rmsnorm_variance", variance)
            rsqrt_val = torch.rsqrt(variance + eps)
            registry.register(f"{prefix}.rmsnorm_rsqrt", rsqrt_val)
            normalized = x_fp32 * rsqrt_val
            registry.register(f"{prefix}.rmsnorm_normalized", normalized.to(orig_dtype))
            scaled = normalized.to(orig_dtype) * weight
            registry.register(f"{prefix}.rmsnorm_scaled", scaled)
            return original_forward(x, *args, **kwargs)

        norm1.forward = patched_norm_forward

    # MoE gate captures
    mlp = find_module(layer, "mlp")
    if mlp:
        gate = find_module(mlp, "gate")
        if gate:

            def gate_hook(module, input, output):
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                registry.register(f"{prefix}.gate_logits", logits)
                sigmoid_out = torch.sigmoid(logits.float())
                registry.register(f"{prefix}.gate_sigmoid", sigmoid_out)

            hooks.append(gate.register_forward_hook(gate_hook))

        # Shared expert
        shared_gate_proj = find_module(mlp, "shared_expert_gate")
        shared_up_proj = find_module(mlp, "shared_expert_up")
        shared_down_proj = find_module(mlp, "shared_expert_down")

        if shared_gate_proj:
            hooks.append(
                shared_gate_proj.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.shared_gate")
                )
            )
        if shared_up_proj:
            hooks.append(
                shared_up_proj.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.shared_up")
                )
            )
        if shared_down_proj:
            hooks.append(
                shared_down_proj.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.shared_output")
                )
            )

    with torch.no_grad():
        ids = (
            input_ids.to(device).squeeze(0)
            if input_ids.dim() > 1
            else input_ids.to(device)
        )
        pos = (
            positions.to(device).squeeze(0)
            if positions.dim() > 1
            else positions.to(device)
        )
        model(ids, positions=pos)

    if norm1 and hasattr(norm1, "weight"):
        norm1.forward = original_forward
    for h in hooks:
        h.remove()


def run_scenario(model, config, scenario_name, device):
    """Run full capture for one scenario."""
    scenario = config["scenarios"][scenario_name]
    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario_name} — {scenario['description']}")
    print(f"{'=' * 70}")

    registry = CaptureRegistry(framework="vllm", scenario=scenario_name)

    # Extract model config
    hf_config = None
    for attr in ["config", "model_config"]:
        if hasattr(model, attr):
            hf_config = getattr(model, attr)
            break
    base = get_model_base(model)
    if hf_config is None and hasattr(base, "config"):
        hf_config = base.config
    if hf_config:
        registry.model_config = extract_model_config(hf_config)

    # Load input IDs from dInfer captures (for alignment) or tokenize fresh
    dinfer_ids_path = (
        Path(config["capture_root"])
        / scenario_name
        / f"dinfer.{scenario_name}.metadata.json"
    )
    if dinfer_ids_path.exists():
        with open(dinfer_ids_path) as f:
            dinfer_meta = json.load(f)
        input_ids = torch.tensor([dinfer_meta["input_ids"]])
        positions = torch.tensor([dinfer_meta["positions"]])
        print(f"  Loaded input_ids from dInfer capture: {dinfer_meta['input_ids']}")
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], trust_remote_code=True
        )
        input_ids = tokenizer(config["test_prompt"], return_tensors="pt")["input_ids"]
        positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
        print(f"  Tokenized fresh: {input_ids[0].tolist()}")

    registry.input_ids = input_ids[0].tolist()
    registry.positions = positions[0].tolist()

    num_layers = len(list(get_model_base(model).layers))
    print(f"  Tokens: {registry.input_ids}")
    print(f"  Num layers: {num_layers}")

    # L4
    print("\n[L4] Capturing all layer hidden states...")
    capture_L4(model, registry, input_ids, positions, device)
    print(
        f"  Captured {sum(1 for k in registry.captures if k.startswith('L4_'))} L4 checkpoints"
    )

    # L5 for ALL layers
    print(f"\n[L5] Capturing sub-layer operations for all {num_layers} layers...")
    for layer_idx in range(num_layers):
        capture_L5_layer(model, registry, layer_idx, input_ids, positions, device)
        count = sum(
            1
            for k in registry.captures
            if f"layer{layer_idx:02d}" in k and k.startswith("L5_")
        )
        print(f"  Layer {layer_idx:2d}: {count} checkpoints")

    # L6 for ALL layers
    print(f"\n[L6] Capturing sub-attention operations for all {num_layers} layers...")
    for layer_idx in range(num_layers):
        capture_L6_layer(model, registry, layer_idx, input_ids, positions, device)
        count = sum(
            1
            for k in registry.captures
            if f"layer{layer_idx:02d}" in k and k.startswith("L6_")
        )
        print(f"  Layer {layer_idx:2d}: {count} checkpoints")

    # L7 for ALL layers
    print(f"\n[L7] Capturing atomic operations for all {num_layers} layers...")
    for layer_idx in range(num_layers):
        capture_L7_layer(model, registry, layer_idx, input_ids, positions, device)
        count = sum(
            1
            for k in registry.captures
            if f"layer{layer_idx:02d}" in k and k.startswith("L7_")
        )
        print(f"  Layer {layer_idx:2d}: {count} checkpoints")

    # Save
    out_dir = registry.save(config["capture_root"])
    total = len(registry.captures)
    print(f"\n[DONE] {scenario_name}: {total} total checkpoints saved to {out_dir}")
    return registry


def main():
    parser = argparse.ArgumentParser(description="Capture vLLM/dllm-plugin tensors")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda:0")

    print("Loading vLLM model...")
    llm, vllm_model, runner = load_vllm_model(config["model_path"])
    discover_model_structure(vllm_model)

    scenarios = (
        list(config["scenarios"].keys())
        if args.all_scenarios
        else [args.scenario or "first_block"]
    )

    for scenario_name in scenarios:
        run_scenario(vllm_model, config, scenario_name, device)

    print(f"\n{'=' * 70}")
    print("ALL CAPTURES COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
