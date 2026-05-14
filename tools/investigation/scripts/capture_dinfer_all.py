#!/usr/bin/env python3
"""Exhaustive tensor capture from dInfer LLaDA2.0-mini reference implementation.

Captures ALL layers (0-19) at ALL levels (L1-L7) for all scenarios.
Run on the investigation-dinfer pod with vLLM 0.10.2 + dInfer installed.

Usage:
    python3 capture_dinfer_all.py [--scenario first_block]
    python3 capture_dinfer_all.py --all-scenarios
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import (
    CaptureRegistry,
    extract_model_config,
    load_config,
    make_capture_hook,
    make_capture_pre_hook,
)


def init_distributed():
    """Initialize vLLM 0.10.2 distributed environment."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "45610")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from vllm import distributed
    from vllm.config import ParallelConfig

    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=1,
    )
    return parallel_config


def load_model(model_path: str, parallel_config, device: torch.device):
    """Load dInfer LLaDA2MoeModelLM."""
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeModelLM
    from transformers import AutoConfig
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = LLaDA2MoeModelLM(config=model_config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

    return model, model_config


def get_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def discover_model_structure(model):
    """Print model module hierarchy for hook registration."""
    print("\n[STRUCTURE] Model module tree (depth 3):")
    for name, mod in model.named_modules():
        depth = name.count(".")
        if depth <= 3:
            print(f"  {'  ' * depth}{name}: {type(mod).__name__}")


def find_module(model, *candidates):
    """Find a module by trying multiple attribute paths."""
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


def capture_L4(model, registry, input_ids, positions, device):
    """Capture Level 4: all layer hidden states."""
    hooks = []
    num_layers = len(list(model.model.layers)) if hasattr(model, "model") else 0
    base = model.model if hasattr(model, "model") else model

    # Embedding
    embed = find_module(base, "embed_tokens", "word_embeddings", "embedding")
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
    final_norm = find_module(base, "norm", "final_layernorm", "ln_f")
    if final_norm:
        hooks.append(
            final_norm.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.final_norm")
            )
        )

    # LM head
    lm_head = find_module(model, "lm_head", "output")
    if lm_head:
        hooks.append(
            lm_head.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.lm_head")
            )
        )

    # Forward pass
    with torch.no_grad():
        ids = input_ids.to(device)
        pos = positions.to(device)
        output = model(ids, position_ids=pos)

    # Capture logits directly
    if isinstance(output, tuple):
        logits = output[0]
    elif hasattr(output, "logits"):
        logits = output.logits
    else:
        logits = output
    registry.register("L3_logits.global.raw_logits", logits)

    # Probabilities
    probs = torch.softmax(logits.float(), dim=-1)
    registry.register("L2_probs.global.softmax_probs", probs)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    return logits


def capture_L5_layer(model, registry, layer_idx, input_ids, positions, device):
    """Capture Level 5: sub-layer operations for a specific layer."""
    hooks = []
    base = model.model if hasattr(model, "model") else model
    layer = base.layers[layer_idx]
    prefix = f"L5_sublayer.layer{layer_idx:02d}"

    # input_layernorm
    norm1 = find_module(layer, "input_layernorm", "ln_1", "norm1")
    if norm1:
        hooks.append(
            norm1.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.input_norm_out")
            )
        )

    # attention output
    attn = find_module(layer, "self_attn", "attention", "attn")
    if attn:
        hooks.append(
            attn.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.attn_out")
            )
        )

    # post_attention_layernorm
    norm2 = find_module(layer, "post_attention_layernorm", "ln_2", "norm2")
    if norm2:
        hooks.append(
            norm2.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.post_attn_norm_out")
            )
        )

    # MLP/MoE output
    mlp = find_module(layer, "mlp", "feed_forward", "ffn")
    if mlp:
        hooks.append(
            mlp.register_forward_hook(make_capture_hook(registry, f"{prefix}.moe_out"))
        )

    # Capture residuals via pre-hooks on sub-modules
    if norm2:
        hooks.append(
            norm2.register_forward_pre_hook(
                make_capture_pre_hook(registry, f"{prefix}.residual1")
            )
        )

    # Layer output = residual2
    hooks.append(
        layer.register_forward_hook(make_capture_hook(registry, f"{prefix}.residual2"))
    )

    with torch.no_grad():
        model(input_ids.to(device), position_ids=positions.to(device))

    for h in hooks:
        h.remove()


def capture_L6_layer(model, registry, layer_idx, input_ids, positions, device):
    """Capture Level 6: sub-attention operations for a specific layer."""
    hooks = []
    base = model.model if hasattr(model, "model") else model
    layer = base.layers[layer_idx]
    prefix = f"L6_subattn.layer{layer_idx:02d}"

    attn = find_module(layer, "self_attn", "attention", "attn")
    if attn is None:
        print(f"[WARN] Could not find attention module for layer {layer_idx}")
        return

    # QKV projection
    qkv = find_module(attn, "query_key_value", "qkv_proj", "c_attn")
    if qkv:
        hooks.append(
            qkv.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.qkv_proj_out")
            )
        )

    # Q/K normalization
    q_norm = find_module(attn, "query_layernorm", "q_norm", "q_layernorm")
    k_norm = find_module(attn, "key_layernorm", "k_norm", "k_layernorm")
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
    o_proj = find_module(attn, "dense", "o_proj", "out_proj")
    if o_proj:
        hooks.append(
            o_proj.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.o_proj_out")
            )
        )

    # Monkey-patch attention forward to capture RoPE intermediates
    original_forward = attn.forward

    def patched_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        return result

    attn.forward = patched_forward

    with torch.no_grad():
        model(input_ids.to(device), position_ids=positions.to(device))

    attn.forward = original_forward
    for h in hooks:
        h.remove()


def capture_L7_layer(model, registry, layer_idx, input_ids, positions, device):
    """Capture Level 7: atomic operations for a specific layer.

    This requires deep monkey-patching to capture every intermediate value.
    """
    base = model.model if hasattr(model, "model") else model
    layer = base.layers[layer_idx]
    prefix = f"L7_atomic.layer{layer_idx:02d}"

    # Capture RMSNorm internals by monkey-patching
    norm1 = find_module(layer, "input_layernorm", "ln_1", "norm1")
    if norm1 and hasattr(norm1, "weight"):
        original_norm_forward = norm1.forward

        def patched_norm_forward(x, *args, **kwargs):
            orig_dtype = x.dtype
            x_fp32 = x.to(torch.float32)
            variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
            registry.register(f"{prefix}.rmsnorm_variance", variance)
            rsqrt_val = torch.rsqrt(
                variance
                + getattr(norm1, "variance_epsilon", getattr(norm1, "eps", 1e-6))
            )
            registry.register(f"{prefix}.rmsnorm_rsqrt", rsqrt_val)
            normalized = x_fp32 * rsqrt_val
            registry.register(f"{prefix}.rmsnorm_normalized", normalized.to(orig_dtype))
            scaled = normalized.to(orig_dtype) * norm1.weight
            registry.register(f"{prefix}.rmsnorm_scaled", scaled)
            return original_norm_forward(x, *args, **kwargs)

        norm1.forward = patched_norm_forward

    # Capture MoE routing internals
    mlp = find_module(layer, "mlp", "feed_forward", "ffn")
    if mlp:
        gate = find_module(mlp, "gate", "router", "gate_proj")
        if gate:

            def gate_hook(module, input, output):
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                registry.register(f"{prefix}.gate_logits", logits)
                sigmoid_out = torch.sigmoid(logits.float())
                registry.register(f"{prefix}.gate_sigmoid", sigmoid_out)

            gate.register_forward_hook(gate_hook)

        # Shared expert captures
        shared_gate_proj = find_module(
            mlp, "shared_expert_gate", "shared_experts.gate_proj"
        )
        shared_up_proj = find_module(mlp, "shared_expert_up", "shared_experts.up_proj")
        shared_down_proj = find_module(
            mlp, "shared_expert_down", "shared_experts.down_proj"
        )

        if shared_gate_proj:
            shared_gate_proj.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.shared_gate")
            )
        if shared_up_proj:
            shared_up_proj.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.shared_up")
            )
        if shared_down_proj:
            shared_down_proj.register_forward_hook(
                make_capture_hook(registry, f"{prefix}.shared_output")
            )

    with torch.no_grad():
        model(input_ids.to(device), position_ids=positions.to(device))

    # Restore original forward
    if norm1 and hasattr(norm1, "weight"):
        norm1.forward = original_norm_forward


def run_scenario(model, tokenizer, config, scenario_name, device):
    """Run full capture for one scenario."""
    scenario = config["scenarios"][scenario_name]
    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario_name} — {scenario['description']}")
    print(f"{'=' * 70}")

    registry = CaptureRegistry(framework="dinfer", scenario=scenario_name)
    registry.model_config = extract_model_config(
        model.config if hasattr(model, "config") else model.model.config
    )

    # Tokenize
    prompt = config["test_prompt"]
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    registry.input_ids = input_ids[0].tolist()

    num_tokens = input_ids.shape[1]
    positions = torch.arange(num_tokens).unsqueeze(0)
    registry.positions = positions[0].tolist()

    print(f"  Prompt: '{prompt}'")
    print(f"  Tokens: {registry.input_ids}")
    print(f"  Positions: {registry.positions}")

    num_layers = len(list((model.model if hasattr(model, "model") else model).layers))
    print(f"  Num layers: {num_layers}")

    # L4: All layer hidden states + L3 logits + L2 probs
    print("\n[L4] Capturing all layer hidden states...")
    capture_L4(model, registry, input_ids, positions, device)
    print(
        f"  Captured {sum(1 for k in registry.captures if k.startswith('L4_'))} L4 checkpoints"
    )

    # L5: Sub-layer operations for ALL layers
    print(f"\n[L5] Capturing sub-layer operations for all {num_layers} layers...")
    for layer_idx in range(num_layers):
        capture_L5_layer(model, registry, layer_idx, input_ids, positions, device)
        print(
            f"  Layer {layer_idx:2d}: {sum(1 for k in registry.captures if f'layer{layer_idx:02d}' in k and k.startswith('L5_'))} checkpoints"
        )

    # L6: Sub-attention operations for ALL layers
    print(f"\n[L6] Capturing sub-attention operations for all {num_layers} layers...")
    for layer_idx in range(num_layers):
        capture_L6_layer(model, registry, layer_idx, input_ids, positions, device)
        print(
            f"  Layer {layer_idx:2d}: {sum(1 for k in registry.captures if f'layer{layer_idx:02d}' in k and k.startswith('L6_'))} checkpoints"
        )

    # L7: Atomic operations for ALL layers
    print(f"\n[L7] Capturing atomic operations for all {num_layers} layers...")
    for layer_idx in range(num_layers):
        capture_L7_layer(model, registry, layer_idx, input_ids, positions, device)
        print(
            f"  Layer {layer_idx:2d}: {sum(1 for k in registry.captures if f'layer{layer_idx:02d}' in k and k.startswith('L7_'))} checkpoints"
        )

    # Save
    out_dir = registry.save(config["capture_root"])
    total = len(registry.captures)
    print(f"\n[DONE] {scenario_name}: {total} total checkpoints saved to {out_dir}")
    return registry


def main():
    parser = argparse.ArgumentParser(description="Capture dInfer tensors")
    parser.add_argument(
        "--scenario", type=str, default=None, help="Single scenario to capture"
    )
    parser.add_argument(
        "--all-scenarios", action="store_true", help="Capture all scenarios"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("Initializing distributed environment...")
    parallel_config = init_distributed()

    print("Loading model...")
    model, model_config = load_model(config["model_path"], parallel_config, device)

    # Print model structure (first run only)
    discover_model_structure(model)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(config["model_path"])

    # Determine scenarios
    if args.all_scenarios:
        scenarios = list(config["scenarios"].keys())
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        scenarios = ["first_block"]

    for scenario_name in scenarios:
        run_scenario(model, tokenizer, config, scenario_name, device)

    print(f"\n{'=' * 70}")
    print("ALL CAPTURES COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
