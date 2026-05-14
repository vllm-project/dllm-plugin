#!/usr/bin/env python3
"""Standalone tensor capture from dllm-plugin LLaDA2ForCausalLM.

Loads the model directly (no vLLM engine) for pure numerical comparison
against dInfer captures. Uses identical weights and forward pass logic.

Usage:
    PYTHONPATH=/workspace/dllm-plugin python3 capture_vllm_standalone.py --scenario first_block
    PYTHONPATH=/workspace/dllm-plugin python3 capture_vllm_standalone.py --all-scenarios
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "dllm")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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
    """Load LLaDA2ForCausalLM via vLLM LLM engine with model_impl bypass.

    This is the established pattern from the validation scripts — uses
    register_dllm() + model_impl parameter to bypass ModelConfig validation.
    """
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
    hf_config = llm.llm_engine.model_config.hf_config
    vllm_config = runner.vllm_config
    return llm, vllm_model, hf_config, vllm_config


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


def get_model_base(model):
    """Get the base transformer model."""
    for attr in ["model", "transformer"]:
        if hasattr(model, attr):
            base = getattr(model, attr)
            if hasattr(base, "layers"):
                return base
    return model


def capture_all_levels(model, registry, input_ids, positions, device):
    """Capture L4-L7 in a single forward pass with comprehensive hooks."""
    hooks = []
    base = get_model_base(model)
    num_layers = len(list(base.layers))

    # ── L4: Layer hidden states ──
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

    # ── L5: Sub-layer operations ──
    for i, layer in enumerate(base.layers):
        prefix = f"L5_sublayer.layer{i:02d}"

        norm1 = find_module(layer, "input_layernorm")
        if norm1:
            hooks.append(
                norm1.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.input_norm_out")
                )
            )

        attn = find_module(layer, "self_attn")
        if attn:
            hooks.append(
                attn.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.attn_out")
                )
            )

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

    # ── L6: Sub-attention operations ──
    for i, layer in enumerate(base.layers):
        prefix = f"L6_subattn.layer{i:02d}"
        attn = find_module(layer, "self_attn")
        if attn is None:
            continue

        qkv = find_module(attn, "qkv_proj")
        if qkv:
            hooks.append(
                qkv.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.qkv_proj_out")
                )
            )

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

        o_proj = find_module(attn, "o_proj")
        if o_proj:
            hooks.append(
                o_proj.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.o_proj_out")
                )
            )

    # ── L7: Atomic operations (RMSNorm internals + MoE gate) ──
    for i, layer in enumerate(base.layers):
        prefix = f"L7_atomic.layer{i:02d}"

        # RMSNorm internals via monkey-patch
        norm1 = find_module(layer, "input_layernorm")
        if norm1 and hasattr(norm1, "weight"):
            original_forward = norm1.forward
            eps = getattr(norm1, "variance_epsilon", getattr(norm1, "eps", 1e-6))
            w = norm1.weight
            _prefix = prefix  # capture in closure
            _registry = registry

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

        # MoE gate
        mlp = find_module(layer, "mlp")
        if mlp:
            gate = find_module(mlp, "gate")
            if gate:
                _pfx = prefix

                def make_gate_hook(pfx, reg):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            logits = output[0]
                        else:
                            logits = output
                        reg.register(f"{pfx}.gate_logits", logits)
                        sig = torch.sigmoid(logits.float())
                        reg.register(f"{pfx}.gate_sigmoid", sig)

                    return hook

                hooks.append(
                    gate.register_forward_hook(make_gate_hook(prefix, registry))
                )

            # Shared expert
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

    # ── Run forward pass ──
    print(f"  Running forward pass with {len(hooks)} hooks...")
    with torch.no_grad():
        ids = input_ids.to(device)
        pos = positions.to(device)
        if ids.dim() > 1:
            ids = ids.squeeze(0)
        if pos.dim() > 1:
            pos = pos.squeeze(0)

        # Set up minimal forward context required by LLaDA2BlockAttention
        from vllm.config import get_current_vllm_config
        from vllm.forward_context import set_forward_context

        vllm_config = get_current_vllm_config()
        with set_forward_context(None, vllm_config):
            output = model(ids, positions=pos)

    # Capture logits
    if isinstance(output, tuple):
        logits = output[0]
    elif hasattr(output, "logits"):
        logits = output.logits
    else:
        logits = output

    if logits is not None and isinstance(logits, torch.Tensor):
        registry.register("L3_logits.global.raw_logits", logits)
        probs = torch.softmax(logits.float(), dim=-1)
        registry.register("L2_probs.global.softmax_probs", probs)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Restore original forwards
    for i, layer in enumerate(base.layers):
        norm1 = find_module(layer, "input_layernorm")
        if norm1 and hasattr(norm1, "_original_forward"):
            norm1.forward = norm1._original_forward
            del norm1._original_forward

    return logits


def run_scenario(model, config, scenario_name, device):
    """Run full capture for one scenario."""
    scenario = config["scenarios"][scenario_name]
    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario_name} — {scenario['description']}")
    print(f"{'=' * 70}")

    registry = CaptureRegistry(framework="vllm", scenario=scenario_name)

    # Extract model config
    hf_config = model.config if hasattr(model, "config") else None
    if hf_config is None:
        base = get_model_base(model)
        hf_config = getattr(base, "config", None)
    if hf_config:
        registry.model_config = extract_model_config(hf_config)

    # Load input IDs from dInfer captures for alignment
    dinfer_meta_path = (
        Path(config["capture_root"])
        / scenario_name
        / f"dinfer.{scenario_name}.metadata.json"
    )
    if dinfer_meta_path.exists():
        with open(dinfer_meta_path) as f:
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

    # Capture everything in one pass
    capture_all_levels(model, registry, input_ids, positions, device)

    # Count per level
    for level in ["L2", "L3", "L4", "L5", "L6", "L7"]:
        count = sum(1 for k in registry.captures if k.startswith(f"{level}_"))
        print(f"  {level}: {count} checkpoints")

    out_dir = registry.save(config["capture_root"])
    total = len(registry.captures)
    print(f"\n[DONE] {scenario_name}: {total} total checkpoints saved to {out_dir}")
    return registry


def main():
    parser = argparse.ArgumentParser(description="Standalone vLLM capture")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("Loading LLaDA2ForCausalLM via vLLM engine...")
    llm, model, hf_config, vllm_config = load_vllm_model(config["model_path"])

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

    # Run captures inside vLLM config context (required for forward pass)
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        for scenario_name in scenarios:
            run_scenario(model, config, scenario_name, device)

    print(f"\n{'=' * 70}")
    print("ALL CAPTURES COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
