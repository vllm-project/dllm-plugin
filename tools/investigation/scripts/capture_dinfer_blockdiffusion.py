#!/usr/bin/env python3
"""Capture tensors from dInfer using the validated BlockDiffusionLLM wrapper.

Uses dInfer's BlockDiffusionLLM.generate() with proper block semi-causal
attention mask (bidirectional within blocks, causal across blocks).

This is the CORRECT way to run dInfer — the HuggingFace AutoModelForCausalLM
path is faulty and produces garbled outputs.

Usage:
    python3 capture_dinfer_blockdiffusion.py --scenario first_block
    python3 capture_dinfer_blockdiffusion.py --all-scenarios
"""

import argparse
import os
import sys
from pathlib import Path

# Disable torch.compile to avoid Triton compilation overhead with 256-expert MoE
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch

# Also disable at runtime
torch.compiler.disable(recursive=True)

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

    return ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        expert_parallel_size=1,
    )


def load_model_and_wrapper(
    model_path: str,
    parallel_config,
    device: torch.device,
    test_prompt: str = "The quick brown fox",
):
    """Load dInfer model and create BlockDiffusionLLM wrapper."""
    from dinfer import (
        BlockDiffusionLLM,
        BlockIteratorFactory,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeModelLM
    from transformers import AutoConfig, AutoTokenizer
    from vllm.config import VllmConfig, set_current_vllm_config

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = LLaDA2MoeModelLM(config=model_config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

    mask_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id

    decoder = ThresholdParallelDecoder(
        temperature=0,
        threshold=0.9,
        mask_id=mask_id,
        eos_id=eos_id,
    )

    # max_length must match actual sequence length (prompt + gen_length, rounded to block)
    # Default of 2048 creates a 2048x2048 attention mask, killing GPU utilization
    prompt_len = tokenizer(test_prompt, return_tensors="pt")["input_ids"].shape[1]
    gen_length = 32
    total_len = ((prompt_len + gen_length + 31) // 32) * 32
    max_len = max(total_len, 64)

    dllm = BlockDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True, max_length=max_len),
        early_stop=True,
    )
    print(f"  BlockDiffusionLLM created with max_length={max_len}")

    return model, dllm, tokenizer, model_config


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


def register_all_hooks(model, registry):
    """Register hooks on the raw model (before wrapping with BlockDiffusionLLM).

    Hooks capture from every forward pass. Since BlockDiffusionLLM runs
    multiple iterations per block (to unmask tokens), hooks will be called
    multiple times. The last call's values overwrite previous ones, so we
    capture the LAST iteration's state for each block.
    """
    hooks = []
    base = model.model if hasattr(model, "model") else model

    # L4: Layer hidden states
    embed = find_module(base, "word_embeddings", "embed_tokens")
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

    final_norm = find_module(base, "norm", "final_layernorm")
    if final_norm:
        hooks.append(
            final_norm.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.final_norm")
            )
        )

    lm_head = find_module(model, "lm_head", "output")
    if lm_head:
        hooks.append(
            lm_head.register_forward_hook(
                make_capture_hook(registry, "L4_layer.global.lm_head")
            )
        )

    # L5: Sub-layer
    for i, layer in enumerate(base.layers):
        prefix = f"L5_sublayer.layer{i:02d}"
        norm1 = find_module(layer, "input_layernorm")
        if norm1:
            hooks.append(
                norm1.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.input_norm_out")
                )
            )
        attn = find_module(layer, "attention", "self_attn")
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

    # L6: Sub-attention
    for i, layer in enumerate(base.layers):
        prefix = f"L6_subattn.layer{i:02d}"
        attn = find_module(layer, "attention", "self_attn")
        if not attn:
            continue
        qkv = find_module(attn, "query_key_value", "qkv_proj")
        if qkv:
            hooks.append(
                qkv.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.qkv_proj_out")
                )
            )
        q_norm = find_module(attn, "query_layernorm", "q_norm")
        k_norm = find_module(attn, "key_layernorm", "k_norm")
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
        o_proj = find_module(attn, "dense", "o_proj")
        if o_proj:
            hooks.append(
                o_proj.register_forward_hook(
                    make_capture_hook(registry, f"{prefix}.o_proj_out")
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
            gate = find_module(mlp, "gate", "router")
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
                ("shared_experts.gate_proj", "shared_gate"),
                ("shared_experts.up_proj", "shared_up"),
                ("shared_experts.down_proj", "shared_output"),
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
    base = model.model if hasattr(model, "model") else model
    for layer in base.layers:
        norm1 = find_module(layer, "input_layernorm")
        if norm1 and hasattr(norm1, "_original_forward"):
            norm1.forward = norm1._original_forward
            del norm1._original_forward


def run_scenario(model, dllm, tokenizer, config, scenario_name, device):
    """Run capture using BlockDiffusionLLM.generate()."""
    scenario = config["scenarios"][scenario_name]
    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario_name} — {scenario['description']}")
    print(f"{'=' * 70}")

    registry = CaptureRegistry(framework="dinfer", scenario=scenario_name)
    model_config = model.config if hasattr(model, "config") else model.model.config
    registry.model_config = extract_model_config(model_config)

    prompt = config["test_prompt"]
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    registry.input_ids = input_ids[0].tolist()
    registry.positions = list(range(input_ids.shape[1]))

    print(f"  Prompt: '{prompt}'")
    print(f"  Tokens: {registry.input_ids} ({input_ids.shape[1]} tokens)")

    base = model.model if hasattr(model, "model") else model
    num_layers = len(list(base.layers))
    print(f"  Num layers: {num_layers}")

    # Register hooks on the raw model
    hooks = register_all_hooks(model, registry)

    # Run generation via BlockDiffusionLLM (validated path)
    from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config
    from vllm.forward_context import set_forward_context

    print("  Running BlockDiffusionLLM.generate(block_length=32)...")

    # dInfer requires forward_context for vLLM backend
    try:
        vllm_config = get_current_vllm_config()
    except (AssertionError, RuntimeError):
        from vllm.config import ParallelConfig

        vllm_config = VllmConfig(
            parallel_config=ParallelConfig(
                pipeline_parallel_size=1, tensor_parallel_size=1, expert_parallel_size=1
            )
        )

    with set_current_vllm_config(vllm_config):
        with set_forward_context(None, vllm_config):
            outputs = dllm.generate(
                input_ids,
                gen_length=32,
                block_length=32,
            )

    # Capture generated tokens
    if outputs is not None:
        generated_ids = outputs[0, input_ids.shape[1] :].cpu().tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        registry.register_non_tensor("L1_output.token_ids", generated_ids)
        registry.register_non_tensor("L1_output.token_text", generated_text)
        print(f"  Generated: {generated_text[:100]}...")

    cleanup_hooks(model, hooks)

    for level in ["L2", "L3", "L4", "L5", "L6", "L7"]:
        count = sum(1 for k in registry.captures if k.startswith(f"{level}_"))
        print(f"  {level}: {count} checkpoints")

    out_dir = registry.save(config["capture_root"])
    total = len(registry.captures)
    print(f"\n[DONE] {scenario_name}: {total} checkpoints -> {out_dir}")
    return registry


def main():
    parser = argparse.ArgumentParser(description="dInfer BlockDiffusionLLM capture")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("Initializing distributed...")
    parallel_config = init_distributed()

    print("Loading dInfer model + BlockDiffusionLLM wrapper...")
    model, dllm, tokenizer, model_config = load_model_and_wrapper(
        config["model_path"],
        parallel_config,
        device,
        test_prompt=config["test_prompt"],
    )

    # Print structure
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
        run_scenario(model, dllm, tokenizer, config, scenario_name, device)

    print(f"\n{'=' * 70}")
    print("ALL CAPTURES COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
