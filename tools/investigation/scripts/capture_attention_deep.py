#!/usr/bin/env python3
"""Deep attention kernel capture: instruments the attention forward to capture
every intermediate between Q/K norm and o_proj output.

Captures 9 checkpoints (A1-A9) per layer by monkey-patching the attention
forward method, calling the REAL operations, and saving intermediates.

Usage:
    # dInfer pod:
    python3 capture_attention_deep.py --framework dinfer --layer 0

    # vLLM pod:
    PYTHONPATH=/workspace/dllm-plugin python3 capture_attention_deep.py --framework vllm --layer 0
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from capture_lib import CaptureRegistry, extract_model_config, load_config

# ═══════════════════════════════════════════════════════════════════════
#  dInfer instrumented attention forward
# ═══════════════════════════════════════════════════════════════════════


def instrument_dinfer_attention(model, layer_idx, registry):
    """Monkey-patch dInfer's LLaDA2MoeSdpaAttention.forward for layer_idx."""
    layer = model.model.layers[layer_idx]
    attn = layer.attention
    original_forward = attn.forward

    def instrumented_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        position_embeddings=None,
        cache_position=None,
        replace_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        tp_size = getattr(attn, "tp_size", 1)

        # QKV projection
        qkv = attn.query_key_value(hidden_states)
        num_heads = attn.num_heads // tp_size
        num_kv_heads = attn.num_key_value_heads // tp_size
        head_dim = attn.head_dim

        qkv = qkv.view(bsz, q_len, num_heads + 2 * num_kv_heads, head_dim)
        query_states, key_states, value_states = qkv.split(
            [num_heads, num_kv_heads, num_kv_heads], dim=-2
        )

        query_states = query_states.transpose(1, 2)  # [bsz, heads, seq, dim]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # A1, A2: Q/K after norm (pre-RoPE)
        query_states = attn.query_layernorm(query_states)
        key_states = attn.key_layernorm(key_states)
        registry.register("A1_q_after_norm", query_states)
        registry.register("A2_k_after_norm", key_states)

        # A3, A4: RoPE cos/sin
        cos, sin = position_embeddings  # type: ignore[misc]
        registry.register("A3_rope_cos", cos)
        registry.register("A4_rope_sin", sin)

        # A5, A6: Q/K after RoPE
        from dinfer.model.modeling_llada2_moe import apply_rotary_pos_emb

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        registry.register("A5_q_after_rope", query_states)
        registry.register("A6_k_after_rope", key_states)
        registry.register("A6b_v_states", value_states)

        # KV cache update
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, attn.layer_idx, replace_position
            )
        if use_cache:
            past_key_value = (key_states, value_states)

        # A7: GQA expand
        from dinfer.model.modeling_llada2_moe import repeat_kv

        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)
        registry.register("A7_k_after_gqa", key_states)

        # Attention mask handling
        if attention_mask is not None:
            kv_seq_len = key_states.shape[-2]
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                attention_mask = attention_mask.unsqueeze(1)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attention_mask_bool = (
            attention_mask.bool() if attention_mask is not None else None
        )

        # A8: SDPA attention output
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask_bool,
            dropout_p=0.0,
            is_causal=attn.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        registry.register("A8_attn_output_pre_oproj", attn_output)

        # A9: o_proj
        attn_output = attn.dense(attn_output)
        registry.register("A9_oproj_output", attn_output)

        return attn_output, None, past_key_value

    attn.forward = instrumented_forward
    attn._original_forward = original_forward
    print(f"  Instrumented dInfer layer {layer_idx} attention")


# ═══════════════════════════════════════════════════════════════════════
#  vLLM instrumented attention forward
# ═══════════════════════════════════════════════════════════════════════


def instrument_vllm_attention(model, layer_idx, registry):
    """Monkey-patch dllm-plugin's LLaDA2BlockAttention.forward for layer_idx."""
    base = model.model if hasattr(model, "model") else model
    layer = base.layers[layer_idx]
    attn = layer.self_attn
    original_forward = attn.forward

    def instrumented_forward(positions, hidden_states):
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_size = attn.head_size

        # QKV projection
        qkv, _ = attn.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size],
            dim=-1,
        )

        num_tokens = q.shape[0]
        q = q.view(num_tokens, num_heads, head_size)
        k = k.view(num_tokens, num_kv_heads, head_size)

        # A1, A2: Q/K after norm
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        registry.register("A1_q_after_norm", q)
        registry.register("A2_k_after_norm", k)

        # Flatten for RoPE
        q = q.reshape(num_tokens, num_heads * head_size)
        k = k.reshape(num_tokens, num_kv_heads * head_size)

        # A3, A4: Capture RoPE cos/sin by intercepting rotary_emb
        original_rotary = attn.rotary_emb.forward

        rope_captures = {}

        def intercepted_rotary(pos, q_in, k_in, offsets=None):
            # vLLM's get_rope returns a RotaryEmbedding that takes (positions, q, k)
            # We need to capture cos/sin from inside
            result = (
                original_rotary(pos, q_in, k_in, offsets=offsets)
                if offsets is not None
                else original_rotary(pos, q_in, k_in)
            )
            return result

        # Call RoPE
        q, k = attn.rotary_emb(positions, q, k)

        # A5, A6: Q/K after RoPE
        registry.register("A5_q_after_rope", q)
        registry.register("A6_k_after_rope", k)
        registry.register("A6b_v_states", v)

        # A8: Attention output (vLLM combines GQA + attention in self.attn)
        # In prefill mode without num_prefix_tokens, falls through to simple attention
        from dllm_plugin.forward_context import get_num_prefix_tokens_list

        num_prefix_tokens_list = get_num_prefix_tokens_list()

        if num_prefix_tokens_list is not None:
            # Chunked block attention path
            from vllm.forward_context import get_forward_context

            context = get_forward_context()
            attn_output = attn._forward_concatenated(
                query=q,
                key=k,
                value=v,
                attn_metadata=context.attn_metadata if context else None,
                num_prefix_tokens_list=num_prefix_tokens_list,
            )
        else:
            # Simple bidirectional attention (prefill path)
            attn_output = attn.attn(query=q, key=k, value=v)

        registry.register("A8_attn_output_pre_oproj", attn_output)

        # A9: o_proj
        output, _ = attn.o_proj(attn_output)
        registry.register("A9_oproj_output", output)

        return output

    attn.forward = instrumented_forward
    attn._original_forward = original_forward
    print(f"  Instrumented vLLM layer {layer_idx} attention")


# ═══════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════


def load_dinfer(model_path, device):
    """Load dInfer model."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "45610")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from dinfer.model.modeling_llada2_moe import LLaDA2MoeModelLM
    from transformers import AutoConfig, AutoTokenizer
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1, tensor_parallel_size=1, expert_parallel_size=1
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = LLaDA2MoeModelLM(config=model_config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

    return model, tokenizer


def load_vllm(model_path):
    """Load vLLM model via engine."""
    os.environ.setdefault("VLLM_PLUGINS", "dllm")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_DLLM_USE_MOCK_MODEL", "0")
    os.environ.setdefault("VLLM_DLLM_STRICT_STACK_VALIDATION", "0")

    from dllm_plugin import register_dllm

    register_dllm()
    from transformers import AutoTokenizer
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
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    model = runner.model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vllm_config = runner.vllm_config
    return llm, model, tokenizer, vllm_config


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Deep attention capture")
    parser.add_argument("--framework", required=True, choices=["dinfer", "vllm"])
    parser.add_argument(
        "--layer", type=int, default=0, help="Layer index to instrument"
    )
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    layers = list(range(20)) if args.all_layers else [args.layer]

    if args.framework == "dinfer":
        print("Loading dInfer model...")
        model, tokenizer = load_dinfer(config["model_path"], device)

        input_ids = tokenizer(config["test_prompt"], return_tensors="pt")[
            "input_ids"
        ].to(device)
        print(f"Input: {input_ids[0].tolist()} ({input_ids.shape[1]} tokens)")

        for layer_idx in layers:
            registry = CaptureRegistry(
                framework="dinfer", scenario=f"attn_deep_layer{layer_idx:02d}"
            )
            registry.input_ids = input_ids[0].tolist()
            registry.model_config = extract_model_config(model.config)

            instrument_dinfer_attention(model, layer_idx, registry)

            with torch.no_grad():
                model(input_ids)

            # Restore original
            layer = model.model.layers[layer_idx]
            layer.attention.forward = layer.attention._original_forward
            del layer.attention._original_forward

            registry.save(config["capture_root"])
            print(f"  Layer {layer_idx}: {len(registry.captures)} checkpoints")

    elif args.framework == "vllm":
        print("Loading vLLM model...")
        llm, model, tokenizer, vllm_config = load_vllm(config["model_path"])

        input_ids_list = tokenizer(config["test_prompt"])["input_ids"]
        print(f"Input: {input_ids_list} ({len(input_ids_list)} tokens)")

        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(vllm_config):
            for layer_idx in layers:
                registry = CaptureRegistry(
                    framework="vllm", scenario=f"attn_deep_layer{layer_idx:02d}"
                )
                registry.input_ids = input_ids_list
                base = model.model if hasattr(model, "model") else model
                registry.model_config = extract_model_config(
                    base.config if hasattr(base, "config") else model.config
                )

                instrument_vllm_attention(model, layer_idx, registry)

                # Trigger forward via generate
                from vllm import SamplingParams

                llm.generate(
                    [config["test_prompt"]], SamplingParams(temperature=0, max_tokens=1)
                )

                # Restore original
                base_layers = base.layers
                base_layers[layer_idx].self_attn.forward = base_layers[
                    layer_idx
                ].self_attn._original_forward
                del base_layers[layer_idx].self_attn._original_forward

                registry.save(config["capture_root"])
                print(f"  Layer {layer_idx}: {len(registry.captures)} checkpoints")

    print(f"\n{'=' * 70}")
    print("DEEP ATTENTION CAPTURE COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
