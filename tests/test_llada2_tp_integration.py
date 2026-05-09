# SPDX-License-Identifier: Apache-2.0
"""TP integration tests for LLaDA2.0 per-expert weight loading."""

import pytest

pytest.importorskip("torch")  # Skip if torch not available (standard CI)

from unittest.mock import MagicMock, patch  # noqa: E402

import torch  # noqa: E402


# Mock TP group for tests that create vLLM layers
@pytest.fixture(autouse=True)
def mock_tp_group():
    """Mock tensor parallel group for TP integration tests."""
    mock_group = MagicMock()
    mock_group.world_size = 1
    mock_group.rank = 0

    with (
        patch(
            "vllm.distributed.parallel_state.get_tp_group",
            return_value=mock_group,
        ),
        patch(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
            return_value=0,
        ),
    ):
        yield


@pytest.mark.gpu
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_llada2_tp_weight_loading(tp_size):
    """Validate per-expert weight loading distributes experts across TP ranks."""
    from dllm_plugin.models.llada2 import LLaDA2ForCausalLM

    # Mock vLLM config with TP
    config = MagicMock()
    config.model_config.hf_config.architectures = ["LLaDA2ForCausalLM"]
    config.model_config.hf_config.num_hidden_layers = 2
    config.model_config.hf_config.num_attention_heads = 8
    config.model_config.hf_config.hidden_size = 512
    config.model_config.hf_config.intermediate_size = 1024
    config.model_config.hf_config.moe_intermediate_size = 1024
    config.model_config.hf_config.num_experts = 8  # Small for testing
    config.model_config.hf_config.num_experts_per_tok = 2
    config.model_config.hf_config.shared_expert_intermediate_size = 1024
    config.model_config.hf_config.num_key_value_heads = 8
    config.model_config.hf_config.head_dim = 64
    config.model_config.hf_config.vocab_size = 1000
    config.model_config.hf_config.tie_word_embeddings = False
    config.model_config.hf_config.rope_theta = 10000.0
    config.model_config.hf_config.rope_scaling = None
    config.model_config.hf_config.max_position_embeddings = 2048
    config.parallel_config.tensor_parallel_size = tp_size
    config.cache_config = None
    config.quant_config = None
    config.lora_config = None

    # Create model
    model = LLaDA2ForCausalLM(vllm_config=config)

    # Simulate weight loading
    fake_weights = []

    # Add embeddings
    fake_weights.append(("model.embed_tokens.weight", torch.randn(1000, 512)))

    for layer_id in range(2):
        # Attention weights
        fake_weights.extend(
            [
                (
                    f"model.layers.{layer_id}.input_layernorm.weight",
                    torch.randn(512),
                ),
                (
                    f"model.layers.{layer_id}.self_attn.qkv_proj.weight",
                    torch.randn(1536, 512),
                ),
                (
                    f"model.layers.{layer_id}.self_attn.o_proj.weight",
                    torch.randn(512, 512),
                ),
                (
                    f"model.layers.{layer_id}.self_attn.q_norm.weight",
                    torch.randn(64),
                ),
                (
                    f"model.layers.{layer_id}.self_attn.k_norm.weight",
                    torch.randn(64),
                ),
                (
                    f"model.layers.{layer_id}.post_attention_layernorm.weight",
                    torch.randn(512),
                ),
            ]
        )

        # Shared expert weights
        fake_weights.extend(
            [
                (
                    f"model.layers.{layer_id}.mlp.shared_expert.gate_proj.weight",
                    torch.randn(1024, 512),
                ),
                (
                    f"model.layers.{layer_id}.mlp.shared_expert.up_proj.weight",
                    torch.randn(1024, 512),
                ),
                (
                    f"model.layers.{layer_id}.mlp.shared_expert.down_proj.weight",
                    torch.randn(512, 1024),
                ),
            ]
        )

        # Expert weights (8 experts)
        for expert_id in range(8):
            gate_w = torch.randn(1024, 512)
            up_w = torch.randn(1024, 512)
            down_w = torch.randn(512, 1024)

            expert_prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
            fake_weights.extend(
                [
                    (f"{expert_prefix}.gate_proj.weight", gate_w),
                    (f"{expert_prefix}.up_proj.weight", up_w),
                    (f"{expert_prefix}.down_proj.weight", down_w),
                ]
            )

        # Router weights
        fake_weights.append(
            (f"model.layers.{layer_id}.mlp.gate.weight", torch.randn(8, 512))
        )

    # Final layer norm
    fake_weights.append(("model.norm.weight", torch.randn(512)))

    # LM head
    fake_weights.append(("lm_head.weight", torch.randn(1000, 512)))

    # Track which expert_ids were loaded
    loaded_expert_ids = []

    def track_expert_id(param, loaded_weight, expert_id=None, **kwargs):
        """Track expert_id parameters in weight_loader calls."""
        if expert_id is not None:
            loaded_expert_ids.append(expert_id)

    # Patch weight_loader to track expert_id calls
    with patch(
        "dllm_plugin.models.llada2.default_weight_loader", side_effect=track_expert_id
    ):
        model.load_weights(fake_weights)

    # Validate expert_id parameter was passed for expert weights
    # With 8 experts and 2 layers, we expect 16 expert_id calls (8 experts × 2 layers)
    # Each expert has 2 weights (w13 and w2), so 16 expert loads total
    expected_expert_ids = 16  # 8 experts × 2 layers

    assert len(loaded_expert_ids) == expected_expert_ids, (
        f"Expected {expected_expert_ids} expert loads (8 experts × 2 layers), "
        f"got {len(loaded_expert_ids)}"
    )

    # All expert IDs 0-7 should be loaded for each layer
    unique_expert_ids = set(loaded_expert_ids)
    assert unique_expert_ids == set(range(8)), (
        f"Expected expert IDs 0-7, got {sorted(unique_expert_ids)}"
    )


@pytest.mark.gpu
def test_llada2_tp_uneven_distribution_warning(caplog):
    """Test warning for TP size that doesn't evenly divide experts."""
    from dllm_plugin.models.llada2 import LLaDA2MoE

    # Mock config with TP=3 and 256 experts (256 % 3 != 0)
    config = MagicMock()
    config.hidden_size = 512
    config.num_experts = 256
    config.num_experts_per_tok = 2
    config.intermediate_size = 1024
    config.moe_intermediate_size = 1024
    config.shared_expert_intermediate_size = 1024
    tp_size = 3

    # Create MoE layer (triggers warning in __init__)
    import logging

    with caplog.at_level(logging.WARNING):
        moe = LLaDA2MoE(
            config=config,
            tp_size=tp_size,
            prefix="test",
        )

    # Verify model was created successfully (despite uneven distribution)
    assert moe.num_experts == 256
    assert moe.tp_size == 3

    # Verify warning was logged
    assert "does not evenly divide" in caplog.text, (
        "Expected warning about uneven expert distribution"
    )
    assert "256 experts" in caplog.text
    assert f"TP size {tp_size}" in caplog.text or f"{tp_size}" in caplog.text


@pytest.mark.gpu
def test_llada2_no_tp_error_with_tp1():
    """Verify TP=1 doesn't trigger any warnings or errors."""
    from dllm_plugin.models.llada2 import LLaDA2MoE

    # Mock config with TP=1
    config = MagicMock()
    config.hidden_size = 512
    config.num_experts = 256
    config.num_experts_per_tok = 2
    config.intermediate_size = 1024
    config.moe_intermediate_size = 1024
    config.shared_expert_intermediate_size = 1024
    tp_size = 1

    # Create MoE layer (should not trigger any warnings)
    moe = LLaDA2MoE(
        config=config,
        tp_size=tp_size,
        prefix="test",
    )

    # Verify model was created successfully
    assert moe.num_experts == 256
    assert moe.tp_size == 1


@pytest.mark.gpu
def test_llada2_tp_even_distribution_no_warning(caplog):
    """Verify TP size that evenly divides experts doesn't warn."""
    from dllm_plugin.models.llada2 import LLaDA2MoE

    # Mock config with TP=4 and 256 experts (256 % 4 == 0)
    config = MagicMock()
    config.hidden_size = 512
    config.num_experts = 256
    config.num_experts_per_tok = 2
    config.intermediate_size = 1024
    config.moe_intermediate_size = 1024
    config.shared_expert_intermediate_size = 1024
    tp_size = 4

    # Create MoE layer (should not warn)
    import logging

    with caplog.at_level(logging.WARNING):
        moe = LLaDA2MoE(
            config=config,
            tp_size=tp_size,
            prefix="test",
        )

    # Verify model was created successfully
    assert moe.num_experts == 256
    assert moe.tp_size == 4

    # Verify no warning about uneven distribution
    assert "does not evenly divide" not in caplog.text, (
        "Expected no warning for even expert distribution (256 experts, TP=4)"
    )
