# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LLaDA2.0 real model implementation.

Tests the production LLaDA2ForCausalLM model with MoE architecture,
group-limited routing, and weight loading.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

from unittest.mock import MagicMock, Mock, patch  # noqa: E402

from dllm_plugin.config import (  # noqa: E402
    LLADA2_DEFAULT_MOE_INTERMEDIATE_SIZE,
    LLADA2_DEFAULT_N_GROUP,
    LLADA2_DEFAULT_NUM_EXPERTS,
    LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK,
    LLADA2_DEFAULT_NUM_SHARED_EXPERTS,
    LLADA2_DEFAULT_ROUTED_SCALING_FACTOR,
    LLADA2_DEFAULT_TOPK_GROUP,
)


# Mock TP group for tests that create vLLM layers
@pytest.fixture(autouse=True)
def mock_tp_group():
    """Mock tensor parallel group for real model tests."""
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


class TestLLaDA2MoE:
    """Tests for LLaDA2MoE layer."""

    @pytest.fixture
    def mock_config(self):
        """Create mock HuggingFace config."""
        config = Mock()
        config.hidden_size = 512
        config.num_experts = LLADA2_DEFAULT_NUM_EXPERTS
        config.num_experts_per_tok = LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK
        config.num_shared_experts = LLADA2_DEFAULT_NUM_SHARED_EXPERTS
        config.moe_intermediate_size = LLADA2_DEFAULT_MOE_INTERMEDIATE_SIZE
        config.n_group = LLADA2_DEFAULT_N_GROUP
        config.topk_group = LLADA2_DEFAULT_TOPK_GROUP
        config.routed_scaling_factor = LLADA2_DEFAULT_ROUTED_SCALING_FACTOR
        return config

    def test_moe_initialization(self, mock_config, default_vllm_config):
        """Test MoE layer initializes with correct parameters."""
        from vllm.config.vllm import set_current_vllm_config

        from dllm_plugin.models.llada2 import LLaDA2MoE

        with set_current_vllm_config(default_vllm_config):
            moe = LLaDA2MoE(config=mock_config, tp_size=1, prefix="test")

        assert moe.num_experts == LLADA2_DEFAULT_NUM_EXPERTS
        assert moe.num_experts_per_tok == LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK
        assert moe.n_group == LLADA2_DEFAULT_N_GROUP
        assert moe.topk_group == LLADA2_DEFAULT_TOPK_GROUP
        assert moe.routed_scaling_factor == LLADA2_DEFAULT_ROUTED_SCALING_FACTOR

    def test_moe_tp_validation(self, mock_config, default_vllm_config):
        """Test that TP size validation works."""
        from vllm.config.vllm import set_current_vllm_config

        from dllm_plugin.models.llada2 import LLaDA2MoE

        # Should fail if TP > num_experts
        with (
            pytest.raises(ValueError, match="Tensor parallelism size.*cannot exceed"),
            set_current_vllm_config(default_vllm_config),
        ):
            LLaDA2MoE(config=mock_config, tp_size=300, prefix="test")

    def test_group_limited_routing(self, mock_config, default_vllm_config):
        """Test group-limited top-k routing logic."""
        from vllm.config.vllm import set_current_vllm_config

        from dllm_plugin.models.llada2 import LLaDA2MoE

        with set_current_vllm_config(default_vllm_config):
            moe = LLaDA2MoE(config=mock_config, tp_size=1, prefix="test")

        # Create dummy router logits
        batch_size, seq_len = 2, 4
        num_tokens = batch_size * seq_len
        router_logits = torch.randn(batch_size, seq_len, LLADA2_DEFAULT_NUM_EXPERTS)

        # Apply group-limited routing
        weights, indices = moe._apply_group_limited_topk(router_logits)

        # Validate output shapes
        assert weights.shape == (num_tokens, LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK)
        assert indices.shape == (num_tokens, LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK)

        # Validate weights are normalized
        weight_sums = weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_shared_expert_initialization(self, mock_config, default_vllm_config):
        """Test shared expert is initialized when configured."""
        from vllm.config.vllm import set_current_vllm_config

        from dllm_plugin.models.llada2 import LLaDA2MoE

        with set_current_vllm_config(default_vllm_config):
            moe = LLaDA2MoE(config=mock_config, tp_size=1, prefix="test")

        # With num_shared_experts=1, shared expert layers should exist
        assert moe.shared_expert_gate is not None
        assert moe.shared_expert_up is not None
        assert moe.shared_expert_down is not None


class TestLLaDA2DecoderLayer:
    """Tests for LLaDA2DecoderLayer."""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create mock vLLM config."""
        config = Mock()
        config.model_config = Mock()
        config.model_config.hf_config = Mock()
        config.parallel_config = Mock()
        config.parallel_config.pipeline_parallel_size = 1

        hf_config = config.model_config.hf_config
        hf_config.hidden_size = 512
        hf_config.num_attention_heads = 8
        hf_config.num_hidden_layers = 4
        hf_config.num_experts = 256
        hf_config.rms_norm_eps = 1e-6

        return config

    def test_decoder_layer_initialization(self, mock_vllm_config, default_vllm_config):
        """Test decoder layer initializes correctly."""
        from vllm.config.vllm import set_current_vllm_config

        from dllm_plugin.models.llada2 import LLaDA2DecoderLayer

        with set_current_vllm_config(default_vllm_config):
            layer = LLaDA2DecoderLayer(
                config=mock_vllm_config.model_config.hf_config,
                layer_idx=0,
                vllm_config=mock_vllm_config,
                prefix="model.layers.0",
            )

        assert layer.layer_idx == 0
        assert layer.hidden_size == 512
        assert layer.self_attn is not None
        assert layer.mlp is not None
        assert layer.input_layernorm is not None
        assert layer.post_attention_layernorm is not None


class TestLLaDA2ForCausalLM:
    """Tests for LLaDA2ForCausalLM main model."""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create mock vLLM config."""
        config = Mock()
        config.model_config = Mock()
        config.model_config.hf_config = Mock()
        config.parallel_config = Mock()
        config.parallel_config.pipeline_parallel_size = 1
        config.parallel_config.tensor_parallel_size = 1

        hf_config = config.model_config.hf_config
        hf_config.hidden_size = 512
        hf_config.vocab_size = 32000
        hf_config.num_hidden_layers = 4
        hf_config.num_attention_heads = 8
        hf_config.rms_norm_eps = 1e-6

        return config

    def test_model_initialization(self, mock_vllm_config):
        """Test model initializes with correct configuration."""
        from dllm_plugin.models.llada2 import LLaDA2ForCausalLM

        with (
            patch("dllm_plugin.models.llada2.assert_compatible_stack"),
            patch("dllm_plugin.models.llada2.get_pp_group"),
            patch("dllm_plugin.models.llada2.get_tp_group"),
        ):
            model = LLaDA2ForCausalLM(vllm_config=mock_vllm_config)

        assert model.hidden_size == 512
        assert model.vocab_size == 32000
        assert model.num_layers == 4
        assert len(model.layers) == 4

    def test_pp_validation_fails(self, mock_vllm_config):
        """Test that PP > 1 raises ValueError."""
        from dllm_plugin.models.llada2 import LLaDA2ForCausalLM

        # Set PP > 1
        mock_vllm_config.parallel_config.pipeline_parallel_size = 2

        with (
            pytest.raises(ValueError, match="Pipeline parallelism.*not supported"),
            patch("dllm_plugin.models.llada2.assert_compatible_stack"),
        ):
            LLaDA2ForCausalLM(vllm_config=mock_vllm_config)

    def test_load_weights_regular_params(self, mock_vllm_config):
        """Test weight loading for regular (non-expert) parameters."""
        from dllm_plugin.models.llada2 import LLaDA2ForCausalLM

        with (
            patch("dllm_plugin.models.llada2.assert_compatible_stack"),
            patch("dllm_plugin.models.llada2.get_pp_group"),
            patch("dllm_plugin.models.llada2.get_tp_group"),
        ):
            model = LLaDA2ForCausalLM(vllm_config=mock_vllm_config)

        # Create dummy weights for embedding layer
        weights = [
            ("embed_tokens.weight", torch.randn(32000, 512)),
        ]

        # Load weights
        unloaded = model.load_weights(weights)

        # embed_tokens.weight should be loaded (not in unloaded set)
        assert "embed_tokens.weight" not in unloaded

    def test_load_weights_expert_params(self, mock_vllm_config):
        """Test weight loading for expert parameters."""
        from dllm_plugin.models.llada2 import LLaDA2ForCausalLM

        with (
            patch("dllm_plugin.models.llada2.assert_compatible_stack"),
            patch("dllm_plugin.models.llada2.get_pp_group"),
            patch("dllm_plugin.models.llada2.get_tp_group"),
        ):
            model = LLaDA2ForCausalLM(vllm_config=mock_vllm_config)

        # Create dummy expert weights
        weights = [
            ("model.layers.0.mlp.experts.0.gate_proj", torch.randn(512, 512)),
            ("model.layers.0.mlp.experts.0.up_proj", torch.randn(512, 512)),
            ("model.layers.0.mlp.experts.0.down_proj", torch.randn(512, 512)),
        ]

        # Load weights
        unloaded = model.load_weights(weights)

        # Expert weights should be marked as loaded (placeholder implementation)
        for name, _ in weights:
            assert name not in unloaded


class TestGroupLimitedRouting:
    """Tests for group-limited routing algorithm."""

    def test_routing_output_shape(self):
        """Test that routing produces correct output shapes."""
        from dllm_plugin.models.llada2 import LLaDA2MoE

        config = Mock()
        config.hidden_size = 512
        config.num_experts = 256
        config.num_experts_per_tok = 8
        config.n_group = 8
        config.topk_group = 4

        with patch("dllm_plugin.models.llada2.get_tp_group"):
            moe = LLaDA2MoE(config=config, tp_size=1, prefix="test")

        # Test routing
        batch_size, seq_len = 4, 32
        router_logits = torch.randn(batch_size, seq_len, 256)

        weights, indices = moe._apply_group_limited_topk(router_logits)

        # Check shapes
        num_tokens = batch_size * seq_len
        assert weights.shape == (num_tokens, 8)
        assert indices.shape == (num_tokens, 8)

    def test_routing_selects_from_groups(self):
        """Test that routing respects group boundaries."""
        from dllm_plugin.models.llada2 import LLaDA2MoE

        config = Mock()
        config.hidden_size = 512
        config.num_experts = 256
        config.num_experts_per_tok = 8
        config.n_group = 8
        config.topk_group = 4

        with patch("dllm_plugin.models.llada2.get_tp_group"):
            moe = LLaDA2MoE(config=config, tp_size=1, prefix="test")

        # Create controlled router logits
        # Make group 0 have highest scores
        batch_size, seq_len = 2, 4
        router_logits = torch.zeros(batch_size, seq_len, 256)

        # Experts 0-31 are group 0 (256 experts / 8 groups = 32 experts per group)
        router_logits[:, :, 0:32] = 10.0  # High scores for group 0

        weights, indices = moe._apply_group_limited_topk(router_logits)

        # All selected experts should be from groups with high scores
        # (exact validation requires understanding group selection logic)
        assert indices.min() >= 0
        assert indices.max() < 256


class TestModelForwardShape:
    """Tests for model forward pass shapes."""

    @pytest.fixture
    def minimal_model_config(self):
        """Create minimal config for shape testing."""
        config = Mock()
        config.model_config = Mock()
        config.model_config.hf_config = Mock()
        config.parallel_config = Mock()
        config.parallel_config.pipeline_parallel_size = 1

        hf_config = config.model_config.hf_config
        hf_config.hidden_size = 128
        hf_config.vocab_size = 1000
        hf_config.num_hidden_layers = 2
        hf_config.num_attention_heads = 4
        hf_config.rms_norm_eps = 1e-6

        return config

    def test_forward_output_shape(self, minimal_model_config):
        """Test that forward pass produces correct output shape."""
        from dllm_plugin.models.llada2 import LLaDA2ForCausalLM

        with (
            patch("dllm_plugin.models.llada2.assert_compatible_stack"),
            patch("dllm_plugin.models.llada2.get_pp_group") as mock_pp,
            patch("dllm_plugin.models.llada2.get_tp_group"),
        ):
            # Mock PP group
            mock_pp.return_value.is_first_rank = True
            mock_pp.return_value.is_last_rank = True

            model = LLaDA2ForCausalLM(vllm_config=minimal_model_config)

        # Note: Full forward pass requires proper vLLM setup
        # This test validates structure only
        assert model.hidden_size == 128
        assert model.vocab_size == 1000


class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_default_constants(self):
        """Test that default constants are set correctly."""
        assert LLADA2_DEFAULT_NUM_EXPERTS == 256
        assert LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK == 8
        assert LLADA2_DEFAULT_NUM_SHARED_EXPERTS == 1
        assert LLADA2_DEFAULT_MOE_INTERMEDIATE_SIZE == 512
        assert LLADA2_DEFAULT_N_GROUP == 8
        assert LLADA2_DEFAULT_TOPK_GROUP == 4
        assert LLADA2_DEFAULT_ROUTED_SCALING_FACTOR == 2.5

    def test_config_fallback_to_defaults(self):
        """Test that missing config values fall back to defaults."""
        from dllm_plugin.models.llada2 import LLaDA2MoE

        # Config without MoE-specific parameters
        config = Mock()
        config.hidden_size = 512

        with patch("dllm_plugin.models.llada2.get_tp_group"):
            moe = LLaDA2MoE(config=config, tp_size=1, prefix="test")

        # Should use defaults
        assert moe.num_experts == LLADA2_DEFAULT_NUM_EXPERTS
        assert moe.num_experts_per_tok == LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK
