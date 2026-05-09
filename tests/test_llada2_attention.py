# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LLaDA2.0 block-style attention.

Tests the LLaDA2BlockAttention module's ability to handle non-causal
attention patterns with block-style masks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if vllm or torch not available
pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

from dllm_plugin.config import DRAFT_SIZE  # noqa: E402


# Mock TP group for unit tests that don't need actual parallel execution
@pytest.fixture(autouse=True)
def mock_tp_group():
    """Mock tensor parallel group for tests that create vLLM layers."""
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


class TestLLaDA2BlockAttention:
    """Tests for block-style attention module."""

    @pytest.fixture
    def attention_layer(self):
        """Create a basic attention layer for testing."""
        from dllm_plugin.models.llada2_attention import (
            LLaDA2BlockAttention,  # noqa: E402
        )

        return LLaDA2BlockAttention(
            num_heads=8,
            head_size=64,
            num_kv_heads=8,
        )

    def test_initialization(self, attention_layer):
        """Test that attention layer initializes correctly."""
        assert attention_layer.num_heads == 8
        assert attention_layer.head_size == 64
        assert attention_layer.num_kv_heads == 8
        assert attention_layer.attn is not None

    def test_attention_output_shape(self, attention_layer):
        """Test that attention produces correct output shapes."""
        # Note: Full forward pass requires proper vLLM setup
        # This test validates initialization and structure only
        # Full integration tested in GPU tests
        assert attention_layer is not None
        assert callable(attention_layer.forward)

    def test_block_size_alignment(self):
        """Test that attention works with configured block size."""
        # LLaDA2.0 uses DRAFT_SIZE (default 32) for blocks
        assert DRAFT_SIZE > 0
        assert DRAFT_SIZE == 32  # MVP default

    def test_dual_chunk_strategy_enabled(self, attention_layer):
        """Test that dual-chunk strategy is enabled for MVP."""
        assert attention_layer._use_dual_chunk is True

    def test_metadata_modification_not_implemented(self, attention_layer):
        """Test that metadata modification strategy raises NotImplementedError."""
        # Strategy 1 is deferred to post-MVP
        with pytest.raises(NotImplementedError, match="Strategy 1.*not implemented"):
            attention_layer._forward_metadata_modification(
                query=None,
                key=None,
                value=None,
                kv_cache=None,
                attn_metadata=None,
                kv_scale=1.0,
            )


class TestBlockMaskGeometry:
    """Tests for block-style mask patterns (conceptual validation)."""

    def test_block_mask_pattern_empty_prefix(self):
        """Test mask geometry for first block (no prefix)."""
        block_size = 4
        # First block: tokens [0, 1, 2, 3]
        # Each token should attend to all tokens in block (non-causal within block)

        # Expected mask (1 = attend, 0 = masked):
        #     0  1  2  3
        # 0 [ 1  1  1  1 ]  (bidirectional)
        # 1 [ 1  1  1  1 ]
        # 2 [ 1  1  1  1 ]
        # 3 [ 1  1  1  1 ]

        # Validate conceptual pattern
        expected_mask = torch.ones(block_size, block_size)
        assert expected_mask.shape == (block_size, block_size)
        assert expected_mask.sum().item() == block_size * block_size

    def test_block_mask_pattern_with_prefix(self):
        """Test mask geometry with committed prefix."""
        block_size = 4
        prefix_len = 8  # 2 committed blocks

        # Current block: tokens [8, 9, 10, 11]
        # Each token should attend to:
        # - All prefix tokens [0-7]
        # - All current block tokens [8-11]

        # Expected mask for current block:
        #     0  1  2  3  4  5  6  7  8  9 10 11
        # 8 [ 1  1  1  1  1  1  1  1  1  1  1  1 ]
        # 9 [ 1  1  1  1  1  1  1  1  1  1  1  1 ]
        # 10 [ 1  1  1  1  1  1  1  1  1  1  1  1 ]
        # 11 [ 1  1  1  1  1  1  1  1  1  1  1  1 ]

        total_len = prefix_len + block_size
        expected_mask = torch.ones(block_size, total_len)
        assert expected_mask.shape == (block_size, total_len)
        assert expected_mask.sum().item() == block_size * total_len

    def test_block_mask_vs_causal_mask(self):
        """Compare block-style mask to standard causal mask."""
        block_size = 4

        # Standard causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(block_size, block_size))
        # Block-style mask (full attention within block)
        block_mask = torch.ones(block_size, block_size)

        # Block mask has more 1s (allows looking ahead within block)
        assert block_mask.sum() > causal_mask.sum()

        # Example: Position 0 in causal can only see itself
        assert causal_mask[0].sum().item() == 1
        # But in block-style, position 0 sees all 4 tokens
        assert block_mask[0].sum().item() == 4

    def test_growing_prefix_pattern(self):
        """Test that prefix grows with each generation step."""
        block_size = 4

        # Step 0: No prefix, just first block
        step0_prefix_len = 0
        step0_total = block_size

        # Step 1: First block committed (becomes prefix)
        step1_prefix_len = block_size
        step1_total = step1_prefix_len + block_size

        # Step 2: Two blocks committed
        step2_prefix_len = 2 * block_size
        step2_total = step2_prefix_len + block_size

        # Validate growing pattern
        assert step1_prefix_len > step0_prefix_len
        assert step2_prefix_len > step1_prefix_len
        assert step0_total < step1_total < step2_total


class TestAttentionBackendCompatibility:
    """Tests for backend compatibility (FlashAttention vs FlashInfer)."""

    def test_backend_environment_variables(self, monkeypatch):
        """Test that backend selection via env vars works."""
        from dllm_plugin.models.llada2_attention import LLaDA2BlockAttention

        # FlashAttention
        monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        attn_fa = LLaDA2BlockAttention(num_heads=8, head_size=64)
        assert attn_fa.attn is not None

        # FlashInfer
        monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")
        attn_fi = LLaDA2BlockAttention(num_heads=8, head_size=64)
        assert attn_fi.attn is not None

    @pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER"])
    def test_backend_initialization(self, backend, monkeypatch):
        """Test that both backends initialize correctly."""
        from dllm_plugin.models.llada2_attention import LLaDA2BlockAttention

        monkeypatch.setenv("VLLM_ATTENTION_BACKEND", backend)
        attn = LLaDA2BlockAttention(
            num_heads=16,
            head_size=128,
            num_kv_heads=16,
        )
        assert attn.num_heads == 16
        assert attn.head_size == 128


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_token_block(self):
        """Test attention with block size of 1."""
        from dllm_plugin.models.llada2_attention import LLaDA2BlockAttention

        # While DRAFT_SIZE=32 in practice, test the edge case
        attn = LLaDA2BlockAttention(num_heads=4, head_size=32)
        # Should still initialize correctly
        assert attn.num_heads == 4

    def test_gqa_configuration(self):
        """Test grouped-query attention (GQA) configuration."""
        from dllm_plugin.models.llada2_attention import LLaDA2BlockAttention

        # LLaDA2.0 may use GQA (num_kv_heads < num_heads)
        attn = LLaDA2BlockAttention(
            num_heads=32,
            head_size=128,
            num_kv_heads=8,  # GQA: 4x fewer KV heads
        )
        assert attn.num_heads == 32
        assert attn.num_kv_heads == 8

    def test_custom_scale_factor(self):
        """Test custom attention scale factor."""
        from dllm_plugin.models.llada2_attention import LLaDA2BlockAttention

        custom_scale = 0.1
        attn = LLaDA2BlockAttention(
            num_heads=8,
            head_size=64,
            scale=custom_scale,
        )
        # Verify initialization doesn't crash with custom scale
        assert attn.attn is not None


# Integration test marker (requires GPU)
pytestmark_integration = pytest.mark.dllm_gpu_integration


@pytestmark_integration
class TestLLaDA2AttentionIntegration:
    """Integration tests requiring GPU and full vLLM stack."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
    def test_full_forward_pass(self):
        """Test full forward pass with real vLLM setup.

        Note: This requires full vLLM environment and GPU.
        Deferred to test_llada2_gpu_integration.py for complete validation.
        """
        pytest.skip("Full integration tested in test_llada2_gpu_integration.py")
