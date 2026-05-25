# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for numerical validation utilities (validation_utils.py).

These tests validate the tolerance framework itself, ensuring that tolerance
bounds are calculated correctly and comparison functions work as expected.
"""

from __future__ import annotations

import pytest

# Skip if torch not available
pytest.importorskip("torch")

import torch  # noqa: E402

from dllm_plugin.validation_utils import (  # noqa: E402
    TOLERANCE_ACCUMULATED,
    TOLERANCE_BF16,
    TOLERANCE_BF16_LOOSE,
    TOLERANCE_FP32,
    ToleranceBounds,
    assert_numerical_stability,
    assert_output_shape_matches,
    assert_tensors_close,
    compute_kl_divergence,
    expert_selection_agreement,
    get_tolerance_for_dtype,
    get_tolerance_for_layer,
)


class TestToleranceBounds:
    """Test ToleranceBounds dataclass."""

    def test_tolerance_bounds_immutable(self):
        """Test that ToleranceBounds is immutable (frozen)."""
        bounds = ToleranceBounds(atol=1e-5, rtol=1e-4, description="test")
        with pytest.raises(AttributeError):
            bounds.atol = 1e-6  # type: ignore[misc]

    def test_tolerance_presets_exist(self):
        """Test that all tolerance presets are defined."""
        assert TOLERANCE_FP32.atol == 1e-5
        assert TOLERANCE_FP32.rtol == 1e-4

        assert TOLERANCE_BF16.atol == 1e-3
        assert TOLERANCE_BF16.rtol == 1e-2

        assert TOLERANCE_BF16_LOOSE.atol == 1e-2
        assert TOLERANCE_BF16_LOOSE.rtol == 1e-2

        assert TOLERANCE_ACCUMULATED.atol == 5e-2
        assert TOLERANCE_ACCUMULATED.rtol == 5e-2


class TestGetToleranceForDtype:
    """Test get_tolerance_for_dtype() function."""

    def test_fp32_tolerance(self):
        """Test FP32 tolerance bounds."""
        bounds = get_tolerance_for_dtype(torch.float32)
        assert bounds.atol == 1e-5
        assert bounds.rtol == 1e-4
        assert "FP32" in bounds.description

    def test_bf16_tolerance(self):
        """Test BF16 tolerance bounds."""
        bounds = get_tolerance_for_dtype(torch.bfloat16)
        assert bounds.atol == 1e-3
        assert bounds.rtol == 1e-2
        assert "BF16" in bounds.description

    def test_fp16_tolerance(self):
        """Test FP16 tolerance bounds (same as BF16)."""
        bounds = get_tolerance_for_dtype(torch.float16)
        assert bounds.atol == 1e-3
        assert bounds.rtol == 1e-2

    def test_fp64_tolerance(self):
        """Test FP64 tolerance bounds."""
        bounds = get_tolerance_for_dtype(torch.float64)
        assert bounds.atol == 1e-6
        assert bounds.rtol == 1e-5
        assert "FP64" in bounds.description

    def test_unknown_dtype_raises(self):
        """Test that unknown dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dtype"):
            get_tolerance_for_dtype(torch.int32)


class TestGetToleranceForLayer:
    """Test get_tolerance_for_layer() function."""

    def test_embedding_exact_match(self):
        """Test embedding layer expects exact match."""
        bounds = get_tolerance_for_layer("embedding", torch.bfloat16)
        assert bounds.atol == 0.0
        assert bounds.rtol == 0.0
        assert "exact match" in bounds.description.lower()

    def test_rms_norm_fp32(self):
        """Test RMSNorm uses FP32 tolerance."""
        bounds = get_tolerance_for_layer("rms_norm", torch.bfloat16)
        assert bounds == TOLERANCE_FP32

    def test_attention_bf16(self):
        """Test attention uses BF16 tolerance."""
        bounds = get_tolerance_for_layer("attention", torch.bfloat16)
        assert bounds.atol == 1e-3
        assert bounds.rtol == 1e-2

    def test_moe_router_fp32(self):
        """Test MoE router defaults to FP32."""
        bounds = get_tolerance_for_layer("moe_router", torch.bfloat16)
        assert bounds == TOLERANCE_FP32

    def test_decoder_layer_accumulated_error(self):
        """Test decoder layer has 2x accumulated error."""
        base_bounds = get_tolerance_for_dtype(torch.bfloat16)
        bounds = get_tolerance_for_layer("decoder_layer", torch.bfloat16, layer_depth=5)
        assert bounds.atol == base_bounds.atol * 2
        assert bounds.rtol == base_bounds.rtol * 2
        assert "5" in bounds.description  # Layer depth in description

    def test_transformer_stack_accumulated_error(self):
        """Test transformer stack accumulated error scales with depth."""
        base_bounds = get_tolerance_for_dtype(torch.bfloat16)
        bounds = get_tolerance_for_layer(
            "transformer_stack", torch.bfloat16, layer_depth=24
        )
        # 24 layers // 4 = 6x accumulation factor
        assert bounds.atol == base_bounds.atol * 6
        assert bounds.rtol == base_bounds.rtol * 6

    def test_lm_head_loose_tolerance(self):
        """Test LM head uses loose BF16 tolerance."""
        bounds = get_tolerance_for_layer("lm_head", torch.bfloat16)
        assert bounds == TOLERANCE_BF16_LOOSE


class TestAssertTensorsClose:
    """Test assert_tensors_close() function."""

    def test_exact_match(self):
        """Test exact match passes."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        metrics = assert_tensors_close(a, b, atol=1e-5, rtol=1e-4, name="test")
        assert metrics["passed"]
        assert metrics["max_abs_diff"] == 0.0

    def test_within_tolerance(self):
        """Test within tolerance passes."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.000001, 2.000001, 3.000001])
        metrics = assert_tensors_close(a, b, atol=1e-5, rtol=1e-4, name="test")
        assert metrics["passed"]
        assert metrics["max_abs_diff"] < 1e-5

    def test_outside_tolerance_raises(self):
        """Test outside tolerance raises AssertionError."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.1, 2.1, 3.1])
        with pytest.raises(AssertionError, match="mismatch"):
            assert_tensors_close(a, b, atol=1e-5, rtol=1e-4, name="test")

    def test_tolerance_bounds_usage(self):
        """Test using ToleranceBounds object."""
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = torch.tensor([1.00001, 2.00001, 3.00001], dtype=torch.float32)
        tolerance = TOLERANCE_FP32
        metrics = assert_tensors_close(a, b, tolerance=tolerance, name="test")
        assert metrics["passed"]
        assert metrics["atol"] == 1e-5
        assert metrics["rtol"] == 1e-4

    def test_dtype_auto_detection(self):
        """Test automatic dtype-based tolerance."""
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        b = torch.tensor([1.001, 2.001, 3.001], dtype=torch.bfloat16)
        # Without explicit tolerance, should use BF16 tolerance
        metrics = assert_tensors_close(a, b, name="test")
        assert metrics["passed"]
        assert metrics["atol"] == 1e-3  # BF16 tolerance

    def test_metrics_computation(self):
        """Test that metrics are computed correctly."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.01, 2.01, 3.01])
        with pytest.raises(AssertionError):
            assert_tensors_close(a, b, atol=1e-5, rtol=1e-4, name="test")

        # Should still compute metrics even if it fails
        import contextlib

        with contextlib.suppress(AssertionError):
            assert_tensors_close(a, b, atol=1e-5, rtol=1e-4, name="test")


class TestAssertOutputShapeMatches:
    """Test assert_output_shape_matches() function."""

    def test_shape_match(self):
        """Test matching shapes pass."""
        tensor = torch.randn(2, 3, 4)
        assert_output_shape_matches(tensor, (2, 3, 4), name="test")

    def test_shape_mismatch_raises(self):
        """Test mismatched shapes raise AssertionError."""
        tensor = torch.randn(2, 3, 4)
        with pytest.raises(AssertionError, match="shape mismatch"):
            assert_output_shape_matches(tensor, (2, 4, 4), name="test")


class TestAssertNumericalStability:
    """Test assert_numerical_stability() function."""

    def test_valid_tensor(self):
        """Test valid tensor passes."""
        tensor = torch.randn(10, 10)
        assert_numerical_stability(tensor, name="test")

    def test_nan_raises(self):
        """Test NaN raises AssertionError."""
        tensor = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(AssertionError, match="contains NaN"):
            assert_numerical_stability(tensor, name="test")

    def test_inf_raises(self):
        """Test Inf raises AssertionError."""
        tensor = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(AssertionError, match="contains Inf"):
            assert_numerical_stability(tensor, name="test")


class TestComputeKLDivergence:
    """Test compute_kl_divergence() function."""

    def test_identical_distributions(self):
        """Test KL divergence is 0 for identical distributions."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.25, 0.25, 0.25, 0.25])
        kl = compute_kl_divergence(p, q)
        assert abs(kl) < 1e-6

    def test_different_distributions(self):
        """Test KL divergence is positive for different distributions."""
        p = torch.tensor([0.5, 0.3, 0.2])
        q = torch.tensor([0.3, 0.4, 0.3])
        kl = compute_kl_divergence(p, q)
        assert kl > 0

    def test_epsilon_clamping(self):
        """Test epsilon clamping prevents log(0)."""
        p = torch.tensor([0.0, 0.5, 0.5])
        q = torch.tensor([0.3, 0.4, 0.3])
        # Should not raise due to epsilon clamping
        kl = compute_kl_divergence(p, q, eps=1e-10)
        assert not torch.isnan(torch.tensor(kl))


class TestExpertSelectionAgreement:
    """Test expert_selection_agreement() function."""

    def test_perfect_agreement(self):
        """Test 100% agreement for identical selections."""
        experts_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        experts_b = torch.tensor([[1, 2, 3], [4, 5, 6]])
        agreement = expert_selection_agreement(experts_a, experts_b)
        assert agreement == 1.0

    def test_zero_agreement(self):
        """Test 0% agreement for completely different selections."""
        experts_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        experts_b = torch.tensor([[7, 8, 9], [10, 11, 12]])
        agreement = expert_selection_agreement(experts_a, experts_b)
        assert agreement == 0.0

    def test_partial_agreement(self):
        """Test partial agreement."""
        experts_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        experts_b = torch.tensor([[1, 2, 3], [7, 8, 9]])
        agreement = expert_selection_agreement(experts_a, experts_b)
        assert agreement == 0.5  # 1 out of 2 match

    def test_order_independent(self):
        """Test that order doesn't matter for expert selection."""
        experts_a = torch.tensor([[3, 2, 1]])  # Different order
        experts_b = torch.tensor([[1, 2, 3]])  # Sorted
        agreement = expert_selection_agreement(experts_a, experts_b)
        assert agreement == 1.0  # Should match after sorting
