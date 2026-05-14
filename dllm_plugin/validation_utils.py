# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical validation utilities for Phase 9.1 testing.

This module provides tolerance bounds and comparison utilities for validating
numerical correctness of the LLaDA2.0 implementation against reference models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ToleranceBounds:
    """Tolerance bounds for numerical validation.

    Attributes:
        atol: Absolute tolerance for torch.allclose()
        rtol: Relative tolerance for torch.allclose()
        description: Human-readable explanation of why these bounds are appropriate
    """

    atol: float  # Absolute tolerance
    rtol: float  # Relative tolerance
    description: str  # Why these bounds


# Dtype-based tolerance presets
TOLERANCE_FP32 = ToleranceBounds(
    atol=1e-5, rtol=1e-4, description="FP32 computations (RMSNorm, router gate)"
)

TOLERANCE_BF16 = ToleranceBounds(
    atol=1e-3, rtol=1e-2, description="BF16 linear layers (QKV, attention, MLP)"
)

TOLERANCE_BF16_LOOSE = ToleranceBounds(
    atol=1e-2, rtol=1e-2, description="BF16 with potential accumulated error"
)

TOLERANCE_ACCUMULATED = ToleranceBounds(
    atol=5e-2, rtol=5e-2, description="Deep layer accumulated error (24+ layers)"
)


def get_tolerance_for_dtype(dtype: torch.dtype) -> ToleranceBounds:
    """Return tolerance bounds based on compute dtype.

    Args:
        dtype: PyTorch dtype (float32, float16, bfloat16, etc.)

    Returns:
        ToleranceBounds with appropriate atol/rtol for the dtype

    Raises:
        ValueError: If dtype is not recognized
    """
    if dtype in (torch.float32, torch.float):
        return TOLERANCE_FP32
    elif dtype in (torch.float16, torch.half) or dtype == torch.bfloat16:
        return TOLERANCE_BF16
    elif dtype == torch.float64:
        return ToleranceBounds(atol=1e-6, rtol=1e-5, description="FP64")
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_tolerance_for_layer(
    layer_type: str, dtype: torch.dtype, layer_depth: int = 0
) -> ToleranceBounds:
    """Return tolerance bounds based on layer type and depth.

    Different layer types have different numerical precision characteristics:
    - Embeddings: Exact match (deterministic lookup)
    - RMSNorm: FP32 precision
    - Attention: BF16 with FlashAttention
    - MoE router: FP32 (default) or BF16 (experimental)
    - Decoder layers: Accumulated error from sub-components
    - Transformer stack: Linear error accumulation over layers

    Args:
        layer_type: Component type (embedding, attention, moe, decoder, etc.)
        dtype: Compute dtype
        layer_depth: Layer index (0-indexed), used for accumulated error estimation

    Returns:
        ToleranceBounds with appropriate atol/rtol
    """
    base_tolerance = get_tolerance_for_dtype(dtype)

    if layer_type == "embedding":
        # Embeddings are deterministic lookups
        return ToleranceBounds(
            atol=0.0, rtol=0.0, description="Embedding lookup (exact match)"
        )

    elif layer_type == "rms_norm":
        # RMSNorm typically in FP32
        return TOLERANCE_FP32

    elif layer_type == "attention":
        # Attention with FlashAttention
        return base_tolerance

    elif layer_type == "moe_router":
        # Router precision depends on mode (FP32 default, BF16 experimental)
        return TOLERANCE_FP32  # Default FP32

    elif layer_type == "moe_experts":
        # Expert computation in BF16
        return TOLERANCE_BF16

    elif layer_type == "decoder_layer":
        # Single decoder layer (attention + MoE + residuals)
        # Accumulate errors from sub-components
        return ToleranceBounds(
            atol=base_tolerance.atol * 2,
            rtol=base_tolerance.rtol * 2,
            description=f"Decoder layer {layer_depth} (residual accumulation)",
        )

    elif layer_type == "transformer_stack":
        # Full stack - estimate accumulated error
        # Assume linear accumulation (conservative estimate)
        num_layers = layer_depth
        accumulation_factor = max(2, num_layers // 4)
        return ToleranceBounds(
            atol=base_tolerance.atol * accumulation_factor,
            rtol=base_tolerance.rtol * accumulation_factor,
            description=f"Transformer stack (layers={num_layers})",
        )

    elif layer_type == "lm_head":
        # Final projection after accumulated error
        return TOLERANCE_BF16_LOOSE

    else:
        # Default: use dtype-based tolerance
        return base_tolerance


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tolerance: ToleranceBounds | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    name: str = "tensor",
) -> dict:
    """Assert tensors are close within tolerance bounds.

    Computes detailed comparison metrics and raises AssertionError if tensors
    don't match within tolerance. Returns metrics dict on success.

    Args:
        actual: Actual tensor from implementation under test
        expected: Expected tensor from reference implementation
        tolerance: ToleranceBounds object (overrides atol/rtol if provided)
        atol: Absolute tolerance (manual override)
        rtol: Relative tolerance (manual override)
        name: Tensor name for error messages

    Returns:
        Dictionary with comparison metrics:
            - max_abs_diff: Maximum absolute difference
            - mean_abs_diff: Mean absolute difference
            - max_rel_diff: Maximum relative difference
            - atol: Absolute tolerance used
            - rtol: Relative tolerance used
            - passed: Whether comparison passed

    Raises:
        AssertionError: If tensors don't match within tolerance
    """
    # Determine tolerance
    if tolerance is not None:
        atol = atol or tolerance.atol
        rtol = rtol or tolerance.rtol
    elif atol is None or rtol is None:
        # Default to dtype-based tolerance
        tolerance = get_tolerance_for_dtype(actual.dtype)
        atol = atol or tolerance.atol
        rtol = rtol or tolerance.rtol

    # Compute metrics
    abs_diff = torch.abs(actual - expected)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    # Relative difference (avoid division by zero)
    rel_diff = abs_diff / (torch.abs(expected) + 1e-10)
    max_rel_diff = rel_diff.max().item()

    # Check tolerance
    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)

    metrics = {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
    }

    if not passed:
        raise AssertionError(
            f"{name} mismatch:\n"
            f"  Max absolute diff: {max_abs_diff:.2e} (atol={atol:.2e})\n"
            f"  Max relative diff: {max_rel_diff:.2e} (rtol={rtol:.2e})\n"
            f"  Mean absolute diff: {mean_abs_diff:.2e}\n"
            f"  Shape: {actual.shape}\n"
            f"  Dtype: {actual.dtype}"
        )

    return metrics


def assert_output_shape_matches(
    output: torch.Tensor,
    expected_shape: tuple[int, ...],
    name: str = "output",
) -> None:
    """Assert output shape matches expected.

    Args:
        output: Output tensor to validate
        expected_shape: Expected shape tuple
        name: Tensor name for error messages

    Raises:
        AssertionError: If shapes don't match
    """
    assert output.shape == expected_shape, (
        f"{name} shape mismatch: got {output.shape}, expected {expected_shape}"
    )


def assert_numerical_stability(
    tensor: torch.Tensor,
    name: str = "tensor",
) -> None:
    """Assert tensor has no NaN/Inf values.

    Args:
        tensor: Tensor to validate
        name: Tensor name for error messages

    Raises:
        AssertionError: If tensor contains NaN or Inf
    """
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


def compute_kl_divergence(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10
) -> float:
    """Compute KL divergence between two probability distributions.

    Used for comparing router distributions (e.g., FP32 vs BF16 router precision).
    KL divergence measures how one probability distribution differs from another.

    Args:
        p: Reference distribution (e.g., FP32 router output)
        q: Test distribution (e.g., BF16 router output)
        eps: Small epsilon to avoid log(0)

    Returns:
        KL divergence: D_KL(p || q)

    Note:
        Assumes p and q are already normalized probability distributions.
        Clamps to eps to avoid numerical issues with log(0).
    """
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return (p * torch.log(p / q)).sum().item()


def expert_selection_agreement(
    selected_experts_a: torch.Tensor,
    selected_experts_b: torch.Tensor,
) -> float:
    """Compute percentage of tokens with identical expert selection.

    Compares two sets of expert selections and returns the fraction of tokens
    where the exact same set of experts was selected (order-independent).

    Args:
        selected_experts_a: (N, k) expert indices from implementation A
        selected_experts_b: (N, k) expert indices from implementation B
            where N = num_tokens, k = num_experts_per_tok

    Returns:
        Agreement percentage [0.0, 1.0]
        1.0 = all tokens selected identical experts
        0.0 = no tokens selected identical experts

    Note:
        Expert order doesn't matter for MoE, so we sort before comparing.
    """
    # Sort expert indices per token (order doesn't matter for MoE)
    experts_a_sorted, _ = torch.sort(selected_experts_a, dim=1)
    experts_b_sorted, _ = torch.sort(selected_experts_b, dim=1)

    # Check exact match per token
    matches = (experts_a_sorted == experts_b_sorted).all(dim=1)
    return matches.float().mean().item()
