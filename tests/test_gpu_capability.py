# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the dLLM project
"""Unit tests for GPU capability detection.

Tests GPU detection logic and capability checks for A100, H100, B200.
Uses real CUDA queries when available, falls back to mocking for CI/test environments.
"""

from __future__ import annotations

import pytest

# Skip entire module if torch not available (standard CI)
# Required before @patch decorators which reference torch.cuda at collection time
pytest.importorskip("torch")

from unittest.mock import MagicMock, patch  # noqa: E402

from dllm_plugin.gpu_capability import (  # noqa: E402
    GPUCapabilities,
    clear_capability_cache,
    detect_gpu_capabilities,
)


class TestGPUCapabilities:
    """Test GPUCapabilities dataclass methods."""

    def test_a100_capabilities(self) -> None:
        """A100 (8.0) supports CUTLASS but not H100+ features."""
        gpu = GPUCapabilities(
            compute_capability=(8, 0),
            device_name="NVIDIA A100-SXM4-40GB",
            total_memory_gb=40.0,
        )

        assert gpu.supports_cutlass_moe() is True

        assert gpu.supports_flashinfer_fused_topk() is False
        assert gpu.supports_fp8_efficient() is False
        assert gpu.supports_trtllm_moe() is False
        assert gpu.recommended_moe_backend() == "cutlass"

    def test_h100_capabilities(self) -> None:
        """H100 (9.0) supports all A100 features plus FP8 and FlashInfer."""
        gpu = GPUCapabilities(
            compute_capability=(9, 0),
            device_name="NVIDIA H100-SXM5-80GB",
            total_memory_gb=80.0,
        )

        assert gpu.supports_cutlass_moe() is True

        assert gpu.supports_flashinfer_fused_topk() is True
        assert gpu.supports_fp8_efficient() is True
        assert gpu.supports_trtllm_moe() is True
        assert gpu.recommended_moe_backend() == "trtllm"

    def test_b200_capabilities(self) -> None:
        """B200 (10.0) supports all H100 features (future-proofing)."""
        gpu = GPUCapabilities(
            compute_capability=(10, 0),
            device_name="NVIDIA B200-SXM-192GB",
            total_memory_gb=192.0,
        )

        assert gpu.supports_cutlass_moe() is True

        assert gpu.supports_flashinfer_fused_topk() is True
        assert gpu.supports_fp8_efficient() is True
        assert gpu.supports_trtllm_moe() is True
        assert gpu.recommended_moe_backend() == "trtllm"

    def test_v100_capabilities(self) -> None:
        """V100 (7.0) does not support Ampere+ features."""
        gpu = GPUCapabilities(
            compute_capability=(7, 0),
            device_name="Tesla V100-SXM2-16GB",
            total_memory_gb=16.0,
        )

        assert gpu.supports_cutlass_moe() is False
        assert gpu.supports_flashinfer_fused_topk() is False
        assert gpu.supports_fp8_efficient() is False
        assert gpu.supports_trtllm_moe() is False
        assert gpu.recommended_moe_backend() == "triton"

    def test_t4_capabilities(self) -> None:
        """T4 (7.5) does not support Ampere+ features."""
        gpu = GPUCapabilities(
            compute_capability=(7, 5),
            device_name="Tesla T4",
            total_memory_gb=16.0,
        )

        assert gpu.supports_cutlass_moe() is False
        assert gpu.supports_flashinfer_fused_topk() is False
        assert gpu.supports_fp8_efficient() is False
        assert gpu.recommended_moe_backend() == "triton"

    def test_str_representation(self) -> None:
        """Test human-readable GPU description."""
        gpu = GPUCapabilities(
            compute_capability=(8, 0),
            device_name="NVIDIA A100-SXM4-40GB",
            total_memory_gb=40.5,
        )

        assert str(gpu) == "NVIDIA A100-SXM4-40GB (compute 8.0, 40.5GB)"

    def test_dataclass_immutability(self) -> None:
        """GPUCapabilities should be frozen (immutable)."""
        gpu = GPUCapabilities(
            compute_capability=(8, 0),
            device_name="NVIDIA A100-SXM4-40GB",
            total_memory_gb=40.0,
        )

        with pytest.raises(AttributeError):
            gpu.compute_capability = (9, 0)  # type: ignore


class TestDetectGPUCapabilities:
    """Test detect_gpu_capabilities() function."""

    def test_real_gpu_detection(self) -> None:
        """Test detection on real GPU (if CUDA available)."""
        pytest.importorskip("torch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping real GPU test")

        clear_capability_cache()
        gpu = detect_gpu_capabilities(device=0)

        # Basic sanity checks
        assert isinstance(gpu, GPUCapabilities)
        assert gpu.compute_capability[0] >= 7  # At least Volta
        assert gpu.total_memory_gb > 0
        assert "NVIDIA" in gpu.device_name or "Tesla" in gpu.device_name
        assert gpu.device_id == 0

    def test_caching(self) -> None:
        """Test that detect_gpu_capabilities() caches results."""
        pytest.importorskip("torch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping cache test")

        clear_capability_cache()

        # First call
        gpu1 = detect_gpu_capabilities(device=0)

        # Second call should return cached result (same object)
        gpu2 = detect_gpu_capabilities(device=0)

        assert gpu1 is gpu2  # Same object reference (cached)

    def test_cache_clear(self) -> None:
        """Test cache clearing functionality."""
        pytest.importorskip("torch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping cache clear test")

        clear_capability_cache()
        gpu1 = detect_gpu_capabilities(device=0)

        clear_capability_cache()
        gpu2 = detect_gpu_capabilities(device=0)

        # After cache clear, we get a new object (not cached)
        assert gpu1 is not gpu2
        # But with same values
        assert gpu1.compute_capability == gpu2.compute_capability
        assert gpu1.device_name == gpu2.device_name

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_not_available(self, mock_cuda_available: MagicMock) -> None:
        """Test error handling when CUDA is not available."""
        pytest.importorskip("torch")

        clear_capability_cache()

        with pytest.raises(RuntimeError, match="CUDA is not available"):
            detect_gpu_capabilities()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    def test_invalid_device_id(
        self, mock_device_count: MagicMock, mock_cuda_available: MagicMock
    ) -> None:
        """Test error handling for invalid device ID."""
        pytest.importorskip("torch")

        clear_capability_cache()

        with pytest.raises(RuntimeError, match="Invalid device ID 5"):
            detect_gpu_capabilities(device=5)

        with pytest.raises(RuntimeError, match="Invalid device ID -1"):
            detect_gpu_capabilities(device=-1)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_capability", return_value=(8, 0))
    @patch("torch.cuda.get_device_properties")
    def test_mocked_a100_detection(
        self,
        mock_props: MagicMock,
        mock_capability: MagicMock,
        mock_count: MagicMock,
        mock_available: MagicMock,
    ) -> None:
        """Test A100 detection with mocked CUDA."""
        pytest.importorskip("torch")

        # Mock A100 properties
        mock_props.return_value.name = "NVIDIA A100-SXM4-40GB"
        mock_props.return_value.total_memory = 40 * 1024**3  # 40GB in bytes

        clear_capability_cache()
        gpu = detect_gpu_capabilities(device=0)

        assert gpu.compute_capability == (8, 0)
        assert gpu.device_name == "NVIDIA A100-SXM4-40GB"
        assert gpu.total_memory_gb == 40.0
        assert gpu.supports_cutlass_moe() is True
        assert gpu.supports_flashinfer_fused_topk() is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.get_device_capability", return_value=(9, 0))
    @patch("torch.cuda.get_device_properties")
    def test_mocked_h100_detection(
        self,
        mock_props: MagicMock,
        mock_capability: MagicMock,
        mock_count: MagicMock,
        mock_available: MagicMock,
    ) -> None:
        """Test H100 detection with mocked CUDA."""
        pytest.importorskip("torch")

        # Mock H100 properties
        mock_props.return_value.name = "NVIDIA H100-SXM5-80GB"
        mock_props.return_value.total_memory = 80 * 1024**3  # 80GB in bytes

        clear_capability_cache()
        gpu = detect_gpu_capabilities(device=0)

        assert gpu.compute_capability == (9, 0)
        assert gpu.device_name == "NVIDIA H100-SXM5-80GB"
        assert gpu.total_memory_gb == 80.0
        assert gpu.supports_cutlass_moe() is True
        assert gpu.supports_flashinfer_fused_topk() is True
        assert gpu.supports_fp8_efficient() is True
        assert gpu.recommended_moe_backend() == "trtllm"

    def test_no_torch_import_error(self) -> None:
        """Test error handling when PyTorch is not installed."""
        import sys

        # Temporarily hide torch module
        torch_module = sys.modules.get("torch")
        if torch_module:
            sys.modules["torch"] = None  # type: ignore

        try:
            clear_capability_cache()
            with pytest.raises(RuntimeError, match="PyTorch is required"):
                detect_gpu_capabilities()
        finally:
            # Restore torch module
            if torch_module:
                sys.modules["torch"] = torch_module


class TestMoEBackendSelection:
    """Test MoE backend recommendation logic."""

    def test_backend_selection_a100(self) -> None:
        """A100 should recommend CUTLASS."""
        gpu = GPUCapabilities(
            compute_capability=(8, 0),
            device_name="NVIDIA A100-SXM4-40GB",
            total_memory_gb=40.0,
        )
        assert gpu.recommended_moe_backend() == "cutlass"

    def test_backend_selection_h100(self) -> None:
        """H100 should recommend TensorRT-LLM."""
        gpu = GPUCapabilities(
            compute_capability=(9, 0),
            device_name="NVIDIA H100-SXM5-80GB",
            total_memory_gb=80.0,
        )
        assert gpu.recommended_moe_backend() == "trtllm"

    def test_backend_selection_v100(self) -> None:
        """V100 should fall back to Triton."""
        gpu = GPUCapabilities(
            compute_capability=(7, 0),
            device_name="Tesla V100-SXM2-16GB",
            total_memory_gb=16.0,
        )
        assert gpu.recommended_moe_backend() == "triton"
