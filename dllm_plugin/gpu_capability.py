# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the dLLM project
"""GPU capability detection for automatic optimization selection.

This module provides hardware detection and capability checks to enable/disable
optimizations based on GPU compute capability. Supports A100, H100, B200 and
gracefully handles unavailable features.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GPUCapabilities:
    """GPU hardware capabilities for optimization selection.

    Attributes:
        compute_capability: CUDA compute capability (major, minor), e.g. (8, 0) for A100
        device_name: GPU model name, e.g. "NVIDIA A100-SXM4-40GB"
        total_memory_gb: Total GPU memory in gigabytes
        device_id: CUDA device ID (default 0)
    """

    compute_capability: tuple[int, int]
    device_name: str
    total_memory_gb: float
    device_id: int = 0

    def supports_flashinfer_fused_topk(self) -> bool:
        """FlashInfer fused topk requires compute capability >= 9.0 (H100+).

        FlashInfer's fused_topk_deepseek kernel requires Hopper architecture
        or newer (compute capability 9.0+). A100 (8.0) will use PyTorch fallback.

        Returns:
            True if H100, B200, or newer; False for A100 or older
        """
        return self.compute_capability[0] >= 9

    def supports_fp8_efficient(self) -> bool:
        """FP8 is efficient on H100+ (9.0), slow on A100 (8.0).

        While A100 technically supports FP8 operations, they are emulated
        and significantly slower than native FP16/BF16. H100+ has hardware
        FP8 Tensor Cores that provide 2x throughput vs FP16.

        Returns:
            True if FP8 will improve performance (H100+); False if emulated (A100)
        """
        return self.compute_capability[0] >= 9

    def supports_cutlass_moe(self) -> bool:
        """CUTLASS MoE kernels work on A100+ (8.0+).

        NVIDIA CUTLASS templates support Ampere (A100) and newer. Older
        architectures (V100, T4) should use Triton kernels as fallback.

        Returns:
            True if A100 or newer (compute capability >= 8.0)
        """
        return self.compute_capability[0] >= 8

    def supports_torch_compile(self) -> bool:
        """torch.compile works on all modern GPUs (compute capability >= 7.0).

        PyTorch 2.0+ graph compilation supports Volta (V100) and newer.
        Turing (T4) and Ampere (A100) see better gains due to improved
        compiler backends.

        Returns:
            True for V100 and newer (compute capability >= 7.0)
        """
        return self.compute_capability[0] >= 7

    def supports_trtllm_moe(self) -> bool:
        """TensorRT-LLM MoE kernels prefer H100+ for FP8 support.

        While TensorRT-LLM works on A100, the FP8-optimized MoE kernels
        provide maximum benefit on H100+ with native FP8 Tensor Cores.

        Returns:
            True if H100 or newer for optimal TensorRT-LLM usage
        """
        return self.compute_capability[0] >= 9

    def recommended_moe_backend(self) -> str:
        """Select optimal MoE kernel backend based on GPU capability.

        Backend selection priority:
        - H100+ (9.0): TensorRT-LLM (best for FP8)
        - A100 (8.0): CUTLASS (best for FP16/BF16)
        - Older: Triton (universal fallback)

        Returns:
            "trtllm", "cutlass", or "triton"
        """
        if self.supports_trtllm_moe():
            return "trtllm"
        elif self.supports_cutlass_moe():
            return "cutlass"
        else:
            return "triton"

    def __str__(self) -> str:
        """Human-readable GPU description."""
        major, minor = self.compute_capability
        return (
            f"{self.device_name} "
            f"(compute {major}.{minor}, {self.total_memory_gb:.1f}GB)"
        )


@lru_cache(maxsize=8)
def detect_gpu_capabilities(device: int = 0) -> GPUCapabilities:
    """Detect GPU capabilities for optimization selection.

    This function queries CUDA device properties and caches results for
    fast repeated access. Use the returned GPUCapabilities object to
    check which optimizations are available.

    Args:
        device: CUDA device ID (default 0 for primary GPU)

    Returns:
        GPUCapabilities object with compute capability, name, and memory

    Raises:
        RuntimeError: If CUDA is not available or device ID is invalid

    Example:
        >>> gpu = detect_gpu_capabilities()
        >>> if gpu.supports_cutlass_moe():
        ...     print(f"Using CUTLASS on {gpu.device_name}")
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required for GPU detection. Install with: uv sync --group dev"
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Ensure CUDA toolkit is installed "
            "and GPU drivers are loaded."
        )

    if device < 0 or device >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid device ID {device}. "
            f"Available devices: 0-{torch.cuda.device_count() - 1}"
        )

    # Query CUDA device properties
    major, minor = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)
    name = props.name
    total_memory_gb = props.total_memory / (1024**3)

    return GPUCapabilities(
        compute_capability=(major, minor),
        device_name=name,
        total_memory_gb=total_memory_gb,
        device_id=device,
    )


def clear_capability_cache() -> None:
    """Clear the GPU capability detection cache.

    Useful for testing or if GPU state changes during runtime.
    In normal usage, caching is desirable to avoid repeated CUDA queries.
    """
    detect_gpu_capabilities.cache_clear()


__all__ = [
    "GPUCapabilities",
    "detect_gpu_capabilities",
    "clear_capability_cache",
]
