# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Type protocols for vLLM integration.

Provides type-safe interfaces for vLLM objects without requiring vLLM runtime.
All protocols define only the subset of attributes/methods dllm-plugin uses.

These protocols serve as documentation of our vLLM dependencies and enable:
- Type checking without vLLM installed
- IDE autocomplete for vLLM objects
- Clear contract of what vLLM fields we access
- Runtime validation via type guards

**Version compatibility:** Designed for vLLM >=0.20.0,<0.21
"""

from __future__ import annotations

from typing import Any, Protocol, TypeGuard


class HfConfigProtocol(Protocol):
    """Protocol for HuggingFace config attributes accessed by dllm-plugin.

    Represents the subset of transformers.PretrainedConfig that we use.
    Does not include all HF config fields, only those we access.
    """

    architectures: list[str]
    """Model architecture names (e.g., ['LLaDA2ForCausalLM'])."""

    hidden_size: int
    """Hidden dimension size."""

    num_attention_heads: int
    """Number of attention heads."""

    num_hidden_layers: int
    """Number of transformer layers."""

    vocab_size: int
    """Vocabulary size."""

    # Allow dynamic attribute access for optional fields
    # (e.g., num_key_value_heads, num_experts, first_k_dense_replace)
    def __getattribute__(self, name: str) -> Any: ...


class ModelConfigProtocol(Protocol):
    """Protocol for vLLM ModelConfig.

    Represents vllm.config.ModelConfig attributes we access.
    """

    hf_config: HfConfigProtocol
    """HuggingFace transformers config."""


class SchedulerConfigProtocol(Protocol):
    """Protocol for vLLM SchedulerConfig.

    Represents vllm.config.SchedulerConfig attributes we access.
    """

    def get_scheduler_cls(self) -> type:
        """Get scheduler class (vLLM 0.20+ API).

        Returns:
            Scheduler class to instantiate.
        """
        ...


class ParallelConfigProtocol(Protocol):
    """Protocol for vLLM ParallelConfig.

    Represents vllm.config.ParallelConfig attributes we access.
    """

    worker_cls: str
    """Worker class qualname (e.g., 'dllm_plugin.Worker')."""


class CacheConfigProtocol(Protocol):
    """Protocol for vLLM CacheConfig.

    Represents vllm.config.CacheConfig attributes we access.
    """

    block_size: int | None
    """KV cache block size in tokens (typically 16)."""

    gpu_memory_utilization: float
    """Fraction of GPU memory to use (0.0-1.0)."""


class VllmConfigProtocol(Protocol):
    """Protocol for VllmConfig attributes accessed by dllm-plugin.

    Represents the subset of vllm.config.VllmConfig that we use.
    This is the main config object passed to scheduler/worker/model.
    """

    model_config: ModelConfigProtocol
    """Model configuration."""

    scheduler_config: SchedulerConfigProtocol
    """Scheduler configuration."""

    parallel_config: ParallelConfigProtocol
    """Parallelism configuration."""

    cache_config: CacheConfigProtocol | None
    """KV cache configuration (None if not using cache)."""


# Type guards for runtime validation


def is_valid_vllm_config(obj: Any) -> TypeGuard[VllmConfigProtocol]:
    """Runtime check for VllmConfig structure.

    Validates that an object has the minimum required attributes for
    dllm-plugin to function correctly.

    Args:
        obj: Object to validate.

    Returns:
        True if obj conforms to VllmConfigProtocol structure.

    Example:
        >>> from vllm import EngineArgs
        >>> args = EngineArgs(model="facebook/opt-125m")
        >>> vllm_config = args.create_engine_config()
        >>> assert is_valid_vllm_config(vllm_config)
    """
    return (
        hasattr(obj, "model_config")
        and hasattr(obj.model_config, "hf_config")
        and hasattr(obj.model_config.hf_config, "architectures")
        and isinstance(obj.model_config.hf_config.architectures, list)
        and hasattr(obj, "scheduler_config")
        and hasattr(obj, "parallel_config")
    )


def assert_valid_vllm_config(obj: Any, context: str = "") -> None:
    """Assert VllmConfig is valid, raise ValueError if not.

    Convenience wrapper around is_valid_vllm_config for fail-fast validation.

    Args:
        obj: Object to validate.
        context: Optional context string for error messages.

    Raises:
        ValueError: If obj does not conform to VllmConfigProtocol.

    Example:
        >>> from dllm_plugin.vllm_types import assert_valid_vllm_config
        >>> assert_valid_vllm_config(vllm_config, context="scheduler init")
    """
    if not is_valid_vllm_config(obj):
        ctx = f" (context: {context})" if context else ""
        raise ValueError(
            f"Invalid vLLM config structure{ctx}. "
            "Expected vllm_config with model_config.hf_config.architectures, "
            "scheduler_config, and parallel_config. "
            f"Got type: {type(obj).__name__}"
        )


__all__ = [
    "HfConfigProtocol",
    "ModelConfigProtocol",
    "SchedulerConfigProtocol",
    "ParallelConfigProtocol",
    "CacheConfigProtocol",
    "VllmConfigProtocol",
    "is_valid_vllm_config",
    "assert_valid_vllm_config",
]
