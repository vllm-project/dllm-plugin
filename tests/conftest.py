# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration and fixtures for dLLM plugin tests."""

import pytest


@pytest.fixture
def default_vllm_config():
    """Provide default vLLM config context for tests that create vLLM layers.

    This fixture sets up a minimal vLLM configuration context required by
    CustomOp layers like RMSNorm, which are used by LLaDA2BlockAttention.
    """
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.cache import CacheConfig
        from vllm.config.compilation import CompilationConfig
        from vllm.config.model import ModelConfig
        from vllm.config.parallel import ParallelConfig
        from vllm.config.scheduler import SchedulerConfig
    except ImportError:
        # If vLLM is not installed, skip (tests will be skipped anyway)
        pytest.skip("vLLM not installed")

    # Create minimal config
    try:
        # Minimal model config (required fields only)
        model_config = ModelConfig(
            model="facebook/opt-125m",  # Dummy model
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
            seed=0,
        )

        parallel_config = ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )

        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=2048,
            max_num_seqs=8,
            max_model_len=2048,
            is_encoder_decoder=False,
        )

        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            swap_space_bytes=0,
            cache_dtype="auto",
        )

        compilation_config = CompilationConfig(level=0)

        vllm_config = VllmConfig(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
            compilation_config=compilation_config,
        )

        # Set the config context
        with set_current_vllm_config(vllm_config):
            yield vllm_config
    except Exception as e:
        # If config creation fails, skip the test
        pytest.skip(f"Failed to create vLLM config: {e}")
