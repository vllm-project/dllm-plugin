# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model runner first block generation for dLLM (aligned with upstream vLLM)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("vllm")
pytest.importorskip("torch")

from unittest.mock import Mock

import torch

from dllm_plugin.config import DRAFT_SIZE
from dllm_plugin.gpu_model_runner import DllmGPUModelRunner


def test_model_runner_detects_empty_drafts():
    """Validate model runner detects empty drafts and generates first block.

    Following upstream pattern (vllm/v1/worker/gpu/model_runner.py:1252):
    Draft generation happens in model runner, not scheduler.
    """
    # Skip if first-block generation is disabled (test-only env var)
    if os.environ.get("VLLM_DLLM_SKIP_FIRST_BLOCK_SEED") == "1":
        pytest.skip(
            "First block generation disabled via VLLM_DLLM_SKIP_FIRST_BLOCK_SEED"
        )

    # Mock vllm_config
    mock_config = Mock()
    mock_config.model_config.hf_config.architectures = ["LLaDA2ForCausalLM"]
    mock_config.model_config.hf_text_config = None
    mock_config.model_config.max_model_len = 8192
    mock_config.speculative_config = None
    mock_config.parallel_config.pipeline_parallel_size = 1
    mock_config.parallel_config.tensor_parallel_size = 1

    device = torch.device("cpu")
    runner = DllmGPUModelRunner(vllm_config=mock_config, device=device)

    # Mock req_states.requests to provide prompt_token_ids
    mock_request = Mock()
    mock_request.prompt_token_ids = list(range(128))  # 128-token prompt
    runner.req_states = Mock()
    runner.req_states.requests = {"req-1": mock_request}

    # Mock scheduler output with empty drafts (first iteration)
    scheduler_output = Mock()
    scheduler_output.scheduled_spec_decode_tokens = {"req-1": []}  # Empty!

    # Call before_execute_model - should detect empty draft and generate first block
    runner.before_execute_model(scheduler_output, dummy_run=False)

    # Verify first block generated
    assert "req-1" in runner._dllm_first_block_requests, (
        "Model runner failed to detect empty draft for req-1"
    )
    first_block = runner._dllm_first_block_requests["req-1"]
    assert len(first_block) == DRAFT_SIZE, (
        f"First block has wrong length: expected {DRAFT_SIZE}, got {len(first_block)}"
    )


def test_model_runner_skips_generation_for_existing_drafts():
    """Validate model runner doesn't generate first block when draft exists.

    Subsequent iterations (N>0) have scheduled_spec_decode_tokens populated
    from prior iteration's update_draft_token_ids call.
    """
    # Mock vllm_config
    mock_config = Mock()
    mock_config.model_config.hf_config.architectures = ["LLaDA2ForCausalLM"]
    mock_config.model_config.hf_text_config = None
    mock_config.model_config.max_model_len = 8192
    mock_config.speculative_config = None
    mock_config.parallel_config.pipeline_parallel_size = 1
    mock_config.parallel_config.tensor_parallel_size = 1

    device = torch.device("cpu")
    runner = DllmGPUModelRunner(vllm_config=mock_config, device=device)

    # Mock scheduler output with existing draft (subsequent iteration)
    existing_draft = list(range(DRAFT_SIZE))  # Non-empty draft from prior iteration
    scheduler_output = Mock()
    scheduler_output.scheduled_spec_decode_tokens = {"req-1": existing_draft}

    # Call before_execute_model - should NOT generate first block
    runner.before_execute_model(scheduler_output, dummy_run=False)

    # Verify NO first block generated
    assert "req-1" not in runner._dllm_first_block_requests, (
        "Model runner incorrectly generated first block for existing draft"
    )


def test_model_runner_first_block_deterministic():
    """Validate that first blocks are deterministic from prompt hash.

    This is a key dLLM property - first block is deterministic, unlike
    Eagle/MEDUSA which use hidden states.
    """
    if os.environ.get("VLLM_DLLM_SKIP_FIRST_BLOCK_SEED") == "1":
        pytest.skip(
            "First block generation disabled via VLLM_DLLM_SKIP_FIRST_BLOCK_SEED"
        )

    # Mock vllm_config
    mock_config = Mock()
    mock_config.model_config.hf_config.architectures = ["LLaDA2ForCausalLM"]
    mock_config.model_config.hf_text_config = None
    mock_config.model_config.max_model_len = 8192
    mock_config.speculative_config = None
    mock_config.parallel_config.pipeline_parallel_size = 1
    mock_config.parallel_config.tensor_parallel_size = 1

    device = torch.device("cpu")
    runner = DllmGPUModelRunner(vllm_config=mock_config, device=device)

    # Same prompt should produce same first block
    prompt = list(range(100))
    mock_request = Mock()
    mock_request.prompt_token_ids = prompt
    runner.req_states = Mock()
    runner.req_states.requests = {"req-1": mock_request}

    scheduler_output = Mock()
    scheduler_output.scheduled_spec_decode_tokens = {"req-1": []}

    # Generate first block twice
    runner.before_execute_model(scheduler_output, dummy_run=False)
    first_block_1 = runner._dllm_first_block_requests["req-1"]

    # Reset and generate again
    runner._dllm_first_block_requests = {}
    runner.before_execute_model(scheduler_output, dummy_run=False)
    first_block_2 = runner._dllm_first_block_requests["req-1"]

    # Should be identical (deterministic)
    assert first_block_1 == first_block_2, (
        "First block generation is not deterministic for same prompt"
    )
