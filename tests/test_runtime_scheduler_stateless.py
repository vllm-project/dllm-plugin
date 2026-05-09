# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test stateless scheduler behavior for dLLM (aligned with upstream vLLM)."""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")

from unittest.mock import Mock

from dllm_plugin.runtime_scheduler import DllmRuntimeScheduler


def test_scheduler_stateless_add_request():
    """Validate that scheduler doesn't initialize spec_token_ids in add_request().

    Following upstream vLLM pattern (v1/core/sched/scheduler.py:1741):
    Scheduler does NOT initialize spec_token_ids - that happens in model runner.
    """
    # Mock vllm_config to pass validation
    mock_config = Mock()
    mock_config.model_config.hf_config.architectures = ["LLaDA2ForCausalLM"]
    mock_config.model_config.hf_text_config = None
    mock_config.model_config.max_model_len = 8192
    mock_config.scheduler_config.max_num_seqs = 256
    mock_config.scheduler_config.max_num_batched_tokens = 8192
    mock_config.observability_config.kv_cache_metrics_sample = 1.0
    mock_config.kv_transfer_config = None
    mock_config.kv_events_config = None
    mock_config.ec_transfer_config = None

    # Mock required scheduler parameters
    mock_kv_cache_config = Mock()
    mock_structured_output_manager = Mock()

    scheduler = DllmRuntimeScheduler(
        vllm_config=mock_config,
        kv_cache_config=mock_kv_cache_config,
        structured_output_manager=mock_structured_output_manager,
        block_size=16,
    )

    # Create mock request with empty spec_token_ids (upstream default)
    request = Mock()
    request.request_id = "req-1"
    request.prompt_token_ids = list(range(10))
    request.spec_token_ids = []  # Empty - upstream vLLM default
    request.num_tokens = [10]  # Prefill
    request.is_finished = Mock(return_value=False)
    request.is_prefill_chunk = True

    # Add request - should NOT mutate spec_token_ids
    scheduler.add_request(request)

    # Verify scheduler didn't touch spec_token_ids
    assert request.spec_token_ids == [], (
        "Scheduler violated stateless pattern - modified spec_token_ids"
    )


def test_scheduler_stateless_schedule_first_iteration():
    """Validate that scheduler handles empty spec_token_ids gracefully.

    Following upstream vLLM pattern: first iteration has empty spec_token_ids,
    scheduler should handle this without initializing drafts.
    """
    # Mock vllm_config
    mock_config = Mock()
    mock_config.model_config.hf_config.architectures = ["LLaDA2ForCausalLM"]
    mock_config.model_config.hf_text_config = None
    mock_config.model_config.max_model_len = 8192
    mock_config.scheduler_config.max_num_seqs = 256
    mock_config.scheduler_config.max_num_batched_tokens = 8192
    mock_config.observability_config.kv_cache_metrics_sample = 1.0
    mock_config.kv_transfer_config = None
    mock_config.kv_events_config = None
    mock_config.ec_transfer_config = None

    # Mock required scheduler parameters
    mock_kv_cache_config = Mock()
    mock_structured_output_manager = Mock()

    scheduler = DllmRuntimeScheduler(
        vllm_config=mock_config,
        kv_cache_config=mock_kv_cache_config,
        structured_output_manager=mock_structured_output_manager,
        block_size=16,
    )

    # Create mock request with empty spec_token_ids
    request = Mock()
    request.request_id = "req-1"
    request.prompt_token_ids = list(range(128))
    request.spec_token_ids = []  # First iteration - empty!
    request.num_tokens = [128]
    request.is_finished = Mock(return_value=False)
    request.is_prefill_chunk = False
    request.num_computed_tokens = 128

    scheduler.requests = {"req-1": request}

    # Schedule should handle empty spec_token_ids gracefully
    out = scheduler.schedule()

    # Either empty or absent - both valid for first iteration
    draft = out.scheduled_spec_decode_tokens.get("req-1")
    assert draft is None or len(draft) == 0, (
        f"Expected empty or absent draft for first iteration, got {draft}"
    )


def test_scheduler_stateless_no_cache_infrastructure():
    """Verify that scheduler no longer has cache-related attributes.

    The refactor removes all thread-safe cache infrastructure that was
    causing race conditions at high concurrency.
    """
    mock_config = Mock()
    mock_config.model_config.hf_config.architectures = ["LLaDA2ForCausalLM"]
    mock_config.model_config.hf_text_config = None
    mock_config.model_config.max_model_len = 8192
    mock_config.scheduler_config.max_num_seqs = 256
    mock_config.scheduler_config.max_num_batched_tokens = 8192
    mock_config.observability_config.kv_cache_metrics_sample = 1.0
    mock_config.kv_transfer_config = None
    mock_config.kv_events_config = None
    mock_config.ec_transfer_config = None

    # Mock required scheduler parameters
    mock_kv_cache_config = Mock()
    mock_structured_output_manager = Mock()

    scheduler = DllmRuntimeScheduler(
        vllm_config=mock_config,
        kv_cache_config=mock_kv_cache_config,
        structured_output_manager=mock_structured_output_manager,
        block_size=16,
    )

    # Verify no cache attributes exist
    assert not hasattr(scheduler, "_spec_token_cache"), (
        "Scheduler still has _spec_token_cache - refactor incomplete"
    )
    assert not hasattr(scheduler, "_cache_lock"), (
        "Scheduler still has _cache_lock - refactor incomplete"
    )
