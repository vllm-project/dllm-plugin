# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for virtual batch multi-request limitation enforcement.

Phase 7 MVP only supports single-request batching (max_num_seqs=1) due to
complexity of handling heterogeneous prefix lengths across multiple requests.
These tests validate that the limitation is enforced with clear error messages.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

from dllm_plugin.attention.virtual_batches import (  # noqa: E402
    make_block_attention_virtual_batches,
)
from dllm_plugin.vllm_compat import CommonAttentionMetadata  # noqa: E402


def test_virtual_batch_multi_request_fails():
    """Validate that num_reqs > 1 raises NotImplementedError.

    Phase 7 MVP limitation: Virtual batch attention with multiple requests
    requires per-request metadata transformation to handle heterogeneous
    prefix lengths. This is deferred to Phase 7.1.

    This test ensures the limitation is enforced at runtime with a clear
    error message directing operators to use max_num_seqs=1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create metadata with 2 concurrent requests
    attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 32, 64], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 32, 64], dtype=torch.int32),
        seq_lens=torch.tensor([32, 32], dtype=torch.int32, device=device),
        num_reqs=2,  # Multiple requests - should fail
        num_actual_tokens=64,
        max_query_len=32,
        max_seq_len=32,
        block_table_tensor=torch.zeros((2, 4), dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(64, dtype=torch.long, device=device),
        causal=False,
    )

    # Should raise NotImplementedError with clear message about limitation
    with pytest.raises(
        NotImplementedError,
        match="LLaDA2.0 virtual batch attention does not support multi-request",
    ):
        make_block_attention_virtual_batches(
            attn_metadata=attn_metadata,
            num_prefix_tokens=16,
            block_size=32,
        )


def test_virtual_batch_single_request_succeeds():
    """Validate that num_reqs == 1 works correctly (baseline).

    This test ensures that single-request batching (the supported MVP path)
    continues to work as expected. It serves as a regression check and
    demonstrates the correct usage pattern for Phase 7.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create metadata with single request
    attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 32], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 32], dtype=torch.int32),
        seq_lens=torch.tensor([32], dtype=torch.int32, device=device),
        num_reqs=1,  # Single request - should succeed
        num_actual_tokens=32,
        max_query_len=32,
        max_seq_len=32,
        block_table_tensor=torch.zeros((1, 4), dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(32, dtype=torch.long, device=device),
        causal=False,
    )

    # Should succeed and return both virtual batch metadata objects
    prefix_metadata, block_metadata = make_block_attention_virtual_batches(
        attn_metadata=attn_metadata,
        num_prefix_tokens=16,
        block_size=32,
    )

    # Verify both chunks created successfully
    assert prefix_metadata is not None, "Prefix chunk metadata should exist"
    assert block_metadata is not None, "Block chunk metadata should exist"

    # Verify metadata structure
    assert prefix_metadata.num_reqs == 1
    assert block_metadata.num_reqs == 1
    assert prefix_metadata.max_seq_len == 16  # Prefix length
    assert block_metadata.max_seq_len == 32  # Block length


def test_virtual_batch_zero_prefix_single_request():
    """Test edge case: first block (no prefix) with single request.

    When num_prefix_tokens=0 (first generation block), only block self-attention
    is needed. This test validates the edge case handling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 32], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 32], dtype=torch.int32),
        seq_lens=torch.tensor([32], dtype=torch.int32, device=device),
        num_reqs=1,
        num_actual_tokens=32,
        max_query_len=32,
        max_seq_len=32,
        block_table_tensor=torch.zeros((1, 4), dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(32, dtype=torch.long, device=device),
        causal=False,
    )

    # First block: no prefix
    prefix_metadata, block_metadata = make_block_attention_virtual_batches(
        attn_metadata=attn_metadata,
        num_prefix_tokens=0,  # No prefix tokens yet
        block_size=32,
    )

    # Should return None for prefix (no prefix chunk needed)
    assert prefix_metadata is None, "No prefix chunk for first block"
    assert block_metadata is not None, "Block chunk always needed"
    assert block_metadata.num_reqs == 1
    assert block_metadata.max_seq_len == 32
