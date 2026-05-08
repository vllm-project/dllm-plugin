# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge case tests for virtual batch attention (PR #38 review feedback)."""

from __future__ import annotations

import pytest

# Skip if vLLM not installed (macOS dev environments)
pytest.importorskip("vllm")

import torch

from dllm_plugin.attention.virtual_batches import (
    make_block_attention_virtual_batches,
)
from dllm_plugin.vllm_compat import CommonAttentionMetadata


def test_heterogeneous_prefix_lengths():
    """Test virtual batches with heterogeneous prefix lengths [0, 16, 32, 48].

    Validates that each request's block chunk pages are extracted from the
    correct position in the block table (based on actual prefix length, not max).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_reqs = 4
    block_size = 32
    kv_cache_block_size = 16

    # Heterogeneous prefix lengths
    num_prefix_tokens_per_request = [0, 16, 32, 48]
    max_prefix_tokens = 48
    max_prefix_blocks = 3  # 48 // 16

    # Mock block table: each request has pages allocated for prefix + block
    # Request 0: 0 prefix blocks + 2 block blocks = pages [100, 101]
    # Request 1: 1 prefix block + 2 block blocks = pages [10, 20, 21]
    # Request 2: 2 prefix blocks + 2 block blocks = pages [30, 40, 50, 51]
    # Request 3: 3 prefix blocks + 2 block blocks = pages [60, 70, 80, 90, 91]
    block_table_tensor = torch.tensor(
        [
            [100, 101, -1, -1, -1],  # Req 0: block pages at [0:2]
            [10, 20, 21, -1, -1],  # Req 1: prefix at [0], block at [1:3]
            [30, 40, 50, 51, -1],  # Req 2: prefix at [0:2], block at [2:4]
            [60, 70, 80, 90, 91],  # Req 3: prefix at [0:3], block at [3:5]
        ],
        dtype=torch.int32,
        device=device,
    )

    # Mock metadata
    query_start_loc = torch.tensor(
        [0, 32, 64, 96, 128], dtype=torch.int32, device=device
    )
    slot_mapping = torch.zeros(128, dtype=torch.int64, device=device)

    attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([0, 16, 32, 48], dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=128,
        max_query_len=block_size,
        max_seq_len=max_prefix_tokens,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=False,
    )

    # Create virtual batches
    prefix_metadata, block_metadata = make_block_attention_virtual_batches(
        attn_metadata=attn_metadata,
        num_prefix_tokens_per_request=num_prefix_tokens_per_request,
        block_size=block_size,
        kv_cache_block_size=kv_cache_block_size,
    )

    # Validate prefix metadata
    assert prefix_metadata is not None, "Should have prefix metadata"
    assert prefix_metadata.num_reqs == num_reqs
    assert prefix_metadata.seq_lens.tolist() == [0, 16, 32, 48]

    # Validate prefix block table padding uses -1 sentinel
    prefix_block_table = prefix_metadata.block_table_tensor
    assert prefix_block_table.shape == (num_reqs, max_prefix_blocks)
    # Request 0: no prefix, should be all -1
    assert (prefix_block_table[0] == -1).all(), "Request 0 should have -1 padding"
    # Request 1: 1 prefix block, should be [10, -1, -1]
    assert prefix_block_table[1].tolist() == [10, -1, -1]
    # Request 2: 2 prefix blocks, should be [30, 40, -1]
    assert prefix_block_table[2].tolist() == [30, 40, -1]
    # Request 3: 3 prefix blocks, should be [60, 70, 80]
    assert prefix_block_table[3].tolist() == [60, 70, 80]

    # Validate block metadata
    assert block_metadata.num_reqs == num_reqs
    assert (block_metadata.seq_lens == block_size).all()

    # CRITICAL: Validate per-request block table slicing
    # Each request's block chunk pages should be extracted from correct position
    block_block_table = block_metadata.block_table_tensor
    num_block_blocks = 2  # 32 tokens / 16 tokens per block

    assert block_block_table.shape == (num_reqs, num_block_blocks)

    # Request 0: prefix=0 blocks, block starts at index 0
    assert block_block_table[0].tolist() == [100, 101]

    # Request 1: prefix=1 block, block starts at index 1
    assert block_block_table[1].tolist() == [20, 21]

    # Request 2: prefix=2 blocks, block starts at index 2
    assert block_block_table[2].tolist() == [50, 51]

    # Request 3: prefix=3 blocks, block starts at index 3
    assert block_block_table[3].tolist() == [90, 91]


def test_first_block_all_zero_prefix():
    """Test first block case where all requests have num_prefix_tokens == 0.

    Should return prefix_metadata=None and only block_metadata.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_reqs = 2
    block_size = 32
    kv_cache_block_size = 16

    # All zero prefix
    num_prefix_tokens_per_request = [0, 0]

    # Mock block table (only block chunk pages)
    block_table_tensor = torch.tensor(
        [
            [100, 101],
            [200, 201],
        ],
        dtype=torch.int32,
        device=device,
    )

    query_start_loc = torch.tensor([0, 32, 64], dtype=torch.int32, device=device)
    slot_mapping = torch.zeros(64, dtype=torch.int64, device=device)

    attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([0, 0], dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=64,
        max_query_len=block_size,
        max_seq_len=0,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=False,
    )

    prefix_metadata, block_metadata = make_block_attention_virtual_batches(
        attn_metadata=attn_metadata,
        num_prefix_tokens_per_request=num_prefix_tokens_per_request,
        block_size=block_size,
        kv_cache_block_size=kv_cache_block_size,
    )

    # First block: no prefix chunk
    assert prefix_metadata is None, "First block should have no prefix metadata"

    # Block chunk should exist
    assert block_metadata is not None
    assert block_metadata.num_reqs == num_reqs
    assert (block_metadata.seq_lens == block_size).all()

    # Block table should extract first 2 blocks for each request
    assert block_metadata.block_table_tensor.shape == (num_reqs, 2)
    assert block_metadata.block_table_tensor[0].tolist() == [100, 101]
    assert block_metadata.block_table_tensor[1].tolist() == [200, 201]


def test_padding_sentinel_not_zero():
    """Verify that padding uses -1 sentinel, not 0 (which is a valid page ID).

    Regression test for PR #38 review issue #3.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_reqs = 2
    block_size = 32
    kv_cache_block_size = 16

    # Heterogeneous: request 0 has no prefix, request 1 has 32 tokens
    num_prefix_tokens_per_request = [0, 32]

    # Mock block table
    # Request 0: no prefix, block at [100, 101]
    # Request 1: prefix at [0, 1], block at [2, 3]
    block_table_tensor = torch.tensor(
        [
            [100, 101, -1, -1],
            [0, 1, 2, 3],  # Note: page 0 is VALID!
        ],
        dtype=torch.int32,
        device=device,
    )

    query_start_loc = torch.tensor([0, 32, 64], dtype=torch.int32, device=device)
    slot_mapping = torch.zeros(64, dtype=torch.int64, device=device)

    attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([0, 32], dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=64,
        max_query_len=block_size,
        max_seq_len=32,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=False,
    )

    prefix_metadata, _ = make_block_attention_virtual_batches(
        attn_metadata=attn_metadata,
        num_prefix_tokens_per_request=num_prefix_tokens_per_request,
        block_size=block_size,
        kv_cache_block_size=kv_cache_block_size,
    )

    # Validate padding uses -1, not 0
    assert prefix_metadata is not None, "Should have prefix metadata"
    prefix_block_table = prefix_metadata.block_table_tensor
    max_prefix_blocks = 2  # 32 tokens / 16 = 2 blocks

    assert prefix_block_table.shape == (num_reqs, max_prefix_blocks)

    # Request 0: no prefix, should be [-1, -1] NOT [0, 0]
    assert prefix_block_table[0].tolist() == [-1, -1], (
        "Padding must use -1 sentinel, not 0 (0 is a valid page ID)"
    )

    # Request 1: 2 prefix blocks, should be [0, 1]
    assert prefix_block_table[1].tolist() == [0, 1]


def test_input_validation():
    """Test that mismatched num_prefix_tokens_per_request raises ValueError."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_reqs = 3
    block_size = 32

    # Wrong length: 2 lengths for 3 requests
    num_prefix_tokens_per_request = [0, 16]  # Should be 3 elements

    block_table_tensor = torch.zeros((num_reqs, 5), dtype=torch.int32, device=device)
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * 32
    slot_mapping = torch.zeros(num_reqs * block_size, dtype=torch.int64, device=device)

    attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.zeros(num_reqs, dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs * block_size,
        max_query_len=block_size,
        max_seq_len=0,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=False,
    )

    with pytest.raises(ValueError, match="Expected 3 prefix lengths, got 2"):
        make_block_attention_virtual_batches(
            attn_metadata=attn_metadata,
            num_prefix_tokens_per_request=num_prefix_tokens_per_request,
            block_size=block_size,
        )
