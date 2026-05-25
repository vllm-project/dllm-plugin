"""Unit tests for concatenated virtual batch implementation.

Tests the fix for softmax normalization bug without requiring GPU.
Validates that metadata is constructed correctly for single-batch attention.
"""

import pytest

torch = pytest.importorskip("torch")


def test_concatenated_metadata_construction():
    """Test that concatenated metadata combines prefix + block correctly."""
    pytest.importorskip("vllm")

    from vllm.v1.attention.backend import CommonAttentionMetadata

    from dllm_plugin.attention.concatenated_virtual_batch import (
        create_concatenated_virtual_batch,
    )

    # Setup: 2 requests with different prefix lengths
    num_reqs = 2
    num_prefix_tokens_per_request = [64, 32]  # Heterogeneous prefixes
    block_size = 32
    kv_cache_block_size = 16
    device = torch.device("cpu")

    # Create mock original metadata
    # Request 0: 64 prefix tokens → 4 prefix blocks, + 2 block blocks = 6 total blocks
    # Request 1: 32 prefix tokens → 2 prefix blocks, + 2 block blocks = 4 total blocks

    # Mock block table (physical page IDs)
    block_table_tensor = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],  # Request 0: 6 blocks
            [6, 7, 8, 9, -1, -1],  # Request 1: 4 blocks + padding
        ],
        dtype=torch.int32,
        device=device,
    )

    original_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 32, 64], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 32, 64], dtype=torch.int32),
        seq_lens=torch.tensor([64, 32], dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=64,  # Total query tokens
        max_query_len=32,
        max_seq_len=64,
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.zeros(64, dtype=torch.int64, device=device),
        causal=False,
    )

    # Execute: Create concatenated virtual batch
    concatenated_metadata = create_concatenated_virtual_batch(
        attn_metadata=original_metadata,
        num_prefix_tokens_per_request=num_prefix_tokens_per_request,
        block_size=block_size,
        kv_cache_block_size=kv_cache_block_size,
    )

    # Verify: Check metadata properties

    # 1. Combined seq_lens should be prefix + block
    expected_seq_lens = torch.tensor(
        [
            64 + 32,  # Request 0: 64 prefix + 32 block = 96
            32 + 32,  # Request 1: 32 prefix + 32 block = 64
        ],
        dtype=torch.int32,
        device=device,
    )

    assert torch.equal(concatenated_metadata.seq_lens, expected_seq_lens), (
        f"Expected seq_lens={expected_seq_lens.tolist()}, "
        f"got {concatenated_metadata.seq_lens.tolist()}"
    )

    # 2. max_seq_len should be max(combined_seq_lens)
    assert concatenated_metadata.max_seq_len == 96, (
        f"Expected max_seq_len=96, got {concatenated_metadata.max_seq_len}"
    )

    # 3. Block table should concatenate prefix + block pages
    # Request 0: pages [0,1,2,3] (prefix) + [4,5] (block) = [0,1,2,3,4,5]
    # Request 1: pages [6,7] (prefix) + [8,9] (block) = [6,7,8,9] + padding
    expected_block_table = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],  # Request 0: all 6 pages
            [6, 7, 8, 9, -1, -1],  # Request 1: 4 pages + padding
        ],
        dtype=torch.int32,
        device=device,
    )

    assert torch.equal(
        concatenated_metadata.block_table_tensor, expected_block_table
    ), (
        f"Block table mismatch:\n"
        f"Expected:\n{expected_block_table}\n"
        f"Got:\n{concatenated_metadata.block_table_tensor}"
    )

    # 4. num_reqs should be preserved
    assert concatenated_metadata.num_reqs == num_reqs

    # 5. max_query_len should be block_size
    assert concatenated_metadata.max_query_len == block_size

    # 6. causal should be False (bidirectional within block, block-causal across blocks)
    assert not concatenated_metadata.causal

    print("✅ Concatenated metadata constructed correctly")
    print(f"   Req 0: {num_prefix_tokens_per_request[0]}+{block_size}")
    print(f"   Req 1: {num_prefix_tokens_per_request[1]}+{block_size}")
    print(f"   Softmax over {expected_seq_lens[0]}/{expected_seq_lens[1]} keys")


def test_concatenated_first_block_edge_case():
    """Test edge case: first block with no prefix (prefix_length=0)."""
    pytest.importorskip("vllm")

    from vllm.v1.attention.backend import CommonAttentionMetadata

    from dllm_plugin.attention.concatenated_virtual_batch import (
        create_concatenated_virtual_batch,
    )

    # Setup: First block, no prefix
    num_reqs = 2
    num_prefix_tokens_per_request = [0, 0]  # Both requests at first block
    block_size = 32
    kv_cache_block_size = 16
    device = torch.device("cpu")

    # Mock block table: only block pages (no prefix)
    # Each request needs 2 blocks (32 / 16 = 2)
    block_table_tensor = torch.tensor(
        [
            [0, 1],  # Request 0: 2 block pages
            [2, 3],  # Request 1: 2 block pages
        ],
        dtype=torch.int32,
        device=device,
    )

    original_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 32, 64], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 32, 64], dtype=torch.int32),
        seq_lens=torch.tensor([0, 0], dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=64,
        max_query_len=32,
        max_seq_len=0,
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.zeros(64, dtype=torch.int64, device=device),
        causal=False,
    )

    # Execute
    concatenated_metadata = create_concatenated_virtual_batch(
        attn_metadata=original_metadata,
        num_prefix_tokens_per_request=num_prefix_tokens_per_request,
        block_size=block_size,
        kv_cache_block_size=kv_cache_block_size,
    )

    # Verify: seq_lens should be just block_size (no prefix)
    expected_seq_lens = torch.tensor([32, 32], dtype=torch.int32, device=device)
    assert torch.equal(concatenated_metadata.seq_lens, expected_seq_lens)

    # Block table should be just the block pages
    assert torch.equal(concatenated_metadata.block_table_tensor, block_table_tensor)

    print("✅ First block (no prefix) handled correctly")
    print(f"   seq_lens: {concatenated_metadata.seq_lens.tolist()}")


def test_concatenated_heterogeneous_prefixes():
    """Test heterogeneous prefix lengths across requests."""
    pytest.importorskip("vllm")

    from vllm.v1.attention.backend import CommonAttentionMetadata

    from dllm_plugin.attention.concatenated_virtual_batch import (
        create_concatenated_virtual_batch,
    )

    # Setup: 3 requests with very different prefix lengths
    num_reqs = 3
    num_prefix_tokens_per_request = [128, 64, 0]  # Large, medium, none
    block_size = 32
    kv_cache_block_size = 16
    device = torch.device("cpu")

    # Mock block table
    # Request 0: 128 prefix → 8 blocks, + 2 block blocks = 10 total
    # Request 1: 64 prefix → 4 blocks, + 2 block blocks = 6 total
    # Request 2: 0 prefix → 0 blocks, + 2 block blocks = 2 total

    block_table_tensor = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Request 0: 10 blocks
            [10, 11, 12, 13, 14, 15, -1, -1, -1, -1],  # Request 1: 6 blocks + padding
            [16, 17, -1, -1, -1, -1, -1, -1, -1, -1],  # Request 2: 2 blocks + padding
        ],
        dtype=torch.int32,
        device=device,
    )

    original_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 32, 64, 96], dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor([0, 32, 64, 96], dtype=torch.int32),
        seq_lens=torch.tensor([128, 64, 0], dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=96,
        max_query_len=32,
        max_seq_len=128,
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.zeros(96, dtype=torch.int64, device=device),
        causal=False,
    )

    # Execute
    concatenated_metadata = create_concatenated_virtual_batch(
        attn_metadata=original_metadata,
        num_prefix_tokens_per_request=num_prefix_tokens_per_request,
        block_size=block_size,
        kv_cache_block_size=kv_cache_block_size,
    )

    # Verify: Combined seq_lens
    expected_seq_lens = torch.tensor(
        [
            128 + 32,  # Request 0: 160
            64 + 32,  # Request 1: 96
            0 + 32,  # Request 2: 32
        ],
        dtype=torch.int32,
        device=device,
    )

    assert torch.equal(concatenated_metadata.seq_lens, expected_seq_lens), (
        "seq_lens mismatch"
    )

    # max_seq_len should be max(160, 96, 32) = 160
    assert concatenated_metadata.max_seq_len == 160

    print("✅ Heterogeneous prefix lengths handled correctly")
    print(f"   Req0: {num_prefix_tokens_per_request[0]}+{block_size}")
    print(f"   Req1: {num_prefix_tokens_per_request[1]}+{block_size}")
    print(f"   Req2: {num_prefix_tokens_per_request[2]}+{block_size}")


if __name__ == "__main__":
    # Run tests directly (useful for debugging)
    print("Running concatenated virtual batch unit tests...")
    print()

    try:
        test_concatenated_metadata_construction()
        print()
        test_concatenated_first_block_edge_case()
        print()
        test_concatenated_heterogeneous_prefixes()
        print()
        print("=" * 50)
        print("✅ All unit tests passed!")
        print("=" * 50)
    except Exception as e:
        print()
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        print("=" * 50)
        raise
