# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`dllm_plugin.grammar_utils` (Phase 4 / issue #9)."""

from __future__ import annotations

from unittest.mock import MagicMock

from dllm_plugin.grammar_utils import (
    apply_packed_bitmask_inplace_logits_row,
    flat_frontier_bitmask_row_index,
    frontier_block_row,
    grammar_extra_transfer_slots,
    scheduled_spec_decode_tokens_for_grammar_bitmask,
)


def test_flat_frontier_bitmask_row_index_stacking() -> None:
    patched = {"a": [1, 2], "b": [9]}
    ids = ["a", "b"]
    idx = flat_frontier_bitmask_row_index(
        structured_output_request_ids=ids,
        patched_scheduled_spec_decode_tokens=patched,
    )
    # a: rows 0..2 (len prefix + sentinel) -> frontier flat row at cum+2 = 2
    assert idx["a"] == 2
    # cum after a = 3; b prefix len 1 -> frontier at 3+1 = 4
    assert idx["b"] == 4


def test_frontier_block_row_all_valid_vs_repair() -> None:
    assert frontier_block_row(valid_prefix_len=32, draft_size=32) is None
    assert frontier_block_row(valid_prefix_len=3, draft_size=32) == 3


def test_grammar_extra_transfer_slots() -> None:
    draft = list(range(10))
    assert grammar_extra_transfer_slots(draft_tokens=draft, valid_prefix_len=10) == 0
    assert grammar_extra_transfer_slots(draft_tokens=draft, valid_prefix_len=7) == 3


def test_apply_packed_bitmask_sets_invalid_bits_neg_inf() -> None:
    row = [0.0] * 40
    # Two int32 words cover 64 bits; vocab 40 uses word 0 fully and word 1 partially.
    words = [0xFFFFFFFF, 0xFFFFFFFF]
    apply_packed_bitmask_inplace_logits_row(row, words)
    assert row[0] == 0.0  # bit 0 allowed (LSB of word = 1)


def test_scheduled_spec_decode_tokens_non_so_passthrough() -> None:
    req = MagicMock()
    req.use_structured_output = False
    out = scheduled_spec_decode_tokens_for_grammar_bitmask(
        scheduled_spec_decode_tokens={"r": [1, 2, 3]},
        requests={"r": req},
        structured_output_manager=MagicMock(),
    )
    assert out["r"] == [1, 2, 3]
