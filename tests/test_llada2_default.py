# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for :class:`~vllm_dllm_plugin.remasking.Llada2DefaultRemaskingPolicy`."""

from __future__ import annotations

import pytest

from vllm_dllm_plugin.config import (
    DRAFT_SIZE,
    LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD,
    LLADA2_DEFAULT_MASK_TOKEN_ID,
)
from vllm_dllm_plugin.remasking import (
    Llada2DefaultRemaskingPolicy,
    RemaskingPolicy,
    validate_remask_step_result,
)


def _draft_all_mask() -> tuple[int, ...]:
    return (LLADA2_DEFAULT_MASK_TOKEN_ID,) * DRAFT_SIZE


def _mock_stub_row(*, vocab_size: int = 256) -> list[float]:
    """Match ``docs/MOCK_STACK_MODEL.md`` stub: zeros plus ``1.0`` at index ``0``."""
    row = [0.0] * vocab_size
    row[0] = 1.0
    return row


def _mock_logits(*, vocab_size: int = 256) -> list[list[float]]:
    return [_mock_stub_row(vocab_size=vocab_size) for _ in range(DRAFT_SIZE)]


def test_llada2_default_is_remasking_policy() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    assert isinstance(policy, RemaskingPolicy)


def test_mock_shaped_logits_all_mask_terminal() -> None:
    """Stub logits: all masks high-confidence; one step clears the block."""
    policy = Llada2DefaultRemaskingPolicy()
    logits = _mock_logits(vocab_size=256)
    out = policy.apply(input_draft=_draft_all_mask(), logits=logits)
    validate_remask_step_result(out)
    assert out.committed_token_ids == (0,) * DRAFT_SIZE
    assert out.next_input_block == (LLADA2_DEFAULT_MASK_TOKEN_ID,) * DRAFT_SIZE


def test_uniform_logits_topk_one_transfer_option_a() -> None:
    """Equal logits below threshold: top-k picks one masked index (smallest)."""
    policy = Llada2DefaultRemaskingPolicy()
    row = [0.0] * 128
    logits = [list(row) for _ in range(DRAFT_SIZE)]
    out = policy.apply(input_draft=_draft_all_mask(), logits=logits)
    validate_remask_step_result(out)
    assert out.committed_token_ids == ()
    expected = [LLADA2_DEFAULT_MASK_TOKEN_ID] * DRAFT_SIZE
    expected[0] = 0
    assert out.next_input_block == tuple(expected)


def test_logit_tie_breaks_to_smallest_index_topk() -> None:
    """Tied max logits: argmax at 0; top-k uses smallest index among ties."""
    policy = Llada2DefaultRemaskingPolicy()
    row = [1.0, 1.0, 0.0, 0.0]
    logits = [list(row) for _ in range(DRAFT_SIZE)]
    out = policy.apply(
        input_draft=_draft_all_mask(),
        logits=logits,
        remasking_config={"commit_confidence_threshold": 0.99},
    )
    assert out.committed_token_ids == ()
    expected = [LLADA2_DEFAULT_MASK_TOKEN_ID] * DRAFT_SIZE
    expected[0] = 0
    assert out.next_input_block == tuple(expected)


def test_preserve_decoded_positions_only_masked_transfer() -> None:
    """Literal tokens stay fixed; only mask slots use logits-driven transfers."""
    policy = Llada2DefaultRemaskingPolicy()
    prefix = (42,) * (DRAFT_SIZE // 2)
    suffix_masks = (LLADA2_DEFAULT_MASK_TOKEN_ID,) * (DRAFT_SIZE // 2)
    draft = prefix + suffix_masks
    row_low = [0.0] * 128
    row_high = [0.0] * 128
    row_high[3] = 50.0
    logits: list[list[float]] = []
    for i in range(DRAFT_SIZE):
        if i < DRAFT_SIZE // 2:
            logits.append(list(row_low))
        elif i < DRAFT_SIZE // 2 + DRAFT_SIZE // 4:
            logits.append(list(row_high))
        else:
            logits.append(list(row_low))
    out = policy.apply(
        input_draft=draft,
        logits=logits,
        remasking_config={"commit_confidence_threshold": 0.15},
    )
    assert out.committed_token_ids == ()
    next_list = list(out.next_input_block)
    assert next_list[: DRAFT_SIZE // 2] == list(prefix)
    for i in range(DRAFT_SIZE // 2, DRAFT_SIZE // 2 + DRAFT_SIZE // 4):
        assert next_list[i] == 3
    for i in range(DRAFT_SIZE // 2 + DRAFT_SIZE // 4, DRAFT_SIZE):
        assert next_list[i] == LLADA2_DEFAULT_MASK_TOKEN_ID


def test_terminal_input_already_unmasked() -> None:
    """No mask in input: full block commit and all-mask next draft."""
    policy = Llada2DefaultRemaskingPolicy()
    # Avoid token id ``LLADA2_DEFAULT_MASK_TOKEN_ID`` in the draft (e.g. ``1``).
    draft = tuple(i * 2 for i in range(DRAFT_SIZE))
    logits = _mock_logits(vocab_size=16)
    out = policy.apply(input_draft=draft, logits=logits)
    validate_remask_step_result(out)
    assert out.committed_token_ids == draft
    assert out.next_input_block == (LLADA2_DEFAULT_MASK_TOKEN_ID,) * DRAFT_SIZE


def test_high_confidence_flood_transfers_all_above_threshold() -> None:
    """When h >= num_to_transfer, every high-confidence masked position transfers."""
    policy = Llada2DefaultRemaskingPolicy()
    high = [0.0] * 128
    high[7] = 50.0
    low = [0.0] * 128
    logits = [list(high) for _ in range(5)] + [list(low) for _ in range(DRAFT_SIZE - 5)]
    out = policy.apply(
        input_draft=_draft_all_mask(),
        logits=logits,
        remasking_config={
            "commit_confidence_threshold": 0.15,
            "num_transfer": 2,
        },
    )
    assert out.committed_token_ids == ()
    for i in range(5):
        assert out.next_input_block[i] == 7
    for i in range(5, DRAFT_SIZE):
        assert out.next_input_block[i] == LLADA2_DEFAULT_MASK_TOKEN_ID


def test_topk_branch_all_strictly_below_threshold() -> None:
    """No position meets threshold; top-k moves min(num_transfer, num_masked) slots."""
    policy = Llada2DefaultRemaskingPolicy()
    row = [0.0] * 128
    logits = [list(row) for _ in range(DRAFT_SIZE)]
    out = policy.apply(
        input_draft=_draft_all_mask(),
        logits=logits,
        remasking_config={
            "commit_confidence_threshold": 1.0,
            "num_transfer": 3,
        },
    )
    assert out.committed_token_ids == ()
    expected = [LLADA2_DEFAULT_MASK_TOKEN_ID] * DRAFT_SIZE
    for i in range(3):
        expected[i] = 0
    assert out.next_input_block == tuple(expected)


def test_threshold_inclusive_equals_counts_as_high_confidence() -> None:
    """``confidence == threshold`` counts as high-confidence (inclusive bound)."""
    policy = Llada2DefaultRemaskingPolicy()
    row = [0.0] * 8
    logits = [list(row) for _ in range(DRAFT_SIZE)]
    out = policy.apply(
        input_draft=_draft_all_mask(),
        logits=logits,
        remasking_config={"commit_confidence_threshold": 0.125},
    )
    assert out.committed_token_ids == (0,) * DRAFT_SIZE
    assert out.next_input_block == (LLADA2_DEFAULT_MASK_TOKEN_ID,) * DRAFT_SIZE


def test_option_a_mid_step_empty_committed() -> None:
    """Draft still has mask: ``committed_token_ids`` stays empty (Option A)."""
    policy = Llada2DefaultRemaskingPolicy()
    row = [0.0] * 128
    logits = [list(row) for _ in range(DRAFT_SIZE)]
    out = policy.apply(input_draft=_draft_all_mask(), logits=logits)
    assert out.committed_token_ids == ()
    assert LLADA2_DEFAULT_MASK_TOKEN_ID in out.next_input_block


def test_threshold_above_mock_softmax_no_terminal() -> None:
    """Threshold above stub softmax mass -> top-k path; still mid-block (Option A)."""
    policy = Llada2DefaultRemaskingPolicy()
    logits = _mock_logits(vocab_size=256)
    out = policy.apply(
        input_draft=_draft_all_mask(),
        logits=logits,
        remasking_config={"commit_confidence_threshold": 0.011},
    )
    assert out.committed_token_ids == ()
    assert out.next_input_block[0] == 0
    assert LLADA2_DEFAULT_MASK_TOKEN_ID in out.next_input_block


def test_custom_mask_token_id() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    row = [0.0] * 8
    logits = [list(row) for _ in range(DRAFT_SIZE)]
    out = policy.apply(
        input_draft=(99,) * DRAFT_SIZE,
        logits=logits,
        remasking_config={"mask_token_id": 99, "commit_confidence_threshold": 0.5},
    )
    assert out.committed_token_ids == ()
    assert out.next_input_block[0] == 0
    assert 99 in out.next_input_block


def test_logits_none_raises() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    with pytest.raises(ValueError, match="logits is required"):
        policy.apply(input_draft=_draft_all_mask(), logits=None)


def test_wrong_input_draft_length_raises() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    with pytest.raises(ValueError, match="input_draft"):
        policy.apply(input_draft=(1, 2, 3), logits=_mock_logits())


def test_wrong_logits_first_dim_raises() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    short = [_mock_stub_row() for _ in range(3)]
    with pytest.raises(ValueError, match="first dimension"):
        policy.apply(input_draft=_draft_all_mask(), logits=short)


def test_inconsistent_vocab_width_raises() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    bad_logits = [[0.0, 1.0]] + [[0.0] * 8 for _ in range(DRAFT_SIZE - 1)]
    with pytest.raises(ValueError, match="inconsistent vocab"):
        policy.apply(input_draft=_draft_all_mask(), logits=bad_logits)


def test_denoise_steps_zero_raises() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    with pytest.raises(ValueError, match="denoise_steps must be positive"):
        policy.apply(
            input_draft=_draft_all_mask(),
            logits=_mock_logits(),
            remasking_config={"denoise_steps": 0},
        )


def test_default_threshold_constant_documented() -> None:
    assert 0.0 < LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD < 1.0


def test_torch_tensor_logits_matches_nested_list() -> None:
    torch = pytest.importorskip("torch")
    policy = Llada2DefaultRemaskingPolicy()
    rows = _mock_logits(vocab_size=16)
    list_out = policy.apply(input_draft=_draft_all_mask(), logits=rows)
    tensor = torch.tensor(rows, dtype=torch.float32)
    tensor_out = policy.apply(input_draft=_draft_all_mask(), logits=tensor)
    assert list_out == tensor_out
