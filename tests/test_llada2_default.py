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


def _input_block() -> tuple[int, ...]:
    return tuple(range(DRAFT_SIZE))


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


def test_mock_shaped_logits_commit_argmax_zero() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    logits = _mock_logits(vocab_size=256)
    out = policy.apply(input_block=_input_block(), logits=logits)
    validate_remask_step_result(out)
    assert out.committed_token_ids == (0,) * DRAFT_SIZE
    assert out.next_input_block == (0,) * DRAFT_SIZE


def test_uniform_logits_all_remask() -> None:
    """Equal logits -> low max-softmax prob -> no commits; next block is all mask."""
    policy = Llada2DefaultRemaskingPolicy()
    # Need vocab large enough that 1/vocab < default threshold (0.01).
    row = [0.0] * 128
    logits = [list(row) for _ in range(DRAFT_SIZE)]
    out = policy.apply(input_block=_input_block(), logits=logits)
    validate_remask_step_result(out)
    assert out.committed_token_ids == ()
    assert out.next_input_block == (LLADA2_DEFAULT_MASK_TOKEN_ID,) * DRAFT_SIZE


def test_high_confidence_commits_custom_argmax() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    row = [0.0] * 8
    row[3] = 50.0
    logits = [list(row) for _ in range(DRAFT_SIZE)]
    out = policy.apply(input_block=_input_block(), logits=logits)
    assert out.committed_token_ids == (3,) * DRAFT_SIZE
    assert out.next_input_block == (3,) * DRAFT_SIZE


def test_threshold_boundary_remasks_mock_when_raised() -> None:
    """Slightly above mock stub max-softmax (~0.0105 at vocab 256) remasks all."""
    policy = Llada2DefaultRemaskingPolicy()
    logits = _mock_logits(vocab_size=256)
    out = policy.apply(
        input_block=_input_block(),
        logits=logits,
        remasking_config={"commit_confidence_threshold": 0.011},
    )
    assert out.committed_token_ids == ()
    assert out.next_input_block == (LLADA2_DEFAULT_MASK_TOKEN_ID,) * DRAFT_SIZE


def test_custom_mask_token_id() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    row = [0.0] * 8
    logits = [list(row) for _ in range(DRAFT_SIZE)]
    # Uniform row: max softmax prob is 1/8 = 0.125; require > 0.2 to remask all.
    out = policy.apply(
        input_block=_input_block(),
        logits=logits,
        remasking_config={"mask_token_id": 99, "commit_confidence_threshold": 0.2},
    )
    assert out.next_input_block == (99,) * DRAFT_SIZE


def test_logits_none_raises() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    with pytest.raises(ValueError, match="logits is required"):
        policy.apply(input_block=_input_block(), logits=None)


def test_wrong_input_block_length_raises() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    with pytest.raises(ValueError, match="input_block"):
        policy.apply(input_block=(1, 2, 3), logits=_mock_logits())


def test_wrong_logits_first_dim_raises() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    short = [_mock_stub_row() for _ in range(3)]
    with pytest.raises(ValueError, match="first dimension"):
        policy.apply(input_block=_input_block(), logits=short)


def test_inconsistent_vocab_width_raises() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    bad_logits = [[0.0, 1.0]] + [[0.0] * 8 for _ in range(DRAFT_SIZE - 1)]
    with pytest.raises(ValueError, match="inconsistent vocab"):
        policy.apply(input_block=_input_block(), logits=bad_logits)


def test_default_threshold_constant_documented() -> None:
    assert 0.0 < LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD < 1.0


def test_torch_tensor_logits_matches_nested_list() -> None:
    torch = pytest.importorskip("torch")
    policy = Llada2DefaultRemaskingPolicy()
    rows = _mock_logits(vocab_size=16)
    list_out = policy.apply(input_block=_input_block(), logits=rows)
    tensor = torch.tensor(rows, dtype=torch.float32)
    tensor_out = policy.apply(input_block=_input_block(), logits=tensor)
    assert list_out == tensor_out
