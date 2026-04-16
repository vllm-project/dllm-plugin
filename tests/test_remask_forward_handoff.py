# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm_dllm_plugin.remasking.handoff`` (issue #13).

Broader field-mapping / worker-runner contract coverage lives in issue #16.
"""

from __future__ import annotations

import pytest

from vllm_dllm_plugin.config import DRAFT_SIZE, LLADA2_DEFAULT_MASK_TOKEN_ID
from vllm_dllm_plugin.remasking import (
    Llada2DefaultRemaskingPolicy,
    assert_block_logits_shape,
    remask_after_block_forward,
    validate_remask_step_result,
)


def _draft_all_mask() -> tuple[int, ...]:
    return (LLADA2_DEFAULT_MASK_TOKEN_ID,) * DRAFT_SIZE


def _mock_stub_row(*, vocab_size: int = 256) -> list[float]:
    row = [0.0] * vocab_size
    row[0] = 1.0
    return row


def _mock_logits(*, vocab_size: int = 256) -> list[list[float]]:
    return [_mock_stub_row(vocab_size=vocab_size) for _ in range(DRAFT_SIZE)]


def test_remask_rejects_wrong_input_draft_length() -> None:
    with pytest.raises(ValueError, match="input_draft length"):
        remask_after_block_forward(
            input_draft=(0,) * (DRAFT_SIZE - 1),
            logits=_mock_logits(),
        )


def test_remask_rejects_logits_none() -> None:
    with pytest.raises(ValueError, match="logits is None"):
        remask_after_block_forward(
            input_draft=_draft_all_mask(),
            logits=None,
        )


def test_assert_block_logits_shape_rejects_wrong_len_sequence() -> None:
    short = [_mock_stub_row() for _ in range(DRAFT_SIZE - 1)]
    with pytest.raises(ValueError, match="DRAFT_SIZE"):
        assert_block_logits_shape(short)


def test_assert_block_logits_shape_rejects_none() -> None:
    with pytest.raises(ValueError, match="logits is None"):
        assert_block_logits_shape(None)


def test_mock_shaped_logits_terminal_matches_direct_policy_apply() -> None:
    draft = _draft_all_mask()
    logits = _mock_logits(vocab_size=256)
    direct = Llada2DefaultRemaskingPolicy().apply(
        input_draft=draft,
        logits=logits,
    )
    via = remask_after_block_forward(input_draft=draft, logits=logits)
    validate_remask_step_result(via)
    assert via == direct


def test_custom_policy_passed_through() -> None:
    policy = Llada2DefaultRemaskingPolicy()
    draft = _draft_all_mask()
    logits = _mock_logits()
    out = remask_after_block_forward(
        input_draft=draft,
        logits=logits,
        policy=policy,
    )
    validate_remask_step_result(out)
    assert out.committed_token_ids == (0,) * DRAFT_SIZE


def test_torch_tensor_logits_matches_list_path() -> None:
    torch = pytest.importorskip("torch")
    draft = _draft_all_mask()
    vocab = 256
    logits_t = torch.zeros(DRAFT_SIZE, vocab, dtype=torch.float32)
    logits_t[:, 0] = 1.0
    logits_list = _mock_logits(vocab_size=vocab)
    out_t = remask_after_block_forward(input_draft=draft, logits=logits_t)
    out_list = remask_after_block_forward(input_draft=draft, logits=logits_list)
    assert out_t == out_list


def test_assert_block_logits_shape_rejects_wrong_tensor_rank() -> None:
    torch = pytest.importorskip("torch")
    bad = torch.zeros(DRAFT_SIZE, 16, 16)
    with pytest.raises(ValueError, match="2-D"):
        assert_block_logits_shape(bad)


def test_remask_rejects_wrong_tensor_first_dim() -> None:
    torch = pytest.importorskip("torch")
    bad = torch.zeros(DRAFT_SIZE + 1, 128)
    with pytest.raises(ValueError, match="DRAFT_SIZE"):
        remask_after_block_forward(input_draft=_draft_all_mask(), logits=bad)
