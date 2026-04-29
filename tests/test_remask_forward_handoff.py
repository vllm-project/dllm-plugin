# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``dllm_plugin.remasking.handoff`` (issue #13).

Broader field-mapping / worker-runner contract coverage lives in issue #16.
Tensor-shape tests use ``pytest.importorskip("torch")``; they skip in the default
dev sync (no PyTorch) and run in the CI ``vllm-extra`` job, which installs
``--extra vllm`` (Torch is a transitive dependency of vLLM).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from dllm_plugin.config import DRAFT_SIZE, LLADA2_DEFAULT_MASK_TOKEN_ID
from dllm_plugin.remasking import (
    Llada2DefaultRemaskingPolicy,
    RemaskStepResult,
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


@pytest.fixture
def llada2_policy() -> Llada2DefaultRemaskingPolicy:
    return Llada2DefaultRemaskingPolicy()


class _StubRemaskingPolicy:
    """Distinct from Llada2DefaultRemaskingPolicy; proves ``policy=`` is used."""

    __test__ = False

    def apply(
        self,
        *,
        input_draft: Sequence[int],
        logits: Any | None = None,
        remasking_config: Mapping[str, Any] | None = None,
    ) -> RemaskStepResult:
        del input_draft, logits, remasking_config
        return RemaskStepResult(
            committed_token_ids=(123, 456),
            next_input_block=(99,) * DRAFT_SIZE,
        )


def test_remask_rejects_wrong_input_draft_length(
    llada2_policy: Llada2DefaultRemaskingPolicy,
) -> None:
    with pytest.raises(ValueError, match="input_draft length"):
        remask_after_block_forward(
            input_draft=(0,) * (DRAFT_SIZE - 1),
            logits=_mock_logits(),
            policy=llada2_policy,
        )


def test_remask_rejects_logits_none(
    llada2_policy: Llada2DefaultRemaskingPolicy,
) -> None:
    with pytest.raises(ValueError, match="logits is None"):
        remask_after_block_forward(
            input_draft=_draft_all_mask(),
            logits=None,
            policy=llada2_policy,
        )


def test_assert_block_logits_shape_rejects_wrong_len_sequence() -> None:
    short = [_mock_stub_row() for _ in range(DRAFT_SIZE - 1)]
    with pytest.raises(ValueError, match="draft_size"):
        assert_block_logits_shape(short)


def test_assert_block_logits_shape_rejects_none() -> None:
    with pytest.raises(ValueError, match="logits is None"):
        assert_block_logits_shape(None)


def test_mock_shaped_logits_terminal_matches_direct_policy_apply(
    llada2_policy: Llada2DefaultRemaskingPolicy,
) -> None:
    draft = _draft_all_mask()
    logits = _mock_logits(vocab_size=256)
    direct = llada2_policy.apply(
        input_draft=draft,
        logits=logits,
    )
    via = remask_after_block_forward(
        input_draft=draft,
        logits=logits,
        policy=llada2_policy,
    )
    validate_remask_step_result(via)
    assert via == direct


def test_stub_policy_passed_through() -> None:
    stub = _StubRemaskingPolicy()
    out = remask_after_block_forward(
        input_draft=_draft_all_mask(),
        logits=_mock_logits(),
        policy=stub,
    )
    assert out.committed_token_ids == (123, 456)
    assert out.next_input_block == (99,) * DRAFT_SIZE


def test_remasking_config_forwarded_to_policy(
    llada2_policy: Llada2DefaultRemaskingPolicy,
) -> None:
    draft = _draft_all_mask()
    logits = _mock_logits()
    cfg = {"num_transfer": 1}
    direct = llada2_policy.apply(
        input_draft=draft,
        logits=logits,
        remasking_config=cfg,
    )
    via = remask_after_block_forward(
        input_draft=draft,
        logits=logits,
        policy=llada2_policy,
        remasking_config=cfg,
    )
    assert via == direct


def test_torch_tensor_logits_matches_list_path(
    llada2_policy: Llada2DefaultRemaskingPolicy,
) -> None:
    torch = pytest.importorskip("torch")
    draft = _draft_all_mask()
    vocab = 256
    logits_t = torch.zeros(DRAFT_SIZE, vocab, dtype=torch.float32)
    logits_t[:, 0] = 1.0
    logits_list = _mock_logits(vocab_size=vocab)
    out_t = remask_after_block_forward(
        input_draft=draft,
        logits=logits_t,
        policy=llada2_policy,
    )
    out_list = remask_after_block_forward(
        input_draft=draft,
        logits=logits_list,
        policy=llada2_policy,
    )
    assert out_t == out_list


def test_assert_block_logits_shape_rejects_wrong_tensor_rank() -> None:
    torch = pytest.importorskip("torch")
    bad = torch.zeros(DRAFT_SIZE, 16, 16)
    with pytest.raises(ValueError, match="2-D"):
        assert_block_logits_shape(bad)


def test_remask_rejects_wrong_tensor_first_dim(
    llada2_policy: Llada2DefaultRemaskingPolicy,
) -> None:
    torch = pytest.importorskip("torch")
    bad = torch.zeros(DRAFT_SIZE + 1, 128)
    with pytest.raises(ValueError, match="draft_size"):
        remask_after_block_forward(
            input_draft=_draft_all_mask(),
            logits=bad,
            policy=llada2_policy,
        )
