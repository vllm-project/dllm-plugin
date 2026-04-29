# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 4 scheduler tests (issues #8 and #9)."""

from __future__ import annotations

import pytest

from dllm_plugin.config import DRAFT_SIZE, LLADA2_DEFAULT_MASK_TOKEN_ID
from dllm_plugin.scheduler import (
    DllmRequestState,
    DllmScheduler,
    DllmWorkerResult,
)


def test_initialize_first_block_pads_with_mask() -> None:
    sched = DllmScheduler()
    block = sched.initialize_first_block(prompt_token_ids=(10, 11, 12))
    assert len(block) == DRAFT_SIZE
    assert block[:3] == (10, 11, 12)
    assert block[3:] == (LLADA2_DEFAULT_MASK_TOKEN_ID,) * (DRAFT_SIZE - 3)


def test_schedule_decode_step_initializes_spec_tokens_and_counts() -> None:
    sched = DllmScheduler()
    state = DllmRequestState(request_id="r1")

    out = sched.schedule_decode_step(requests=((state, (7, 8)),))

    assert len(out.requests) == 1
    req = out.requests[0]
    assert req.request_id == "r1"
    assert req.num_scheduled_tokens == DRAFT_SIZE
    assert len(req.scheduled_spec_decode_tokens) == DRAFT_SIZE
    assert state.spec_token_ids == req.scheduled_spec_decode_tokens
    assert state.num_computed_tokens == DRAFT_SIZE


def test_update_from_output_rolls_back_on_empty_commit() -> None:
    sched = DllmScheduler()
    state = DllmRequestState(request_id="r1")
    sched.schedule_decode_step(requests=((state, ()),))
    assert state.num_computed_tokens == DRAFT_SIZE

    sched.update_from_output(
        states={"r1": state},
        worker_results=(DllmWorkerResult(request_id="r1", sampled_token_ids=()),),
    )
    assert state.num_computed_tokens == 0


def test_update_from_output_counts_only_committed_tokens() -> None:
    sched = DllmScheduler()
    state = DllmRequestState(request_id="r1")
    sched.schedule_decode_step(requests=((state, ()),))
    assert state.num_computed_tokens == DRAFT_SIZE

    sched.update_from_output(
        states={"r1": state},
        worker_results=(DllmWorkerResult(request_id="r1", sampled_token_ids=(1,)),),
    )
    assert state.num_computed_tokens == 1


def test_update_from_output_multi_step_partial_commits_accumulate_correctly() -> None:
    sched = DllmScheduler()
    state = DllmRequestState(request_id="r1")

    sched.schedule_decode_step(requests=((state, ()),))
    sched.update_from_output(
        states={"r1": state},
        worker_results=(
            DllmWorkerResult(request_id="r1", sampled_token_ids=(1, 2, 3, 4, 5)),
        ),
    )
    assert state.num_computed_tokens == 5

    sched.schedule_decode_step(requests=((state, ()),))
    sched.update_from_output(
        states={"r1": state},
        worker_results=(
            DllmWorkerResult(request_id="r1", sampled_token_ids=(9, 10, 11)),
        ),
    )
    assert state.num_computed_tokens == 8


def test_update_from_output_rejects_unknown_request_id() -> None:
    sched = DllmScheduler()
    state = DllmRequestState(request_id="known")
    sched.schedule_decode_step(requests=((state, ()),))

    with pytest.raises(ValueError, match="unknown request_id"):
        sched.update_from_output(
            states={"known": state},
            worker_results=(
                DllmWorkerResult(request_id="other", sampled_token_ids=()),
            ),
        )


def test_update_from_output_rejects_missing_worker_result() -> None:
    sched = DllmScheduler()
    state1 = DllmRequestState(request_id="r1")
    state2 = DllmRequestState(request_id="r2")
    sched.schedule_decode_step(requests=((state1, ()), (state2, ())))

    with pytest.raises(ValueError, match="missing worker results"):
        sched.update_from_output(
            states={"r1": state1, "r2": state2},
            worker_results=(DllmWorkerResult(request_id="r1", sampled_token_ids=(1,)),),
        )


def test_update_from_output_rejects_duplicate_request_ids() -> None:
    sched = DllmScheduler()
    state = DllmRequestState(request_id="r1")
    sched.schedule_decode_step(requests=((state, ()),))

    with pytest.raises(ValueError, match="duplicate request_id"):
        sched.update_from_output(
            states={"r1": state},
            worker_results=(
                DllmWorkerResult(request_id="r1", sampled_token_ids=(1,)),
                DllmWorkerResult(request_id="r1", sampled_token_ids=(2,)),
            ),
        )


def test_update_draft_token_ids_rejects_grammar_constrained_path() -> None:
    sched = DllmScheduler()
    state = DllmRequestState(request_id="r1")

    with pytest.raises(ValueError, match="incompatible with AR draft grammar"):
        sched.update_draft_token_ids(
            state=state,
            next_input_block=(0,) * DRAFT_SIZE,
            grammar_constrained=True,
        )


def test_update_draft_token_ids_in_output_rejects_grammar_constrained_path() -> None:
    sched = DllmScheduler()
    state = DllmRequestState(request_id="r1")
    output = sched.schedule_decode_step(requests=((state, (1, 2, 3)),))

    with pytest.raises(ValueError, match="incompatible with AR draft grammar"):
        sched.update_draft_token_ids_in_output(
            output=output,
            next_blocks_by_request_id={"r1": (9,) * DRAFT_SIZE},
            grammar_constrained=True,
        )
