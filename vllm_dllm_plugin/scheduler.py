# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 4 scheduler semantics for dLLM block decode.

This module models the scheduler-side contract in ``docs/DESIGN_MVP.md`` and
``docs/CONTRACTS.md`` without forcing a hard runtime dependency on vLLM types.

Key invariants:
- ``spec_token_ids`` holds the next-step input block (length ``DRAFT_SIZE``).
- One decode schedule emits exactly ``DRAFT_SIZE`` draft tokens per request.
- Commit-0 rollback subtracts scheduled tokens when a step commits no tokens.
- Grammar-constrained draft rewriting is explicitly unsupported for dLLM blocks
  in this MVP path; callers get a hard error instead of silent corruption.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vllm_dllm_plugin.config import DRAFT_SIZE, LLADA2_DEFAULT_MASK_TOKEN_ID


@dataclass(slots=True)
class DllmRequestState:
    """Minimal per-request state used by :class:`DllmScheduler`."""

    request_id: str
    num_computed_tokens: int = 0
    spec_token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class DllmScheduledRequest:
    """One scheduled block for a single request."""

    request_id: str
    scheduled_spec_decode_tokens: tuple[int, ...]
    num_scheduled_tokens: int


@dataclass(frozen=True, slots=True)
class DllmSchedulerOutput:
    """Block-schedule output for one scheduler step."""

    requests: tuple[DllmScheduledRequest, ...]


@dataclass(frozen=True, slots=True)
class DllmWorkerResult:
    """Scheduler-relevant subset of worker/model output."""

    request_id: str
    sampled_token_ids: tuple[int, ...]


class DllmScheduler:
    """Scheduler helper implementing issue #8 + issue #9 behavior."""

    def __init__(
        self,
        *,
        draft_size: int = DRAFT_SIZE,
        mask_token_id: int = LLADA2_DEFAULT_MASK_TOKEN_ID,
    ) -> None:
        if draft_size <= 0:
            raise ValueError("draft_size must be positive")
        self.draft_size = int(draft_size)
        self.mask_token_id = int(mask_token_id)

    def initialize_first_block(
        self,
        *,
        prompt_token_ids: Sequence[int],
    ) -> tuple[int, ...]:
        """Return a deterministic first block from prompt suffix + mask padding."""

        tail = tuple(int(tok) for tok in prompt_token_ids[-self.draft_size :])
        if len(tail) < self.draft_size:
            pad = (self.mask_token_id,) * (self.draft_size - len(tail))
            tail = tail + pad
        return tail

    def ensure_spec_token_ids(
        self,
        state: DllmRequestState,
        *,
        prompt_token_ids: Sequence[int],
    ) -> tuple[int, ...]:
        """Initialize ``state.spec_token_ids`` if missing and return the block."""

        if state.spec_token_ids is None:
            state.spec_token_ids = self.initialize_first_block(
                prompt_token_ids=prompt_token_ids,
            )
        self._validate_block(state.spec_token_ids, field_name="spec_token_ids")
        return state.spec_token_ids

    def schedule_decode_step(
        self,
        requests: Sequence[tuple[DllmRequestState, Sequence[int]]],
    ) -> DllmSchedulerOutput:
        """Schedule one block for each request.

        Args:
            requests: Sequence of ``(request_state, prompt_token_ids)`` entries.
                ``prompt_token_ids`` is used only when ``spec_token_ids`` needs
                first-block initialization.
        """

        scheduled: list[DllmScheduledRequest] = []
        for state, prompt in requests:
            block = self.ensure_spec_token_ids(state, prompt_token_ids=prompt)
            scheduled.append(
                DllmScheduledRequest(
                    request_id=state.request_id,
                    scheduled_spec_decode_tokens=block,
                    num_scheduled_tokens=self.draft_size,
                ),
            )
            state.num_computed_tokens += self.draft_size
        return DllmSchedulerOutput(requests=tuple(scheduled))

    def update_from_output(
        self,
        *,
        states: dict[str, DllmRequestState],
        worker_results: Sequence[DllmWorkerResult],
    ) -> None:
        """Apply commit accounting with commit-0 rollback."""

        for result in worker_results:
            state = states[result.request_id]
            committed = len(result.sampled_token_ids)
            if committed == 0:
                # Commit-0 rollback: this step contributed no finalized tokens.
                state.num_computed_tokens -= self.draft_size
            if state.num_computed_tokens < 0:
                state.num_computed_tokens = 0

    def update_draft_token_ids(
        self,
        *,
        state: DllmRequestState,
        next_input_block: Sequence[int],
        grammar_constrained: bool = False,
    ) -> None:
        """Store the next block; grammar-mutating mode is disallowed in MVP."""

        if grammar_constrained:
            raise ValueError(
                "dLLM block drafts are incompatible with AR draft grammar rewriting; "
                "structured-output grammar constraints are out of MVP scope.",
            )
        state.spec_token_ids = self._normalize_block(next_input_block)

    def update_draft_token_ids_in_output(
        self,
        *,
        output: DllmSchedulerOutput,
        next_blocks_by_request_id: dict[str, Sequence[int]],
        grammar_constrained: bool = False,
    ) -> DllmSchedulerOutput:
        """Return a new scheduler output with draft updates applied safely."""

        if grammar_constrained:
            raise ValueError(
                "dLLM block drafts are incompatible with AR draft grammar rewriting; "
                "structured-output grammar constraints are out of MVP scope.",
            )
        updated: list[DllmScheduledRequest] = []
        for req in output.requests:
            if req.request_id in next_blocks_by_request_id:
                block = self._normalize_block(next_blocks_by_request_id[req.request_id])
            else:
                block = req.scheduled_spec_decode_tokens
            updated.append(
                DllmScheduledRequest(
                    request_id=req.request_id,
                    scheduled_spec_decode_tokens=block,
                    num_scheduled_tokens=self.draft_size,
                ),
            )
        return DllmSchedulerOutput(requests=tuple(updated))

    def _normalize_block(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        block = tuple(int(tok) for tok in token_ids)
        self._validate_block(block, field_name="block")
        return block

    def _validate_block(self, block: Sequence[int], *, field_name: str) -> None:
        if len(block) != self.draft_size:
            raise ValueError(
                f"{field_name} length must be draft_size={self.draft_size}, "
                f"got {len(block)}",
            )


__all__ = [
    "DllmRequestState",
    "DllmScheduledRequest",
    "DllmScheduler",
    "DllmSchedulerOutput",
    "DllmWorkerResult",
]
