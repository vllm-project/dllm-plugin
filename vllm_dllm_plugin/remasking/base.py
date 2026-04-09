# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``RemaskingPolicy`` contract (protocol) for MVP dLLM decode steps.

Invariants align with ``docs/DESIGN_MVP.md`` section 8 (remasking composability)
and use :data:`~vllm_dllm_plugin.config.DRAFT_SIZE` for next-block length.
Concrete policies (e.g. LLaDA2.0 default) live in separate modules (issue #7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from vllm_dllm_plugin.config import DRAFT_SIZE


@dataclass(frozen=True, slots=True)
class RemaskStepResult:
    """Structured output of a single remasking step."""

    committed_token_ids: tuple[int, ...]
    """Token ids committed this step; length in ``0..DRAFT_SIZE`` (inclusive).

    Maps to **Committed** semantics in ``DESIGN_MVP.md`` section 8 and to
    ``ModelRunnerOutput.sampled_token_ids`` in section 7.
    """

    next_input_block: tuple[int, ...]
    """Next-step **input block** for the scheduler / draft path.

    Must have length exactly :data:`~vllm_dllm_plugin.config.DRAFT_SIZE`
    (section 8: ``next_input_block`` / MASK + decoded positions).
    """


@runtime_checkable
class RemaskingPolicy(Protocol):
    """Composable remasking policy after one block forward (``DESIGN_MVP`` §8).

    **MVP contract (conceptual, section 8):**

    - **Input:** current input block, logits or equivalent scores, optional
      policy knobs (threshold, top-k, etc.).
    - **Output:** committed ids (subset of positions), next input block of length
      ``DRAFT_SIZE``, and (in concrete implementations) internal mask/draft
      state for logging.

    Implementations must not mutate caller-owned sequences unless documented;
    prefer returning new tuples in :class:`RemaskStepResult`.
    """

    def apply(
        self,
        *,
        input_block: Sequence[int],
        logits: Any | None = None,
        remasking_config: Mapping[str, Any] | None = None,
    ) -> RemaskStepResult:
        """Run one remasking step for the current block.

        Args:
            input_block: This step's **input block** (length typically
                ``DRAFT_SIZE`` for decode), aligned with
                ``SchedulerOutput.scheduled_spec_decode_tokens`` (``DESIGN_MVP`` §7).
            logits: Model output for the block (tensor or equivalent); optional
                for testing or stub forwards.
            remasking_config: Optional policy parameters (e.g. thresholds).

        Returns:
            Committed token ids and the fixed-length next input block.

        Raises:
            ValueError: If ``next_input_block`` would not have length
                ``DRAFT_SIZE`` (recommended for concrete implementations).
        """
        ...


def validate_remask_step_result(result: RemaskStepResult) -> None:
    """Assert MVP shape constraints; for use by implementations and tests."""

    if len(result.next_input_block) != DRAFT_SIZE:
        msg = (
            f"next_input_block must have length DRAFT_SIZE={DRAFT_SIZE}, "
            f"got {len(result.next_input_block)}"
        )
        raise ValueError(msg)
    if not 0 <= len(result.committed_token_ids) <= DRAFT_SIZE:
        msg = (
            "committed_token_ids length must be in "
            f"0..DRAFT_SIZE (inclusive), got {len(result.committed_token_ids)}"
        )
        raise ValueError(msg)
