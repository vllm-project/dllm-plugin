# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``RemaskingPolicy`` contract (protocol) for MVP dLLM decode steps.

Invariants align with ``docs/DESIGN_MVP.md`` section 8 (remasking composability)
and use :data:`~dllm_plugin.config.DRAFT_SIZE` for next-block length.
Concrete policies (e.g. LLaDA2.0 default) live in separate modules (issue #7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from dllm_plugin.config import DRAFT_SIZE


@dataclass(frozen=True, slots=True)
class RemaskStepResult:
    """Structured output of a single remasking step.

    Length invariants are **not** enforced here; invalid instances are possible
    until :func:`validate_remask_step_result` runs (call it at the worker or
    policy boundary after ``apply`` returns, or validate inside concrete policies).
    """

    committed_token_ids: tuple[int, ...]
    """Token ids committed this step; length in ``0..DRAFT_SIZE`` (inclusive).

    Maps to **Committed** semantics in ``DESIGN_MVP.md`` section 8 and to
    ``ModelRunnerOutput.sampled_token_ids`` in section 7.
    """

    next_input_block: tuple[int, ...]
    """Next-step **input block** for the scheduler / draft path.

    Must have length exactly :data:`~dllm_plugin.config.DRAFT_SIZE`
    (section 8: ``next_input_block`` / MASK + decoded positions).
    """


@runtime_checkable
class RemaskingPolicy(Protocol):
    """Composable remasking policy after one block forward (``DESIGN_MVP`` section 8).

    **MVP contract (conceptual, section 8):**

    - **Input:** current input block, logits or equivalent scores, optional
      policy knobs (threshold, top-k, etc.).
    - **Output:** committed ids (subset of positions), next input block of length
      ``DRAFT_SIZE``, and (in concrete implementations) internal mask/draft
      state for logging.

    Implementations must not mutate caller-owned sequences unless documented;
    prefer returning new tuples in :class:`RemaskStepResult`.

    With ``typing.runtime_checkable``, ``isinstance(obj, RemaskingPolicy)`` only
    checks that ``apply`` exists and is callable; it does **not** verify
    keyword-only parameters, return types, or behavior. Treat that check as a
    shallow guard, not a substitute for unit tests or static typing against this
    contract.
    """

    def apply(
        self,
        *,
        input_draft: Sequence[int],
        logits: Any | None = None,
        remasking_config: Mapping[str, Any] | None = None,
    ) -> RemaskStepResult:
        """Run one remasking step for the current block.

        Args:
            input_draft: This step's **input block** (length typically
                ``DRAFT_SIZE`` for decode), aligned with
                ``SchedulerOutput.scheduled_spec_decode_tokens``
                (``DESIGN_MVP`` section 7).
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


def validate_remask_step_result(
    result: RemaskStepResult,
    *,
    draft_size: int = DRAFT_SIZE,
) -> None:
    """Assert MVP shape constraints against ``draft_size``.

    Compare :data:`~dllm_plugin.config.DRAFT_SIZE`. For per-model or
    per-request block sizes, this helper would need an explicit length argument or
    a different entry point; calling it when the live block size differs from
    ``config.DRAFT_SIZE`` would be incorrect (MVP assumes one global draft size).
    """

    if draft_size <= 0:
        raise ValueError(f"draft_size must be positive, got {draft_size}")
    if len(result.next_input_block) != draft_size:
        msg = (
            f"next_input_block must have length draft_size={draft_size}, "
            f"got {len(result.next_input_block)}"
        )
        raise ValueError(msg)
    if not 0 <= len(result.committed_token_ids) <= draft_size:
        msg = (
            "committed_token_ids length must be in "
            f"0..draft_size (inclusive), got {len(result.committed_token_ids)}"
        )
        raise ValueError(msg)
