# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward output to :class:`~dllm_plugin.remasking.RemaskingPolicy` (issue #13).

After one block forward on the **last** pipeline-parallel rank, ``compute_logits``
(see :mod:`dllm_plugin.models.mock_llada2` and ``docs/MOCK_STACK_MODEL.md``)
must yield a **2-D** tensor of shape ``(DRAFT_SIZE, vocab_size)`` (or an
equivalent nested sequence). Row ``i`` aligns with ``input_draft[i]``.

**Out of scope for this MVP helper:** a leading batch dimension (e.g.
``[batch, DRAFT_SIZE, vocab]``). Issue #10 may slice per request or add a
batch-aware wrapper.

Non-last PP ranks return ``logits is None``; do **not** call this module in that
case—there is nothing to remask until the final stage.

See ``docs/DESIGN_MVP.md`` §5–8 and ``docs/CONTRACTS.md`` (forward → remasking).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dllm_plugin.config import DRAFT_SIZE
from dllm_plugin.remasking.base import (
    RemaskingPolicy,
    RemaskStepResult,
    validate_remask_step_result,
)


def assert_block_logits_shape(logits: Any, *, draft_size: int = DRAFT_SIZE) -> None:
    """Require 2-D block logits with first dimension ``draft_size``.

    Accepts ``torch.Tensor`` (via ``shape``) or a length-``draft_size`` sequence
    of per-position rows without importing ``torch``.

    For nested sequences, only the **outer** length is checked here; consistent
    per-row vocabulary width is enforced by the concrete policy (e.g.
    :class:`~dllm_plugin.remasking.Llada2DefaultRemaskingPolicy` via
    ``_logits_to_rows``), not by this helper.
    """
    if logits is None:
        msg = "logits is None (unexpected after caller guard)"
        raise ValueError(msg)
    if hasattr(logits, "shape"):
        shape = logits.shape
        ndim = len(shape)
        if ndim != 2:
            msg = (
                "block logits must be 2-D with shape (DRAFT_SIZE, vocab_size); "
                f"got ndim={ndim} shape={tuple(int(x) for x in shape)}"
            )
            raise ValueError(msg)
        n0 = int(shape[0])
        if n0 != draft_size:
            msg = f"logits first dimension must be draft_size={draft_size}, got {n0}"
            raise ValueError(msg)
        return
    n_pos = len(logits)
    if n_pos != draft_size:
        msg = f"logits first dimension must be draft_size={draft_size}, got {n_pos}"
        raise ValueError(msg)


def remask_after_block_forward(
    *,
    input_draft: Sequence[int],
    logits: Any,
    policy: RemaskingPolicy,
    remasking_config: Mapping[str, Any] | None = None,
    draft_size: int = DRAFT_SIZE,
) -> RemaskStepResult:
    """Run one remasking step after a single-block forward (milestone issue #13).

    Intended call site: ``DllmWorker`` (issue #10) after last-rank
    ``compute_logits``, before mapping
    :class:`~dllm_plugin.remasking.RemaskStepResult` into
    ``ModelRunnerOutput.sampled_token_ids`` and the draft return path
    (``docs/DESIGN_MVP.md`` §6–7).

    Args:
        input_draft: This step's draft, length ``draft_size`` (aligned with
            ``scheduled_spec_decode_tokens``).
        logits: Non-``None`` block logits, 2-D ``(draft_size, vocab_size)``.
        policy: Concrete :class:`~dllm_plugin.remasking.RemaskingPolicy`
            (e.g. :class:`~dllm_plugin.remasking.Llada2DefaultRemaskingPolicy`
            for the LLaDA2 MVP). Callers must supply this; there is no default
            implementation in this helper.
        remasking_config: Optional knobs forwarded to ``policy.apply``.

    Returns:
        Validated :class:`~dllm_plugin.remasking.RemaskStepResult`.

    Raises:
        ValueError: If ``input_draft`` length is wrong, ``logits`` is ``None``,
            or logits fail :func:`assert_block_logits_shape`.
    """
    if draft_size <= 0:
        raise ValueError(f"draft_size must be positive, got {draft_size}")
    if len(input_draft) != draft_size:
        msg = (
            f"input_draft length must be draft_size={draft_size}, "
            f"got {len(input_draft)}"
        )
        raise ValueError(msg)
    if logits is None:
        msg = (
            "logits is None; remasking applies only on the last pipeline-parallel "
            "rank where compute_logits returns a tensor (see docs/MOCK_STACK_MODEL.md)."
        )
        raise ValueError(msg)
    assert_block_logits_shape(logits, draft_size=draft_size)
    result = policy.apply(
        input_draft=input_draft,
        logits=logits,
        remasking_config=remasking_config,
    )
    validate_remask_step_result(result, draft_size=draft_size)
    return result


__all__ = [
    "assert_block_logits_shape",
    "remask_after_block_forward",
]
