# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MVP default remasking policy for LLaDA2.0 (milestone issue #7).

Per-position **argmax** token and **softmax probability** at that token (stable
softmax). Only **mask** positions participate in confidence-based transfer; decoded
positions are preserved. Each step uses a **transfer count** from a schedule over
``denoise_steps`` (or a ``num_transfer`` override). If enough masked positions meet
``commit_confidence_threshold`` (**inclusive** ``>=``), all of them transfer; otherwise
the top-``k`` masked positions by confidence transfer (``k`` capped by mask count),
with **smallest index** winning ties.

While the output draft still contains ``mask_token_id``, ``committed_token_ids`` is
empty and progress is carried in ``next_input_block``. When no mask remains, this
step returns the full decoded block as ``committed_token_ids`` and sets
``next_input_block`` to an all-mask draft for the following block handoff.

**``remasking_config`` keys** (optional; stable for issue #13 / worker wiring):

- ``commit_confidence_threshold`` (``float``): minimum softmax probability on the
  argmax token for a masked position to count as high-confidence (default
  :data:`~vllm_dllm_plugin.config.LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD`).
- ``mask_token_id`` (``int``): mask placeholder in drafts (default
  :data:`~vllm_dllm_plugin.config.LLADA2_DEFAULT_MASK_TOKEN_ID`).
- ``denoise_steps`` (``int``): schedule length; default
  :data:`~vllm_dllm_plugin.config.LLADA2_DEFAULT_DENOISE_STEPS`.
- ``denoise_step_index`` (``int``): zero-based index into the schedule; default ``0``.
  Production callers should pass the real step.
- ``num_transfer`` (``int``): if set, use this count instead of the schedule entry.

**Logits:** 2-D, shape ``(DRAFT_SIZE, vocab_size)``. Accepts nested sequences or
indexable row-major objects (e.g. ``torch.Tensor``) without importing ``torch``
in this module. Callers must not pass ``logits is None`` when logits are required.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_dllm_plugin.config import (
    DRAFT_SIZE,
    LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD,
    LLADA2_DEFAULT_DENOISE_STEPS,
    LLADA2_DEFAULT_MASK_TOKEN_ID,
)
from vllm_dllm_plugin.remasking.base import (
    RemaskStepResult,
    validate_remask_step_result,
)


def _scalar_float(x: Any) -> float:
    return float(x.item()) if hasattr(x, "item") else float(x)


def _row(logits: Any, i: int) -> Any:
    return logits[i]


def _row_length(row: Any) -> int:
    if hasattr(row, "shape"):
        shape = row.shape
        return int(shape[0]) if len(shape) == 1 else int(shape[-1])
    return len(row)


def _cell(row: Any, j: int) -> Any:
    return row[j]


def _logits_to_rows(logits: Any) -> list[list[float]]:
    """Normalize 2-D logits to a list of float rows; shape ``(DRAFT_SIZE, vocab)``."""
    n_pos = len(logits)
    if n_pos != DRAFT_SIZE:
        msg = f"logits first dimension must be DRAFT_SIZE={DRAFT_SIZE}, got {n_pos}"
        raise ValueError(msg)
    rows: list[list[float]] = []
    vocab: int | None = None
    for i in range(DRAFT_SIZE):
        row = _row(logits, i)
        vlen = _row_length(row)
        if vlen == 0:
            raise ValueError("logits row must be non-empty (vocab_size >= 1)")
        if vocab is None:
            vocab = vlen
        elif vlen != vocab:
            msg = f"logits rows have inconsistent vocab size: {vocab} vs {vlen}"
            raise ValueError(msg)
        rows.append([_scalar_float(_cell(row, j)) for j in range(vlen)])
    return rows


def _argmax_and_max_softmax_prob(logits_row: Sequence[float]) -> tuple[int, float]:
    """Return argmax index and softmax probability at that index (stable softmax).

    On logit ties, the **smallest** index among maxima wins (matches common
    ``argmax`` conventions and ``torch.argmax`` on tied values).
    """
    if not logits_row:
        raise ValueError("logits row must be non-empty")
    m = max(logits_row)
    best = min(j for j, x in enumerate(logits_row) if x == m)
    exps = [math.exp(x - m) for x in logits_row]
    total = sum(exps)
    if total <= 0.0:
        raise ValueError("softmax normalization produced non-positive total")
    probs = [e / total for e in exps]
    return best, probs[best]


def _num_transfer_schedule(block_len: int, steps: int) -> tuple[int, ...]:
    if steps <= 0:
        msg = "denoise_steps must be positive"
        raise ValueError(msg)
    base = block_len // steps
    remainder = block_len % steps
    out = [base] * steps
    for i in range(remainder):
        out[i] += 1
    return tuple(out)


def _topk_masked_indices(
    *,
    masked: list[bool],
    confidence: list[float],
    k: int,
) -> list[int]:
    """Return ``k`` masked indices with highest confidence; ties -> smaller index."""
    if k <= 0:
        return []
    candidates = [i for i in range(DRAFT_SIZE) if masked[i]]
    candidates.sort(key=lambda i: (-confidence[i], i))
    return candidates[:k]


class Llada2DefaultRemaskingPolicy:
    """LLaDA2.0 MVP default :class:`~vllm_dllm_plugin.remasking.RemaskingPolicy`."""

    def apply(
        self,
        *,
        input_draft: Sequence[int],
        logits: Any | None = None,
        remasking_config: Mapping[str, Any] | None = None,
    ) -> RemaskStepResult:
        if len(input_draft) != DRAFT_SIZE:
            msg = (
                f"input_draft length must be DRAFT_SIZE={DRAFT_SIZE}, "
                f"got {len(input_draft)}"
            )
            raise ValueError(msg)
        if logits is None:
            raise ValueError(
                "logits is required for Llada2DefaultRemaskingPolicy.apply "
                "(callers must not pass None for non-last PP / missing logits)"
            )
        cfg = dict(remasking_config) if remasking_config else {}
        threshold = float(
            cfg.get(
                "commit_confidence_threshold",
                LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD,
            ),
        )
        mask_token_id = int(cfg.get("mask_token_id", LLADA2_DEFAULT_MASK_TOKEN_ID))
        denoise_steps = int(cfg.get("denoise_steps", LLADA2_DEFAULT_DENOISE_STEPS))
        denoise_step_index = int(cfg.get("denoise_step_index", 0))

        rows = _logits_to_rows(logits)
        masked = [input_draft[i] == mask_token_id for i in range(DRAFT_SIZE)]

        if not any(masked):
            next_tokens = list(input_draft)
            result = RemaskStepResult(
                committed_token_ids=tuple(next_tokens),
                next_input_block=(mask_token_id,) * DRAFT_SIZE,
            )
            validate_remask_step_result(result)
            return result

        token_ids: list[int] = []
        probs: list[float] = []
        for row in rows:
            tid, p = _argmax_and_max_softmax_prob(row)
            token_ids.append(tid)
            probs.append(p)

        neg_inf = float("-inf")
        confidence = [probs[i] if masked[i] else neg_inf for i in range(DRAFT_SIZE)]
        high_conf = [
            masked[i] and (confidence[i] >= threshold) for i in range(DRAFT_SIZE)
        ]
        h = sum(high_conf)

        if "num_transfer" in cfg:
            num_to_transfer = int(cfg["num_transfer"])
            if num_to_transfer < 0:
                msg = "num_transfer must be non-negative"
                raise ValueError(msg)
        else:
            schedule = _num_transfer_schedule(DRAFT_SIZE, denoise_steps)
            step = max(0, min(denoise_step_index, len(schedule) - 1))
            num_to_transfer = schedule[step]

        num_masked = sum(masked)
        transfer_idx: list[int]
        if h >= num_to_transfer:
            transfer_idx = [i for i in range(DRAFT_SIZE) if high_conf[i]]
        else:
            k = min(num_to_transfer, num_masked)
            transfer_idx = _topk_masked_indices(
                masked=masked,
                confidence=confidence,
                k=k,
            )

        next_tokens = list(input_draft)
        for i in transfer_idx:
            next_tokens[i] = token_ids[i]

        if mask_token_id not in next_tokens:
            result = RemaskStepResult(
                committed_token_ids=tuple(next_tokens),
                next_input_block=(mask_token_id,) * DRAFT_SIZE,
            )
        else:
            result = RemaskStepResult(
                committed_token_ids=(),
                next_input_block=tuple(next_tokens),
            )
        validate_remask_step_result(result)
        return result
