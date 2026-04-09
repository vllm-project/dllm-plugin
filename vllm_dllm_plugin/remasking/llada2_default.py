# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MVP default remasking policy for LLaDA2.0 (milestone issue #7).

Uses per-position **argmax** token and **softmax confidence** (probability mass on
that token) after a **numerically stable** softmax. Positions with confidence
below a threshold are **remasked** in ``next_input_block`` using ``mask_token_id``.

**``remasking_config`` keys** (optional; stable for issue #13 / worker wiring):

- ``commit_confidence_threshold`` (``float``): minimum softmax probability on the
  argmax token to commit (default
  :data:`~vllm_dllm_plugin.config.LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD`).
- ``mask_token_id`` (``int``): filler for remasked positions in ``next_input_block``
  (default :data:`~vllm_dllm_plugin.config.LLADA2_DEFAULT_MASK_TOKEN_ID`).

**Logits:** 2-D, shape ``(DRAFT_SIZE, vocab_size)``. Accepts nested sequences or
indexable row-major objects (e.g. ``torch.Tensor``) without importing ``torch``
in this module. Callers must not pass ``logits is None`` (non-last PP ranks return
``None`` from the mock — the worker must not invoke the policy in that case).

**``committed_token_ids``:** argmax token id for each **committed** position, in
**increasing position order** (index ``0 .. DRAFT_SIZE-1``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_dllm_plugin.config import (
    DRAFT_SIZE,
    LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD,
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
    """Return argmax index and softmax probability at that index (stable softmax)."""
    if not logits_row:
        raise ValueError("logits row must be non-empty")
    m = max(logits_row)
    exps = [math.exp(x - m) for x in logits_row]
    total = sum(exps)
    if total <= 0.0:
        raise ValueError("softmax normalization produced non-positive total")
    probs = [e / total for e in exps]
    best = max(range(len(probs)), key=lambda j: probs[j])
    return best, probs[best]


class Llada2DefaultRemaskingPolicy:
    """LLaDA2.0 MVP default :class:`~vllm_dllm_plugin.remasking.RemaskingPolicy`."""

    def apply(
        self,
        *,
        input_block: Sequence[int],
        logits: Any | None = None,
        remasking_config: Mapping[str, Any] | None = None,
    ) -> RemaskStepResult:
        if len(input_block) != DRAFT_SIZE:
            msg = (
                f"input_block length must be DRAFT_SIZE={DRAFT_SIZE}, "
                f"got {len(input_block)}"
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

        rows = _logits_to_rows(logits)
        committed: list[int] = []
        next_tokens: list[int] = []
        for row in rows:
            token_id, conf = _argmax_and_max_softmax_prob(row)
            if conf >= threshold:
                committed.append(token_id)
                next_tokens.append(token_id)
            else:
                next_tokens.append(mask_token_id)

        result = RemaskStepResult(
            committed_token_ids=tuple(committed),
            next_input_block=tuple(next_tokens),
        )
        validate_remask_step_result(result)
        return result
