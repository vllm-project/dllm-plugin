# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured-output frontier helpers for dLLM fixed-size blocks (issue #9).

Aligned with :class:`~vllm.v1.structured_output.StructuredOutputManager` serial
``grammar_bitmask`` semantics: ``scheduled_spec_decode_tokens`` entries passed to
bitmask generation should be **grammar-valid prefixes** produced by
``grammar.validate_tokens``, not raw drafts containing the first invalid tail token.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def valid_prefix_tokens_for_draft(
    *,
    request: Any,
    draft_tokens: Sequence[int],
    structured_output_manager: Any,
) -> list[int]:
    """Return the longest grammar-valid prefix token ids for this draft block."""

    if not getattr(request, "use_structured_output", False):
        return list(draft_tokens)
    if not structured_output_manager.should_advance(request):
        return list(draft_tokens)
    meta = getattr(request, "structured_output_request", None)
    if meta is None or getattr(meta, "grammar", None) is None:
        return list(draft_tokens)
    return meta.grammar.validate_tokens(list(draft_tokens))


def scheduled_spec_decode_tokens_for_grammar_bitmask(
    *,
    scheduled_spec_decode_tokens: dict[str, list[int]],
    requests: dict[str, Any],
    structured_output_manager: Any,
) -> dict[str, list[int]]:
    """Shrink drafts to ``validate_tokens`` prefixes for SO requests only."""

    out = {k: list(v) for k, v in scheduled_spec_decode_tokens.items()}
    for req_id, toks in out.items():
        req = requests.get(req_id)
        if req is None:
            continue
        out[req_id] = valid_prefix_tokens_for_draft(
            request=req,
            draft_tokens=toks,
            structured_output_manager=structured_output_manager,
        )
    return out


def flat_frontier_bitmask_row_index(
    *,
    structured_output_request_ids: list[str],
    patched_scheduled_spec_decode_tokens: dict[str, list[int]],
) -> dict[str, int]:
    """Map req_id -> row index into the stacked ``grammar_bitmask`` tensor."""

    cum = 0
    indices: dict[str, int] = {}
    for req_id in structured_output_request_ids:
        prefix = patched_scheduled_spec_decode_tokens.get(req_id, ())
        L = len(prefix)
        indices[req_id] = cum + L
        cum += L + 1
    return indices


def frontier_block_row(
    *,
    valid_prefix_len: int,
    draft_size: int,
) -> int | None:
    """Block row for frontier masking: ``None`` if every in-block token is valid."""

    if valid_prefix_len >= draft_size:
        return None
    return valid_prefix_len


def apply_packed_bitmask_inplace_logits_row(
    logits_row: list[float],
    bitmask_packed: Sequence[int] | Any,
) -> None:
    """Apply token bitmask to one logits row (matches GPU kernel semantics).

    Packed layout: ``cdiv(vocab_size, 32)`` int32 words; bit 0 means disallow
    (set logit to ``-inf``), matching ``StructuredOutputsWorker`` / xgrammar.
    """

    vocab_size = len(logits_row)
    n_words = (
        len(bitmask_packed)
        if hasattr(bitmask_packed, "__len__") and not isinstance(bitmask_packed, str)
        else 0
    )
    if n_words == 0:
        return
    for word_idx in range(n_words):
        packed = int(bitmask_packed[word_idx])
        for bit in range(32):
            tid = word_idx * 32 + bit
            if tid >= vocab_size:
                break
            if ((packed >> bit) & 1) == 0:
                logits_row[tid] = float("-inf")


def grammar_extra_transfer_slots(
    *,
    draft_tokens: Sequence[int],
    valid_prefix_len: int,
) -> int:
    """Extra decode budget slots suggested when grammar-invalid tail exists."""

    return max(0, len(draft_tokens) - int(valid_prefix_len))
