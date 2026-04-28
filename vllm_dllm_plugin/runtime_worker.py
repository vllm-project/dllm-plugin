# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-facing worker adapter for Phase 4 runtime usage."""

from __future__ import annotations

from typing import Any, cast

from vllm_dllm_plugin.config import DRAFT_SIZE, LLADA2_DEFAULT_MASK_TOKEN_ID
from vllm_dllm_plugin.remasking import Llada2DefaultRemaskingPolicy
from vllm_dllm_plugin.worker import DllmWorker as DllmWorkerHelper
from vllm_dllm_plugin.worker import DllmWorkerStep

try:
    from vllm.v1.outputs import DraftTokenIds
    from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in no-vLLM envs.
    VllmGPUWorker = object
    DraftTokenIds = Any
    _VLLM_AVAILABLE = False


def build_mock_block_logits(
    *,
    input_draft: list[int],
    sampled_token_ids: list[int],
    draft_size: int = DRAFT_SIZE,
) -> list[list[float]]:
    """Create deterministic block logits for remask handoff in mock mode."""

    if len(input_draft) != draft_size:
        raise ValueError(
            "input_draft length must equal draft_size for mock logits: "
            f"got {len(input_draft)} vs {draft_size}",
        )
    max_seen = max(
        [LLADA2_DEFAULT_MASK_TOKEN_ID, *input_draft, *sampled_token_ids],
    )
    vocab_size = max(max_seen + 2, 256)
    rows: list[list[float]] = []
    for i in range(draft_size):
        if i < len(sampled_token_ids):
            target_id = sampled_token_ids[i]
        else:
            draft_tok = input_draft[i]
            target_id = 0 if draft_tok == LLADA2_DEFAULT_MASK_TOKEN_ID else draft_tok
        row = [0.0] * vocab_size
        row[int(target_id)] = 1.0
        rows.append(row)
    return rows


def run_block_contract_from_model_output(
    *,
    helper: DllmWorkerHelper,
    request_id: str,
    input_draft: list[int],
    sampled_token_ids: list[int],
) -> DllmWorkerStep:
    """Apply one helper-level remask step using mock logits."""

    logits = build_mock_block_logits(
        input_draft=input_draft,
        sampled_token_ids=sampled_token_ids,
        draft_size=helper.draft_size,
    )
    return helper.run_one_block(
        request_id=request_id,
        input_draft=input_draft,
        logits=logits,
        policy=Llada2DefaultRemaskingPolicy(),
    )


def validate_runtime_input_draft(
    *,
    request_id: str,
    input_draft: list[int] | None,
    draft_size: int,
) -> list[int]:
    """Fail-fast validation for scheduler-provided draft blocks."""

    if input_draft is None:
        raise ValueError(
            "missing scheduled_spec_decode_tokens for request in runtime worker: "
            f"request_id={request_id!r}",
        )
    if len(input_draft) != draft_size:
        raise ValueError(
            "malformed scheduled_spec_decode_tokens length in runtime worker: "
            f"request_id={request_id!r} expected={draft_size} got={len(input_draft)}",
        )
    return input_draft


def validate_runtime_draft_handoff_coverage(
    *,
    expected_req_ids: set[str],
    produced_req_ids: list[str],
) -> None:
    """Ensure next-step draft handoff covers exactly the expected requests."""

    seen: set[str] = set()
    duplicate: set[str] = set()
    for req_id in produced_req_ids:
        if req_id in seen:
            duplicate.add(req_id)
        seen.add(req_id)
    if duplicate:
        raise ValueError(
            "duplicate request_id values in runtime draft handoff: "
            f"{sorted(duplicate)}",
        )
    missing = sorted(expected_req_ids.difference(seen))
    if missing:
        raise ValueError(
            f"missing request_id values in runtime draft handoff: {missing}",
        )
    unexpected = sorted(seen.difference(expected_req_ids))
    if unexpected:
        raise ValueError(
            f"unexpected request_id values in runtime draft handoff: {unexpected}",
        )


class DllmRuntimeWorker(VllmGPUWorker):
    """Runtime worker adapter meant for CLI ``--worker-cls`` usage."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "DllmRuntimeWorker requires vLLM. Install with "
                "`uv sync --group dev --extra vllm`.",
            )
        super().__init__(*args, **kwargs)
        # Reuse helper to keep one source of truth for v2 requirement and draft
        # block shape validations.
        self._dllm_helper = DllmWorkerHelper(require_v2_model_runner=True)
        self._dllm_last_draft_token_ids: DraftTokenIds | None = None
        self._dllm_expected_draft_req_ids: set[str] | None = None

    def execute_model(self, scheduler_output: Any) -> Any:
        output = super().execute_model(scheduler_output)
        # Only process concrete model outputs from last PP stage.
        if output is None or not hasattr(output, "sampled_token_ids"):
            return output

        expected_req_ids = set(output.req_ids)
        next_req_ids: list[str] = []
        next_blocks: list[list[int]] = []
        for idx, (req_id, sampled_token_ids) in enumerate(
            zip(output.req_ids, output.sampled_token_ids, strict=True),
        ):
            input_draft = validate_runtime_input_draft(
                request_id=req_id,
                input_draft=scheduler_output.scheduled_spec_decode_tokens.get(req_id),
                draft_size=self._dllm_helper.draft_size,
            )
            step = run_block_contract_from_model_output(
                helper=self._dllm_helper,
                request_id=req_id,
                input_draft=list(input_draft),
                sampled_token_ids=list(sampled_token_ids),
            )
            output.sampled_token_ids[idx] = list(step.sampled_token_ids)
            next_req_ids.append(req_id)
            next_blocks.append(list(self._dllm_helper.take_draft_token_ids(step)))

        validate_runtime_draft_handoff_coverage(
            expected_req_ids=expected_req_ids,
            produced_req_ids=next_req_ids,
        )
        self._dllm_expected_draft_req_ids = expected_req_ids
        self._dllm_last_draft_token_ids = (
            cast(Any, DraftTokenIds)(
                req_ids=next_req_ids,
                draft_token_ids=next_blocks,
            )
            if next_req_ids
            else None
        )
        return output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        draft_token_ids = self._dllm_last_draft_token_ids
        if draft_token_ids is not None:
            self._dllm_last_draft_token_ids = None
        else:
            draft_token_ids = super().take_draft_token_ids()
        if draft_token_ids is None:
            return None
        expected_req_ids = self._dllm_expected_draft_req_ids
        if expected_req_ids is not None:
            validate_runtime_draft_handoff_coverage(
                expected_req_ids=expected_req_ids,
                produced_req_ids=list(draft_token_ids.req_ids),
            )
            self._dllm_expected_draft_req_ids = None
        for req_id, next_block in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
            strict=True,
        ):
            self._dllm_helper.take_draft_token_ids(
                DllmWorkerStep(
                    request_id=req_id,
                    sampled_token_ids=(),
                    next_input_block=tuple(next_block),
                ),
            )
        return draft_token_ids


__all__ = [
    "DllmRuntimeWorker",
    "build_mock_block_logits",
    "run_block_contract_from_model_output",
    "validate_runtime_draft_handoff_coverage",
    "validate_runtime_input_draft",
]
