# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-facing scheduler adapter for Phase 4 runtime usage.

This adapter keeps the existing helper scheduler logic as the source of
contract checks while inheriting from the upstream vLLM scheduler class so the
class is usable as ``--scheduler-cls`` when ``vllm`` is installed.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from dllm_plugin.grammar_utils import (
    flat_frontier_bitmask_row_index,
    frontier_block_row,
    scheduled_spec_decode_tokens_for_grammar_bitmask,
)
from dllm_plugin.scheduler import (
    DllmRequestState,
    DllmWorkerResult,
)
from dllm_plugin.scheduler import (
    DllmScheduler as DllmSchedulerHelper,
)
from dllm_plugin.validation import assert_compatible_stack

try:
    from vllm.v1.core.sched.scheduler import Scheduler as VllmScheduler
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in no-vLLM envs.
    VllmScheduler = object
    DraftTokenIds = Any
    ModelRunnerOutput = Any
    _VLLM_AVAILABLE = False


def validate_scheduler_worker_contract(
    *,
    helper: DllmSchedulerHelper,
    expected_req_ids: tuple[str, ...],
    model_runner_output: Any,
) -> None:
    """Apply helper-level scheduler output validation to runtime outputs."""

    helper_states = {
        req_id: DllmRequestState(
            request_id=req_id,
            # Use draft_size so helper accounting checks can run.
            num_computed_tokens=helper.draft_size,
        )
        for req_id in expected_req_ids
    }
    helper_results = tuple(
        DllmWorkerResult(
            request_id=req_id,
            sampled_token_ids=tuple(sampled_token_ids),
        )
        for req_id, sampled_token_ids in zip(
            model_runner_output.req_ids,
            model_runner_output.sampled_token_ids,
            strict=True,
        )
    )
    helper.update_from_output(states=helper_states, worker_results=helper_results)


class DllmRuntimeScheduler(VllmScheduler):
    """Runtime scheduler adapter meant for CLI ``--scheduler-cls`` usage."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "DllmRuntimeScheduler requires vLLM. Install with "
                "`uv sync --group dev --extra vllm`.",
            )
        super().__init__(*args, **kwargs)
        assert_compatible_stack(
            self.vllm_config,
            caller="DllmRuntimeScheduler.__init__",
        )
        self._dllm_helper = DllmSchedulerHelper()

    def schedule(self) -> Any:
        """Attach precomputed grammar bitmask metadata for dLLM workers."""

        out = super().schedule()
        # Frontier repair metadata for dLLM structured outputs. Consumed in phase two
        # by :class:`~dllm_plugin.gpu_model_runner.DllmGPUModelRunner` (stashed from
        # ``SchedulerOutput`` in ``execute_model``). ``GrammarOutput`` still arrives via
        # ``sample_tokens`` from the engine; these fields are not a second grammar path.
        out.dllm_grammar_output = None
        out.dllm_so_frontier_flat_indices = None
        out.dllm_so_frontier_block_rows = None
        out.dllm_so_valid_prefix_lens = None
        if out.has_structured_output_requests:
            patched = scheduled_spec_decode_tokens_for_grammar_bitmask(
                scheduled_spec_decode_tokens=out.scheduled_spec_decode_tokens,
                requests=self.requests,
                structured_output_manager=self.structured_output_manager,
            )
            grammar_output = super().get_grammar_bitmask(
                replace(out, scheduled_spec_decode_tokens=patched),
            )
            out.dllm_grammar_output = grammar_output
            if grammar_output is not None:
                out.dllm_so_frontier_flat_indices = flat_frontier_bitmask_row_index(
                    structured_output_request_ids=grammar_output.structured_output_request_ids,
                    patched_scheduled_spec_decode_tokens=patched,
                )
                rows: dict[str, int | None] = {}
                lens: dict[str, int] = {}
                for req_id in grammar_output.structured_output_request_ids:
                    prefix = patched.get(req_id, ())
                    lens[req_id] = len(prefix)
                    rows[req_id] = frontier_block_row(
                        valid_prefix_len=len(prefix),
                        draft_size=self._dllm_helper.draft_size,
                    )
                out.dllm_so_frontier_block_rows = rows
                out.dllm_so_valid_prefix_lens = lens
        return out

    def get_grammar_bitmask(self, scheduler_output: Any) -> Any:
        """Use grammar-valid draft prefixes so bitmask rows match frontier repair."""

        patched = scheduled_spec_decode_tokens_for_grammar_bitmask(
            scheduled_spec_decode_tokens=scheduler_output.scheduled_spec_decode_tokens,
            requests=self.requests,
            structured_output_manager=self.structured_output_manager,
        )
        return super().get_grammar_bitmask(
            replace(scheduler_output, scheduled_spec_decode_tokens=patched),
        )

    def add_request(self, request: Any) -> None:
        """Ensure first-step dLLM draft block is initialized for new requests."""

        super().add_request(request)
        # Test-only: skip first-block seed for GPU grammar tests (see OPERATOR doc).
        if os.environ.get("VLLM_DLLM_SKIP_FIRST_BLOCK_SEED") == "1":
            return
        live_req = self.requests.get(request.request_id)
        if live_req is None or live_req.spec_token_ids:
            return
        prompt = live_req.prompt_token_ids or []
        live_req.spec_token_ids = list(
            self._dllm_helper.initialize_first_block(prompt_token_ids=prompt),
        )

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        """Keep fixed ``DRAFT_SIZE`` blocks; do not grammar-truncate drafts here."""

        self._validate_draft_lengths(draft_token_ids)
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
            strict=True,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue

            if request.is_prefill_chunk:
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            request.spec_token_ids = list(spec_token_ids)

    def update_draft_token_ids_in_output(
        self,
        draft_token_ids: DraftTokenIds,
        scheduler_output: Any,
    ) -> None:
        """Refresh deferred scheduler drafts without grammar shortening / -1 padding."""

        # Always empty: dLLM blocks do not populate AR spec-decode invalid-token maps.
        # If upstream starts relying on this field for dLLM-shaped batches, revisit.
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
            strict=True,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            row = list(spec_token_ids)
            del row[orig_num_spec_tokens:]
            # Do not call grammar.validate_tokens here — preserve worker block shape.
            if len(row) < orig_num_spec_tokens:
                row.extend([-1] * (orig_num_spec_tokens - len(row)))
            sched_spec_tokens[req_id] = row

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def update_from_output(
        self,
        scheduler_output: Any,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, Any]:
        """Validate dLLM scheduler-worker contract before delegating upstream."""

        validate_scheduler_worker_contract(
            helper=self._dllm_helper,
            expected_req_ids=tuple(scheduler_output.num_scheduled_tokens.keys()),
            model_runner_output=model_runner_output,
        )
        return super().update_from_output(scheduler_output, model_runner_output)

    def _validate_draft_lengths(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
            strict=True,
        ):
            if len(token_ids) != self._dllm_helper.draft_size:
                raise ValueError(
                    "draft token block length mismatch in DllmRuntimeScheduler: "
                    f"request_id={req_id!r} expected={self._dllm_helper.draft_size} "
                    f"got={len(token_ids)}",
                )


__all__ = ["DllmRuntimeScheduler", "validate_scheduler_worker_contract"]
