# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-facing scheduler adapter for Phase 4 runtime usage.

This adapter keeps the existing helper scheduler logic as the source of
contract checks while inheriting from the upstream vLLM scheduler class so the
class is usable as ``--scheduler-cls`` when ``vllm`` is installed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from dllm_plugin.config import LLADA2_DEFAULT_MASK_TOKEN_ID
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

# vLLM imports (centralized in vllm_compat for version handling)
from dllm_plugin.vllm_compat import DraftTokenIds, ModelRunnerOutput

try:
    from vllm.v1.core.sched.scheduler import Scheduler as VllmScheduler

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in no-vLLM envs.
    # VllmScheduler is the only import that may fail in test environments
    VllmScheduler = object
    _VLLM_AVAILABLE = False


def validate_scheduler_worker_contract(
    *,
    helper: DllmSchedulerHelper,
    expected_req_ids: tuple[str, ...],
    model_runner_output: ModelRunnerOutput,
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
    req_ids = model_runner_output.req_ids
    sampled = model_runner_output.sampled_token_ids
    helper_results = tuple(
        DllmWorkerResult(
            request_id=req_id,
            sampled_token_ids=tuple(sampled_token_ids),
        )
        for req_id, sampled_token_ids in zip(req_ids, sampled, strict=True)
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
        """Attach grammar bitmask metadata for dLLM workers."""
        # Capture num_computed_tokens BEFORE super().schedule() advances them.
        # Used by update_from_output() for Commit-0 rollback.
        # Only capture for requests that have COMPLETED prefill — rolling back
        # before prefill completes would re-process prompt tokens on every step.
        self._pre_schedule_nct: dict[str, int] = {}
        for req_id, request in self.requests.items():
            if getattr(request, "spec_token_ids", None):
                prompt_len = getattr(request, "num_prompt_tokens", 0)
                if request.num_computed_tokens >= prompt_len > 0:
                    self._pre_schedule_nct[req_id] = request.num_computed_tokens

        out = super().schedule()

        # Note: spec_token_ids population and negative num_scheduled_tokens
        # filtering are handled by the vLLM fork's scheduler.

        # Frontier repair metadata for dLLM structured outputs.
        out.dllm_grammar_output = None
        out.dllm_so_frontier_flat_indices = None
        out.dllm_so_frontier_block_rows = None
        out.dllm_so_valid_prefix_lens = None

        # Extract num_prefix_tokens for virtual batch attention (Phase 7)
        # Maps request_id -> num_computed_tokens for all scheduled requests
        out.dllm_num_prefix_tokens = {}
        if out.num_scheduled_tokens:
            for req_id in out.num_scheduled_tokens:
                request = self.requests.get(req_id)
                if request and hasattr(request, "dllm_state"):
                    # Prefix = prompt tokens + previously committed blocks.
                    # For the first block, this is the prompt length.
                    # For subsequent blocks, it grows by committed tokens.
                    prompt_len = getattr(request, "num_prompt_tokens", 0)
                    committed = request.dllm_state.num_computed_tokens
                    out.dllm_num_prefix_tokens[req_id] = prompt_len + committed

        # FIX B: Workaround for vLLM 0.20.1 not reliably setting
        # has_structured_output_requests under high concurrency.
        # Explicitly check all requests for structured output usage.
        if not out.has_structured_output_requests:
            for req in self.requests.values():
                if getattr(req, "use_structured_output", False):
                    out.has_structured_output_requests = True
                    break

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
        """Initialize dLLM state and seed the first block."""
        super().add_request(request)

        if not hasattr(request, "dllm_state"):
            request.dllm_state = DllmRequestState(
                request_id=request.request_id,
                num_computed_tokens=0,
            )

        prompt_ids = getattr(request, "prompt_token_ids", [])
        if prompt_ids and (
            not hasattr(request, "spec_token_ids") or not request.spec_token_ids
        ):
            draft_size = self._dllm_helper.draft_size
            mask_id = LLADA2_DEFAULT_MASK_TOKEN_ID
            prompt_in_block = len(prompt_ids) % draft_size
            if prompt_in_block == 0 and len(prompt_ids) > 0:
                prompt_in_block = draft_size
            prompt_tail = list(prompt_ids[-prompt_in_block:])
            n_masks = draft_size - len(prompt_tail)
            first_block = prompt_tail + [mask_id] * n_masks
            request.spec_token_ids = first_block

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        """Keep fixed ``DRAFT_SIZE`` blocks; do not grammar-truncate drafts here.

        Unlike AR spec-decode, dLLM always sets spec_token_ids regardless of
        prefill state — the draft block is needed from the first decode step.
        """
        self._validate_draft_lengths(draft_token_ids)
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
            strict=True,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
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

    def make_spec_decoding_stats(self, *args, **kwargs) -> Any:
        """Skip spec decode metrics for dLLM to avoid assertion failures.

        The parent vLLM spec decode metrics assume traditional spec decode
        where drafted tokens <= accepted tokens per step. dLLM uses fixed
        32-token draft blocks that may exceed max_tokens, which violates
        the parent metrics' assertions. We skip metrics collection for now.
        """
        # Return None to skip metrics (metrics are optional)
        return None

    def update_from_output(
        self,
        scheduler_output: Any,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, Any]:
        """Validate dLLM contract, strip Commit-0 padding, delegate upstream."""

        validate_scheduler_worker_contract(
            helper=self._dllm_helper,
            expected_req_ids=tuple(scheduler_output.num_scheduled_tokens.keys()),
            model_runner_output=model_runner_output,
        )

        # Strip padding from sampled_token_ids: the model runner produces
        # fixed-width rows (slot_width) padded with -1. The parent scheduler
        # interprets the full list as accepted tokens. Filter to only
        # non-negative, non-zero values for committed blocks, and empty
        # lists for Commit-0 (denoising in progress).
        if model_runner_output.sampled_token_ids:
            cleaned: list[list[int]] = []
            for row in model_runner_output.sampled_token_ids:
                tokens = [int(t) for t in row if int(t) != -1]
                cleaned.append(tokens)
            model_runner_output.sampled_token_ids = cleaned

        result = super().update_from_output(scheduler_output, model_runner_output)

        # Commit-0 rollback: schedule() advanced num_computed_tokens by
        # num_scheduled_tokens. For Commit-0 (no tokens committed), restore
        # the pre-schedule value so the same block gets re-scheduled.
        pre_nct = getattr(self, "_pre_schedule_nct", {})
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is None:
                continue
            req_index = model_runner_output.req_id_to_index.get(req_id)
            if req_index is None:
                continue
            generated = (
                model_runner_output.sampled_token_ids[req_index]
                if model_runner_output.sampled_token_ids
                else []
            )
            if not generated and req_id in pre_nct:
                request.num_computed_tokens = pre_nct[req_id]

        # Update dllm_state.num_computed_tokens for committed blocks.
        # This counter tracks the committed prefix length for virtual batch
        # attention and is NOT rolled back on Commit-0 — re-running the same
        # block doesn't change the prefix length.
        if model_runner_output.sampled_token_ids:
            for req_id, token_ids in zip(
                model_runner_output.req_ids,
                model_runner_output.sampled_token_ids,
                strict=True,
            ):
                request = self.requests.get(req_id)
                if request and hasattr(request, "dllm_state") and len(token_ids) > 0:
                    request.dllm_state.num_computed_tokens += len(token_ids)

        # WORKAROUND: override num_computed_tokens using our own accounting.
        # The parent scheduler's update_from_output() assumes bonus_tokens=1
        # when counting accepted spec tokens, causing an off-by-one for
        # diffusion models (bonus=0). The proper fix is to have the parent
        # read ModelState.num_bonus_tokens; until then, we correct it here.
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is None:
                continue
            if hasattr(request, "dllm_state"):
                prompt_len = getattr(request, "num_prompt_tokens", 0)
                committed = request.dllm_state.num_computed_tokens
                correct_nct = prompt_len + committed
                request.num_computed_tokens = correct_nct

        return result

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
