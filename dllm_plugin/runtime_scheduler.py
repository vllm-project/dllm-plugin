# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-facing scheduler adapter for Phase 4 runtime usage.

This adapter keeps the existing helper scheduler logic as the source of
contract checks while inheriting from the upstream vLLM scheduler class so the
class is usable as ``--scheduler-cls`` when ``vllm`` is installed.
"""

from __future__ import annotations

from typing import Any

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

    def add_request(self, request: Any) -> None:
        """Ensure first-step dLLM draft block is initialized for new requests."""

        super().add_request(request)
        live_req = self.requests.get(request.request_id)
        if live_req is None or live_req.spec_token_ids:
            return
        prompt = live_req.prompt_token_ids or []
        live_req.spec_token_ids = list(
            self._dllm_helper.initialize_first_block(prompt_token_ids=prompt),
        )

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        self._fail_if_structured_output_active(draft_token_ids.req_ids)
        self._validate_draft_lengths(draft_token_ids)
        super().update_draft_token_ids(draft_token_ids)

    def update_draft_token_ids_in_output(
        self,
        draft_token_ids: DraftTokenIds,
        scheduler_output: Any,
    ) -> None:
        self._fail_if_structured_output_active(draft_token_ids.req_ids)
        self._validate_draft_lengths(draft_token_ids)
        super().update_draft_token_ids_in_output(draft_token_ids, scheduler_output)

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

    def _fail_if_structured_output_active(self, req_ids: list[str]) -> None:
        for req_id in req_ids:
            request = self.requests.get(req_id)
            if request is None:
                continue
            if self.structured_output_manager.should_advance(request):
                raise ValueError(
                    "structured-output grammar rewriting is not supported for dLLM "
                    f"block mode (request_id={req_id!r})",
                )


__all__ = ["DllmRuntimeScheduler", "validate_scheduler_worker_contract"]
