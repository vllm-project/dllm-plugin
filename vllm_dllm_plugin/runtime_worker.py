# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-facing worker adapter for Phase 4 runtime usage."""

from __future__ import annotations

from typing import Any

from vllm_dllm_plugin.worker import DllmWorker as DllmWorkerHelper
from vllm_dllm_plugin.worker import DllmWorkerStep

try:
    from vllm.v1.outputs import DraftTokenIds
    from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in no-vLLM envs.
    VllmGPUWorker = object  # type: ignore[assignment,misc]
    DraftTokenIds = Any  # type: ignore[assignment]
    _VLLM_AVAILABLE = False


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

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        draft_token_ids = super().take_draft_token_ids()
        if draft_token_ids is None:
            return None
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


__all__ = ["DllmRuntimeWorker"]
