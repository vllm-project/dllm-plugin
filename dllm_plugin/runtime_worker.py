# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-facing worker adapter for Phase 4 runtime usage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from dllm_plugin.config import DLLM_MOCK_STACK_MODEL_ID, DRAFT_SIZE
from dllm_plugin.remasking import Llada2DefaultRemaskingPolicy
from dllm_plugin.validation import (
    assert_compatible_stack,
    assert_runtime_worker_v2_model_runner,
)
from dllm_plugin.worker import DllmWorker, DllmWorkerStep

_MISSING = object()

try:
    from vllm.tracing import instrument
except ImportError:  # pragma: no cover
    # ``vllm`` not installed (e.g. CI without ``--extra vllm``): no-op decorator.
    def instrument(
        obj: Any | None = None,
        *,
        span_name: str = "",
        attributes: dict[str, str] | None = None,
        record_exception: bool = True,
    ) -> Any:
        if obj is None:

            def _partial(fn: Any) -> Any:
                return fn

            return _partial
        return obj


try:
    from vllm.v1.outputs import DraftTokenIds
    from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in no-vLLM envs.
    VllmGPUWorker = object
    DraftTokenIds = Any
    _VLLM_AVAILABLE = False


def build_mock_model_block_logits(
    *,
    draft_size: int,
    vocab_size: int,
) -> list[list[float]]:
    """Create deterministic mock-model logits rows (id=0 score is highest)."""

    if draft_size <= 0:
        raise ValueError(f"draft_size must be positive, got {draft_size}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    rows: list[list[float]] = []
    for _ in range(draft_size):
        row = [0.0] * vocab_size
        row[0] = 1.0
        rows.append(row)
    return rows


def _normalize_block_logits_rows(*, logits: Any, draft_size: int) -> list[list[float]]:
    """Normalize and validate score rows for one remask block.

    Phase 6 mock-stack: Python ``float`` rows are fine for small vocabs. Phase 7
    (real-model logits, large vocabs) should avoid eager full-row materialization here.
    """

    if len(logits) != draft_size:
        raise ValueError(
            "malformed model score rows for runtime remask handoff: "
            f"expected {draft_size} rows, got {len(logits)}",
        )
    normalized: list[list[float]] = []
    vocab_size: int | None = None
    for row in logits:
        row_list = [
            float(value.item()) if hasattr(value, "item") else float(value)
            for value in row
        ]
        if not row_list:
            raise ValueError("runtime remask score row must be non-empty")
        if vocab_size is None:
            vocab_size = len(row_list)
        elif len(row_list) != vocab_size:
            raise ValueError(
                "runtime remask score rows have inconsistent vocab size: "
                f"{vocab_size} vs {len(row_list)}",
            )
        normalized.append(row_list)
    return normalized


def _resolve_output_logits_by_req_id(
    *,
    model_output: Any,
    request_id: str,
    request_index: int,
) -> tuple[Any | None, bool]:
    """Extract per-request block logits from model output when available."""

    raw = getattr(model_output, "dllm_block_logits", _MISSING)
    if raw is _MISSING or raw is None:
        return None, False
    if isinstance(raw, Mapping):
        raw_mapping = cast(Mapping[str, Any], raw)
        if request_id not in raw_mapping:
            raise ValueError(
                "runtime remask handoff missing request coverage in "
                "dllm_block_logits mapping: "
                f"request_id={request_id!r}",
            )
        return raw_mapping[request_id], True
    try:
        return cast(Any, raw)[request_index], True
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            "runtime remask handoff cannot resolve logits row payload for "
            f"request_id={request_id!r} request_index={request_index}",
        ) from exc


def _is_mock_stack_architecture(vllm_config: Any) -> bool:
    hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    if hf_config is None:
        return False
    archs = getattr(hf_config, "architectures", ()) or ()
    if isinstance(archs, str):
        archs = (archs,)
    names = {str(item) for item in archs}
    return DLLM_MOCK_STACK_MODEL_ID in names


def resolve_runtime_block_logits(
    *,
    model_output: Any,
    request_id: str,
    request_index: int,
    draft_size: int = DRAFT_SIZE,
    vllm_config: Any,
) -> list[list[float]]:
    """Resolve block logits/scores for runtime remask handoff."""

    raw_logits, has_logits_payload = _resolve_output_logits_by_req_id(
        model_output=model_output,
        request_id=request_id,
        request_index=request_index,
    )
    if raw_logits is not None:
        return _normalize_block_logits_rows(logits=raw_logits, draft_size=draft_size)
    if has_logits_payload:
        raise ValueError(
            "runtime remask handoff received unusable logits payload for "
            f"request_id={request_id!r}",
        )

    if _is_mock_stack_architecture(vllm_config):
        hf_config = getattr(
            getattr(vllm_config, "model_config", None),
            "hf_config",
            None,
        )
        vocab_size = int(getattr(hf_config, "vocab_size", 256))
        return build_mock_model_block_logits(
            draft_size=draft_size,
            vocab_size=vocab_size,
        )

    raise ValueError(
        "runtime remask handoff requires model score rows in model output "
        f"(missing dllm_block_logits for request_id={request_id!r})",
    )


def run_block_contract_from_model_output(
    *,
    helper: DllmWorker,
    request_id: str,
    input_draft: list[int],
    logits: Any,
    remasking_config: Mapping[str, Any] | None = None,
) -> DllmWorkerStep:
    """Apply one helper-level remask step using model-provided logits."""

    return helper.run_one_block(
        request_id=request_id,
        input_draft=input_draft,
        logits=logits,
        policy=Llada2DefaultRemaskingPolicy(),
        remasking_config=remasking_config,
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
        assert_compatible_stack(self.vllm_config, caller="DllmRuntimeWorker.__init__")
        assert_runtime_worker_v2_model_runner(
            use_v2_model_runner=self.use_v2_model_runner,
            caller="DllmRuntimeWorker.__init__",
        )
        self._dllm_helper = DllmWorker(require_v2_model_runner=True)

    @instrument(span_name="Init device")
    def init_device(self) -> None:
        """Install :class:`~dllm_plugin.gpu_model_runner.DllmGPUModelRunner` for v2."""
        super().init_device()
        if getattr(self, "use_v2_model_runner", False):
            from dllm_plugin.gpu_model_runner import DllmGPUModelRunner

            self.model_runner = DllmGPUModelRunner(self.vllm_config, self.device)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """Prefer dLLM runner hook ``take_dllm_draft_token_ids`` when present.

        Upstream spec decode uses ``model_runner.take_draft_token_ids``; dLLM blocks use
        runner ``take_dllm_draft_token_ids`` when implemented (see gpu_model_runner).
        """
        mr = self.model_runner
        take_dllm = getattr(mr, "take_dllm_draft_token_ids", None)
        if callable(take_dllm):
            draft_token_ids = take_dllm()
            if draft_token_ids is not None:
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
        draft_token_ids = super().take_draft_token_ids()
        return draft_token_ids


__all__ = [
    "DllmRuntimeWorker",
    "build_mock_model_block_logits",
    "resolve_runtime_block_logits",
    "run_block_contract_from_model_output",
    "validate_runtime_draft_handoff_coverage",
    "validate_runtime_input_draft",
]
