# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MVP configuration: block size, model identifiers, and feature flags.

Canonical defaults for the MVP LLaDA2.0 stack live here so scheduler, worker,
remasking, and tests share one source of truth (see ``docs/DESIGN_MVP.md``).
"""

from __future__ import annotations

import os
from typing import Final

#: Environment variable controlling global dLLM block size for the plugin stack.
DLLM_DRAFT_SIZE_ENV_VAR: Final[str] = "VLLM_DLLM_DRAFT_SIZE"


def _read_draft_size() -> int:
    raw = os.environ.get(DLLM_DRAFT_SIZE_ENV_VAR)
    if raw is None:
        return 32
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DLLM_DRAFT_SIZE_ENV_VAR} must be an integer, got {raw!r}",
        ) from exc
    if value <= 0:
        raise ValueError(f"{DLLM_DRAFT_SIZE_ENV_VAR} must be positive, got {value}")
    return value


#: Fixed diffusion / spec-decode **block size** for one plugin step (tokens).
#: Defaults to 32 for LLaDA2.0 MVP and can be overridden via
#: ``VLLM_DLLM_DRAFT_SIZE`` so scheduler/worker/remasking share one value.
DRAFT_SIZE: Final[int] = _read_draft_size()

#: Primary registered architecture key for the real LLaDA2.0 vLLM model module
#: (HF mapping). Until Phase 7 (#12), registration points at the **mock** class
#: (see ``docs/MOCK_STACK_MODEL.md``); HF configs using this name get stub logits.
#: Prefer ``DLLM_MOCK_STACK_MODEL_ID`` when you want an explicit test-only id.
#: Exact registry string may be refined when ``register()`` lands (issue #5).
LLADA2_ARCHITECTURE_NAME: Final[str] = "LLaDA2ForCausalLM"

#: HuggingFace architecture name used by inclusionAI/LLaDA2.0-mini model config.
#: Registered alongside LLADA2_ARCHITECTURE_NAME to support both naming conventions.
LLADA2_HF_ARCHITECTURE_NAME: Final[str] = "LLaDA2MoeModelLM"

#: Registered model id for the **mock / stub** forward used in Phases 2-6 stack
#: testing (deterministic outputs; see milestone issue #24).
DLLM_MOCK_STACK_MODEL_ID: Final[str] = "DllmMockLlada2StackTest"

#: Lazy import target for :func:`register_dllm` (``<module>:<Class>``).
DLLM_MOCK_MODEL_CLASS_FQCN: Final[str] = (
    "dllm_plugin.models.mock_llada2:DllmMockLlada2ForCausalLM"
)

#: When set, overrides default strictness for :func:`resolve_strict_stack_validation`
#: when ``explicit`` is ``None``. Values: ``1``/``true``/``yes``/``on`` (strict),
#: ``0``/``false``/``no``/``off`` (skip validation). Unset means strict.
DLLM_STRICT_STACK_VALIDATION_ENV_VAR: Final[str] = "VLLM_DLLM_STRICT_STACK_VALIDATION"

#: When truthy, ``register_dllm`` may apply the EngineCore PR **#36391** string
#: patch at plugin load (``vllm serve`` / engine). See
#: ``dllm_plugin.engine_core_draft_hook`` and
#: ``VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH``.
DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK_ENV_VAR: Final[str] = (
    "VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK"
)


def _read_strict_stack_validation_from_env() -> bool:
    raw = os.environ.get(DLLM_STRICT_STACK_VALIDATION_ENV_VAR)
    if raw is None or raw.strip() == "":
        return True
    lowered = raw.strip().lower()
    if lowered in {"0", "false", "no", "off"}:
        return False
    if lowered in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{DLLM_STRICT_STACK_VALIDATION_ENV_VAR} must be one of "
        f"0/false/no/off or 1/true/yes/on; got {raw!r}",
    )


def resolve_strict_stack_validation(explicit: bool | None) -> bool:
    """Return whether strict stack validation should run.

    Callers pass ``explicit`` when they need a definite on/off regardless of
    environment (tests, imperative overrides). When ``explicit`` is ``None``,
    read :data:`DLLM_STRICT_STACK_VALIDATION_ENV_VAR` (default: strict on).
    """

    if explicit is not None:
        return explicit
    return _read_strict_stack_validation_from_env()


#: Placeholder **mask** token id for :mod:`~dllm_plugin.remasking.llada2_default`
#: ``next_input_block`` remasked positions until real HF config lands (Phase 7 / #12).
LLADA2_DEFAULT_MASK_TOKEN_ID: Final[int] = 156895

#: Default number of denoise steps used to build the per-step **transfer count**
#: schedule (``block_len // steps`` layout). Matches configured ``DRAFT_SIZE`` for
#: one transfer per step when the schedule is not overridden.
LLADA2_DEFAULT_DENOISE_STEPS: Final[int] = DRAFT_SIZE

#: Default minimum softmax probability on the per-position argmax token required to
#: **commit** that position (issue #7). Tuned so the Phase 2 mock stub logits
#: (zeros + ``1.0`` at index ``0``, ``docs/MOCK_STACK_MODEL.md``) commit under
#: default settings for stack tests.
LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD: Final[float] = 0.9

#: Gumbel-max temperature for stochastic sampling.  ``0.0`` = deterministic
#: argmax (dInfer LLaDA2.0 default).  ``> 0`` enables Gumbel-max sampling
#: with float64 precision per arXiv:2409.02908.
LLADA2_DEFAULT_TEMPERATURE: Final[float] = 0.0

#: Use float64 precision for softmax confidence computation.
LLADA2_DEFAULT_USE_FLOAT64: Final[bool] = False

# Phase 7: Real LLaDA2.0 model configuration (issue #12)

#: Lazy import target for real LLaDA2.0 vLLM model (``<module>:<Class>``).
#: Phase 7 adds production-ready model with MoE weight loading and
#: block-style attention.
LLADA2_REAL_MODEL_CLASS_FQCN: Final[str] = "dllm_plugin.models.llada2:LLaDA2ForCausalLM"

#: Default number of experts per MoE layer (from HuggingFace LLaDA2.0 config).
LLADA2_DEFAULT_NUM_EXPERTS: Final[int] = 256

#: Default number of experts activated per token (top-k routing).
LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK: Final[int] = 8

#: Default number of shared experts (always active, not routed).
LLADA2_DEFAULT_NUM_SHARED_EXPERTS: Final[int] = 1

#: Default MoE intermediate size (FFN hidden dimension per expert).
LLADA2_DEFAULT_MOE_INTERMEDIATE_SIZE: Final[int] = 512

#: Default number of expert groups for group-limited routing.
#: LLaDA2.0 uses 8 groups for two-stage expert selection.
LLADA2_DEFAULT_N_GROUP: Final[int] = 8

#: Default number of groups to select in group-limited routing.
#: First selects top-4 groups from 8, then top-k experts from selected groups.
LLADA2_DEFAULT_TOPK_GROUP: Final[int] = 4

#: Default scaling factor applied to routed expert output.
#: LLaDA2.0 uses 2.5x scaling on routed experts before adding shared expert output.
LLADA2_DEFAULT_ROUTED_SCALING_FACTOR: Final[float] = 2.5
