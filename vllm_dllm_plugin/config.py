# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MVP configuration: block size, model identifiers, and feature flags.

Canonical defaults for the MVP LLaDA2.0 stack live here so scheduler, worker,
remasking, and tests share one source of truth (see ``docs/DESIGN_MVP.md``).
"""

from __future__ import annotations

from typing import Final

#: Fixed diffusion / spec-decode **block size** for one plugin step (tokens).
#: LLaDA2.0 MVP uses 32 per ``docs/DESIGN_MVP.md`` (Goals table, section 1).
DRAFT_SIZE: Final[int] = 32

#: Primary registered architecture key for the real LLaDA2.0 vLLM model module
#: (HF mapping). Until Phase 7 (#12), registration points at the **mock** class
#: (see ``docs/MOCK_STACK_MODEL.md``); HF configs using this name get stub logits.
#: Prefer ``DLLM_MOCK_STACK_MODEL_ID`` when you want an explicit test-only id.
#: Exact registry string may be refined when ``register()`` lands (issue #5).
LLADA2_ARCHITECTURE_NAME: Final[str] = "LLaDA2ForCausalLM"

#: Registered model id for the **mock / stub** forward used in Phases 2-6 stack
#: testing (deterministic outputs; see milestone issue #24).
DLLM_MOCK_STACK_MODEL_ID: Final[str] = "DllmMockLlada2StackTest"

#: Lazy import target for :func:`register_dllm` (``<module>:<Class>``).
DLLM_MOCK_MODEL_CLASS_FQCN: Final[str] = (
    "vllm_dllm_plugin.models.mock_llada2:DllmMockLlada2ForCausalLM"
)

#: When ``True``, ``validation.py`` (issue #4) should treat incompatible
#: scheduler / worker / model combinations as errors once that module exists.
#: Operators or tests may override via future config wiring; this is the default.
DLLM_STRICT_STACK_VALIDATION_DEFAULT: Final[bool] = True

#: Placeholder **mask** token id for :mod:`~vllm_dllm_plugin.remasking.llada2_default`
#: ``next_input_block`` remasked positions until real HF config lands (Phase 7 / #12).
LLADA2_DEFAULT_MASK_TOKEN_ID: Final[int] = 1

#: Default number of denoise steps used to build the per-step **transfer count**
#: schedule (``block_len // steps`` layout). Matches ``DRAFT_SIZE`` for one transfer
#: per step when the schedule is not overridden.
LLADA2_DEFAULT_DENOISE_STEPS: Final[int] = DRAFT_SIZE

#: Default minimum softmax probability on the per-position argmax token required to
#: **commit** that position (issue #7). Tuned so the Phase 2 mock stub logits
#: (zeros + ``1.0`` at index ``0``, ``docs/MOCK_STACK_MODEL.md``) commit under
#: default settings for stack tests.
LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD: Final[float] = 0.01
