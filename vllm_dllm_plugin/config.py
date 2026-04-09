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
#: (HF mapping). Exact registry string may be refined when ``register()`` lands
#: (see milestone issue #5).
LLADA2_ARCHITECTURE_NAME: Final[str] = "LLaDA2ForCausalLM"

#: Registered model id for the **mock / stub** forward used in Phases 2-6 stack
#: testing (deterministic outputs; see milestone issue #24).
DLLM_MOCK_STACK_MODEL_ID: Final[str] = "DllmMockLlada2StackTest"

#: When ``True``, ``validation.py`` (issue #4) should treat incompatible
#: scheduler / worker / model combinations as errors once that module exists.
#: Operators or tests may override via future config wiring; this is the default.
DLLM_STRICT_STACK_VALIDATION_DEFAULT: Final[bool] = True
