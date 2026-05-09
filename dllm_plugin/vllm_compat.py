# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM compatibility layer - version-aware imports.

Centralizes all vLLM version-dependent imports to avoid scattered try/except chains.
Requires vllm>=0.20.0 (enforced by pyproject.toml).

**Version Policy:**
- Minimum: vllm>=0.20.0 (v2 model runner API)
- Maximum: <0.21 (upper bound for safety)
- Pre-0.20 fallback imports removed in Phase 7

**Usage:**
Instead of scattered try/except fallback chains:
```python
# DON'T DO THIS (old pattern):
try:
    from vllm.model_executor.layers.attention import Attention
except ImportError:
    try:
        from vllm.model_executor.layers.attention.layer import Attention
    except ImportError:
        from vllm.attention.layer import Attention
```

Use centralized imports:
```python
# DO THIS (new pattern):
from dllm_plugin.vllm_compat import Attention
```

This module handles vLLM version compatibility so plugin code doesn't have to.
"""

from __future__ import annotations

# Core config types
from vllm.config import VllmConfig

# Attention layer types (vLLM 0.20+ paths)
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import CommonAttentionMetadata

# Scheduler/worker output types (vLLM 0.20+ v2 API)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

__all__ = [
    "VllmConfig",
    "Attention",
    "CommonAttentionMetadata",
    "SchedulerOutput",
    "ModelRunnerOutput",
    "DraftTokenIds",
]
