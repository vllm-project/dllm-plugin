# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation hooks for capturing attention and layer tensors."""

try:
    from dllm_plugin.validation_hooks.chunked_attention_hooks import (
        ChunkedAttentionCaptureHarness,
        ChunkedAttentionCheckpoint,
    )

    __all__ = [
        "ChunkedAttentionCaptureHarness",
        "ChunkedAttentionCheckpoint",
    ]
except ModuleNotFoundError:
    pass
