# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test helper utilities for numerical validation.

These helpers require torch and are only importable in GPU test environments.
Import them inside test functions that use pytest.importorskip("torch").
"""

try:
    from .tensor_extraction import (
        TensorExtractor,
        extract_hf_embeddings,
        extract_hf_intermediate_activations,
        extract_hf_logits,
        extract_vllm_embeddings,
        extract_vllm_intermediate_activations,
        extract_vllm_logits,
    )

    __all__ = [
        "TensorExtractor",
        "extract_hf_embeddings",
        "extract_hf_logits",
        "extract_hf_intermediate_activations",
        "extract_vllm_embeddings",
        "extract_vllm_logits",
        "extract_vllm_intermediate_activations",
    ]
except ModuleNotFoundError:
    pass
