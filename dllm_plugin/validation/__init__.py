"""Validation utilities for dllm-plugin.

This package provides validation hooks and utilities. Note that there's also
a validation.py MODULE at the parent level (dllm_plugin/validation.py) which
contains stack validation functions. We re-export those here to maintain
backward compatibility.
"""

import importlib.util
from pathlib import Path

# Import from this package's submodules
from dllm_plugin.validation.chunked_attention_hooks import (
    CHECKPOINT_DEFINITIONS,
    CHECKPOINT_TOLERANCES,
    ChunkedAttentionCaptureHarness,
    ChunkedAttentionCheckpoint,
)

# Load the validation.py MODULE (sibling to this package directory)
# This is necessary because Python prioritizes the package over the module
# when both exist with the same name
_validation_module_path = Path(__file__).parent.parent / "validation.py"
_spec = importlib.util.spec_from_file_location(
    "_validation_module", _validation_module_path
)
assert _spec is not None
_validation_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_validation_module)

# Re-export functions from validation.py module
assert_compatible_stack = _validation_module.assert_compatible_stack
assert_runtime_worker_v2_model_runner = (
    _validation_module.assert_runtime_worker_v2_model_runner
)


__all__ = [
    "ChunkedAttentionCheckpoint",
    "ChunkedAttentionCaptureHarness",
    "CHECKPOINT_DEFINITIONS",
    "CHECKPOINT_TOLERANCES",
    "assert_compatible_stack",
    "assert_runtime_worker_v2_model_runner",
]
