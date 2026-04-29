# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for package import, version, and plugin registration."""

from __future__ import annotations

import sys

import pytest

import dllm_plugin

_EXPECTED_EP_VALUE = "dllm_plugin:register_dllm"


def _dllm_plugin_entry_points():
    """Return ``dllm`` entry points in group ``vllm.general_plugins``."""
    from importlib.metadata import entry_points

    if sys.version_info >= (3, 13):
        return tuple(entry_points(group="vllm.general_plugins", name="dllm"))
    return tuple(
        entry_points().select(group="vllm.general_plugins", name="dllm"),
    )


def test_version() -> None:
    assert dllm_plugin.__version__
    assert isinstance(dllm_plugin.__version__, str)


def test_register_dllm_does_not_raise() -> None:
    dllm_plugin.register_dllm()


@pytest.mark.parametrize(
    "attr",
    (
        "register_dllm",
        "__version__",
        "DllmScheduler",
        "DllmWorker",
        "DllmRuntimeScheduler",
        "DllmRuntimeWorker",
        "Scheduler",
        "Worker",
        "assert_compatible_stack",
    ),
)
def test_public_api(attr: str) -> None:
    assert hasattr(dllm_plugin, attr)


def test_scheduler_worker_aliases_are_runtime_classes() -> None:
    assert dllm_plugin.Scheduler is dllm_plugin.DllmRuntimeScheduler
    assert dllm_plugin.Worker is dllm_plugin.DllmRuntimeWorker


def test_register_with_vllm_if_installed() -> None:
    pytest.importorskip("vllm")
    # With vLLM importable, the entry point must still complete without error.
    dllm_plugin.register_dllm()


def test_register_dllm_registers_architectures_when_vllm_present() -> None:
    pytest.importorskip("vllm")
    from vllm import ModelRegistry

    from dllm_plugin.config import (
        DLLM_MOCK_STACK_MODEL_ID,
        LLADA2_ARCHITECTURE_NAME,
    )

    dllm_plugin.register_dllm()
    archs = ModelRegistry.get_supported_archs()
    assert LLADA2_ARCHITECTURE_NAME in archs
    assert DLLM_MOCK_STACK_MODEL_ID in archs


def test_mock_model_class_importable_when_vllm_present() -> None:
    pytest.importorskip("vllm")
    import torch.nn as nn

    from dllm_plugin.models.mock_llada2 import DllmMockLlada2ForCausalLM

    assert issubclass(DllmMockLlada2ForCausalLM, nn.Module)


def test_entry_point_resolves_dllm() -> None:
    """``dllm`` entry point loads and targets ``dllm_plugin:register_dllm``.

    Allows multiple distributions to expose the same name; requires at least one
    entry whose ``value`` matches this package’s target, and that all ``dllm``
    entries agree on ``value`` when several exist.
    """
    eps = _dllm_plugin_entry_points()
    assert len(eps) >= 1, "expected at least one vllm.general_plugins entry named dllm"
    matching = [ep for ep in eps if ep.value == _EXPECTED_EP_VALUE]
    assert matching, (
        f"no dllm entry point with value {_EXPECTED_EP_VALUE!r}; got "
        f"{[ep.value for ep in eps]!r}"
    )
    values = {ep.value for ep in eps}
    assert len(values) == 1, (
        "multiple dllm entry-point values in one environment — "
        f"expected a single consistent target, got {values!r}"
    )
    ep = matching[0]
    fn = ep.load()
    assert callable(fn)
    fn()  # smoke: same behavior as register_dllm() for the stub
