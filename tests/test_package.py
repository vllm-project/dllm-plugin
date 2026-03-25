# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for package import, version, and plugin registration stub."""

from __future__ import annotations

import sys

import pytest

import vllm_dllm_plugin


def _dllm_plugin_entry_points():
    """Return ``dllm`` entry points in group ``vllm.general_plugins``."""
    from importlib.metadata import entry_points

    if sys.version_info >= (3, 13):
        return tuple(entry_points(group="vllm.general_plugins", name="dllm"))
    return tuple(
        entry_points().select(group="vllm.general_plugins", name="dllm"),
    )


def test_version() -> None:
    assert vllm_dllm_plugin.__version__
    assert isinstance(vllm_dllm_plugin.__version__, str)


def test_register_does_not_raise() -> None:
    vllm_dllm_plugin.register()


@pytest.mark.parametrize("attr", ("register", "__version__"))
def test_public_api(attr: str) -> None:
    assert hasattr(vllm_dllm_plugin, attr)


def test_register_with_vllm_if_installed() -> None:
    pytest.importorskip("vllm")
    # With vLLM importable, the entry point must still complete without error.
    vllm_dllm_plugin.register()


def test_register_debug_stub_when_vllm_present(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pytest.importorskip("vllm")
    import logging

    caplog.set_level(logging.DEBUG, logger="vllm_dllm_plugin")
    vllm_dllm_plugin.register()
    assert "skeleton" in caplog.text.lower()


def test_entry_point_resolves_dllm() -> None:
    """``dllm`` entry point loads and targets ``vllm_dllm_plugin:register``.

    Expects exactly one provider named ``dllm`` in ``vllm.general_plugins`` (normal
    install). Unusual environments with multiple distributions registering the same
    name are out of scope for this test.
    """
    eps = _dllm_plugin_entry_points()
    assert len(eps) == 1
    ep = eps[0]
    assert ep.value == "vllm_dllm_plugin:register"
    fn = ep.load()
    assert callable(fn)
    fn()  # smoke: same behavior as register() for the stub
