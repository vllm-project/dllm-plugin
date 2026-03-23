"""Smoke tests for package import, version, and plugin registration stub."""

from __future__ import annotations

import pytest

import vllm_dllm_plugin


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
    # When vLLM is present, registration hook should still complete without error.
    vllm_dllm_plugin.register()


def test_entry_point_target_callable() -> None:
    """``dllm`` entry resolves to the same callable as ``register``."""
    from importlib.metadata import entry_points

    eps = tuple(
        entry_points().select(group="vllm.general_plugins", name="dllm"),
    )
    assert len(eps) == 1
    fn = eps[0].load()
    assert callable(fn)
    assert fn is vllm_dllm_plugin.register
