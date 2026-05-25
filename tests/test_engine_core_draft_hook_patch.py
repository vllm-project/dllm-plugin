# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for EngineCore draft-hook shim (issue #35, vLLM PR #36391)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import dllm_plugin.engine_core_draft_hook as ech_mod
from tests.support.engine_core_draft_hook import (
    engine_core_draft_hook_patch_needed,
    patch_engine_core_draft_hook_semantics,
)

pytest.importorskip("vllm")

pytestmark = pytest.mark.dllm_engine_patch


def _make_engine_core_stub(
    *, use_spec_decode: bool, async_scheduling: bool = False
) -> MagicMock:
    stub = MagicMock()
    stub.use_spec_decode = use_spec_decode
    stub.async_scheduling = async_scheduling
    stub.model_executor = MagicMock()
    stub.scheduler = MagicMock()
    return stub


@pytest.fixture
def _draft_hook_semantics():
    """Apply PR #36391 semantics when the wheel still uses legacy spec-decode gates."""

    if engine_core_draft_hook_patch_needed():
        with patch_engine_core_draft_hook_semantics():
            yield
    else:
        yield


def test_post_step_calls_hook_without_spec_decode(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=False)
    draft_ids = MagicMock()
    stub.model_executor.take_draft_token_ids.return_value = draft_ids

    EngineCore.post_step(stub, model_executed=True)

    stub.model_executor.take_draft_token_ids.assert_called_once()
    stub.scheduler.update_draft_token_ids.assert_called_once_with(draft_ids)


def test_post_step_noop_when_draft_ids_none(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=False)
    stub.model_executor.take_draft_token_ids.return_value = None

    EngineCore.post_step(stub, model_executed=True)

    stub.model_executor.take_draft_token_ids.assert_called_once()
    stub.scheduler.update_draft_token_ids.assert_not_called()


def test_post_step_noop_when_model_not_executed(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=False)

    EngineCore.post_step(stub, model_executed=False)

    stub.model_executor.take_draft_token_ids.assert_not_called()


def test_post_step_noop_with_async_scheduling(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=False, async_scheduling=True)

    EngineCore.post_step(stub, model_executed=True)

    stub.model_executor.take_draft_token_ids.assert_not_called()


def test_post_step_still_works_with_spec_decode(_draft_hook_semantics) -> None:
    from vllm.v1.engine.core import EngineCore

    stub = _make_engine_core_stub(use_spec_decode=True)
    draft_ids = MagicMock()
    stub.model_executor.take_draft_token_ids.return_value = draft_ids

    EngineCore.post_step(stub, model_executed=True)

    stub.model_executor.take_draft_token_ids.assert_called_once()
    stub.scheduler.update_draft_token_ids.assert_called_once_with(draft_ids)


def test_patch_compiles_when_legacy_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check that source rewrite succeeds for the supported layout."""

    monkeypatch.delenv("VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH", raising=False)
    if not engine_core_draft_hook_patch_needed():
        pytest.skip("installed vLLM already matches PR #36391 post_step layout")

    compile_fn = ech_mod._compile_patched_engine_core_methods  # noqa: SLF001
    post_fn, step_fn = compile_fn()
    assert callable(post_fn)
    assert callable(step_fn)


def test_skip_env_disables_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.engine.core import EngineCore

    monkeypatch.setenv("VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH", "1")
    stub = _make_engine_core_stub(use_spec_decode=False)
    draft_ids = MagicMock()
    stub.model_executor.take_draft_token_ids.return_value = draft_ids

    with patch_engine_core_draft_hook_semantics():
        EngineCore.post_step(stub, model_executed=True)

    if engine_core_draft_hook_patch_needed():
        stub.model_executor.take_draft_token_ids.assert_not_called()
    else:
        stub.model_executor.take_draft_token_ids.assert_called_once()


def test_register_dllm_applies_runtime_patch_when_apply_env_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in env plus ``register_dllm()`` installs the EngineCore runtime patch."""

    from vllm.v1.engine.core import EngineCore

    from dllm_plugin import register_dllm

    monkeypatch.delenv("VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH", raising=False)
    monkeypatch.setenv("VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK", "1")
    if not engine_core_draft_hook_patch_needed():
        pytest.skip("installed vLLM already matches PR #36391 post_step layout")

    # Check if register_dllm still applies the patch (removed when using fork)
    from dllm_plugin import register_dllm as _rd

    if not hasattr(_rd, "_apply_engine_core_draft_hook"):
        pytest.skip(
            "register_dllm no longer applies engine core patch (fork handles it)"
        )

    orig_post = EngineCore.post_step
    orig_step = EngineCore.step_with_batch_queue
    try:
        ech_mod._reset_runtime_patch_applied_for_tests()
        register_dllm()
        assert EngineCore.post_step is not orig_post
        assert EngineCore.step_with_batch_queue is not orig_step
    finally:
        EngineCore.post_step = orig_post
        EngineCore.step_with_batch_queue = orig_step
        ech_mod._reset_runtime_patch_applied_for_tests()


def test_apply_runtime_patch_idempotent_when_legacy_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``apply_engine_core_draft_hook_patch_if_needed`` is safe to call twice."""

    from vllm.v1.engine.core import EngineCore

    monkeypatch.delenv("VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH", raising=False)
    monkeypatch.setenv("VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK", "1")
    if not engine_core_draft_hook_patch_needed():
        pytest.skip("installed vLLM already matches PR #36391 post_step layout")

    orig_post = EngineCore.post_step
    orig_step = EngineCore.step_with_batch_queue
    try:
        ech_mod._reset_runtime_patch_applied_for_tests()
        ech_mod.apply_engine_core_draft_hook_patch_if_needed()
        first_post = EngineCore.post_step
        ech_mod.apply_engine_core_draft_hook_patch_if_needed()
        assert EngineCore.post_step is first_post
    finally:
        EngineCore.post_step = orig_post
        EngineCore.step_with_batch_queue = orig_step
        ech_mod._reset_runtime_patch_applied_for_tests()


def test_apply_runtime_patch_noop_twice_when_upstream_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if engine_core_draft_hook_patch_needed():
        pytest.skip("legacy EngineCore layout required for this assertion")

    monkeypatch.delenv("VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH", raising=False)
    ech_mod._reset_runtime_patch_applied_for_tests()
    ech_mod.apply_engine_core_draft_hook_patch_if_needed()
    ech_mod.apply_engine_core_draft_hook_patch_if_needed()
    ech_mod._reset_runtime_patch_applied_for_tests()
