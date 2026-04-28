# Operator guide (mock stack, Phases 5-6)

This guide describes the MVP **mock-stack** operator path for `vllm-dllm-plugin`
with strict stack validation and the Phase 6 integration test.

## Prerequisites

- Linux/CUDA environment with working GPU (Phase 6 integration test is GPU-gated).
- Plugin repo checked out and synced with vLLM extra:

```bash
uv sync --group dev --extra vllm
```

- vLLM compatibility/minimum tested range follows `pyproject.toml`
  (`vllm>=0.14.0,<0.15`); track pin and hook confidence updates via issue `#2`.

- vLLM plugin loading enabled:

```bash
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
```

## Required runtime wiring

Use runtime adapters via CLI overrides:

```bash
vllm serve <model> \
  --scheduler-cls vllm_dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
  --worker-cls vllm_dllm_plugin.runtime_worker.DllmRuntimeWorker
```

Strict stack validation (`vllm_dllm_plugin.validation.assert_compatible_stack`)
fails fast when scheduler/worker/model architecture combinations are incompatible.

## First block initialization

- The scheduler initializes `Request.spec_token_ids` for new requests when empty.
- Block size is global: `vllm_dllm_plugin.config.DRAFT_SIZE`.
- Override block size consistently across scheduler/worker/remasking by setting
  `VLLM_DLLM_DRAFT_SIZE` **before importing** plugin modules.

## Integration test (mock stack)

Run the concrete vLLM runtime integration test:

```bash
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
uv run pytest -q tests/test_vllm_mock_integration.py
```

Expected behavior:

- Non-GPU hosts: test is skipped (`requires CUDA GPU`).
- GPU hosts: test executes one end-to-end step through vLLM runtime objects with
  plugin scheduler/worker adapters and mock model config.

For PR/release evidence, include GPU job status plus a persisted log query
(`gcloud logging read ... labels.\"k8s-pod/job-name\"=\"<job-name>\"`) so the
integration result remains auditable after pod cleanup.

## Notes

- This guide covers **mock-stack** MVP only (Phases 2-6), not real LLaDA2 weights.
- Structured-output grammar rewriting remains explicitly rejected for dLLM block
  mode in this MVP path.
