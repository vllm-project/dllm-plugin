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

### v1 vs v2 model runner (mock-stack path)

| Runner | Mock-stack support | Notes |
|--------|-------------------|--------|
| **v2** (`VLLM_USE_V2_MODEL_RUNNER=1`) | **Supported** | Required for the Phase 6 integration test, CI (`vllm-extra`), and operator docs. Adapters are written against vLLM's **v2 model-runner** hooks (`DllmRuntimeWorker` subclasses vLLM's worker on that stack). |
| **v1** (`VLLM_USE_V2_MODEL_RUNNER=0` or unset) | **Unsupported** | Runtime adapters target the **v2 model-runner** stack only. Expect incorrect scheduler/worker pairing or runtime failures rather than silent corruption; do not use for mock-stack validation. |

There is **no supported fallback** from v2 to v1 for the mock-stack path: if your environment cannot enable the v2 model runner, treat the mock-stack integration test and operator workflow as **not applicable** until v2 is available—do not expect partial correctness on v1.

Keep `VLLM_ENABLE_V1_MULTIPROCESSING=0` for the documented integration test to avoid multiprocessing differences on single-process bring-up.

### Strict stack validation toggle

Issue [#4](https://github.com/vllm-project/dllm-plugin/issues/4) validation is **on by default**. To disable it when `strict` is left unset (e.g. temporary debugging), set:

```bash
export VLLM_DLLM_STRICT_STACK_VALIDATION=0
```

Prefer fixing scheduler/worker/model wiring; disabling checks can mask misconfiguration. Explicit `strict=True` / `strict=False` in code overrides the environment.

## Required runtime wiring

Use runtime adapters via CLI overrides:

```bash
vllm serve <model> \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker
```

Short aliases (`dllm_plugin.Scheduler` / `dllm_plugin.Worker`) match `DllmRuntimeScheduler` / `DllmRuntimeWorker`. vLLM class resolution expects **dotted** names (`module.Class`), not `module:Class`.

Strict stack validation (`dllm_plugin.validation.assert_compatible_stack`)
fails fast when scheduler/worker/model architecture combinations are incompatible.

Checks resolve scheduler and worker types and compare them to the **concrete**
adapter classes (`DllmRuntimeScheduler`, `DllmRuntimeWorker`) by fully-qualified
name. **Subclasses** of those adapters are rejected until validation is relaxed
or extended—by design for MVP mock-stack gatekeeping; forks should adjust
validation if they introduce subclassed workers/schedulers.

## First block initialization

- The scheduler initializes `Request.spec_token_ids` for new requests when empty.
- Block size is global: `dllm_plugin.config.DRAFT_SIZE`.
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

- **CPU / GitHub-hosted CI:** `tests/test_vllm_mock_integration_cpu_smoke.py` runs in the
  default `vllm-extra` job: it builds `EngineArgs`, calls `create_engine_config`, resolves
  scheduler/worker classes, and runs `assert_compatible_stack` (no CUDA required). This
  catches upstream API or qualname drift even when the GPU test is skipped.
- **Non-GPU hosts:** the full `LLM.generate` test in `test_vllm_mock_integration.py` is
  skipped (`requires CUDA GPU`).
- **GPU hosts:** that test executes one end-to-end step through vLLM runtime objects with
  plugin scheduler/worker adapters and mock model config.
- **Depth:** the GPU test asserts `LLM.generate` returns token ids; it does **not**
  check remasking block shape, `dllm_block_logits` consumption, or other remask
  invariants—tighter assertions are optional follow-ups beyond this Phase 6 smoke.

For PR/release evidence, include GPU job status plus a persisted log query
(`gcloud logging read ... labels.\"k8s-pod/job-name\"=\"<job-name>\"`) so the
integration result remains auditable after pod cleanup.

## Helm GPU job (`tools/helm/dllm-plugin-gpu-test`)

The chart is a **template**: default `values.yaml` uses **empty** `scheduling.nodeSelector`
and `scheduling.extraTolerations` so the Job schedules anywhere GPU capacity exists.
**You must fork** and set selectors/tolerations to match **your** cluster’s accelerator
labels (see `tools/helm/dllm-plugin-gpu-test/README.md`). Do not assume organization-specific
keys from another fleet apply to yours.

## Notes

- This guide covers **mock-stack** MVP only (Phases 2-6), not real LLaDA2 weights.
- Structured-output grammar rewriting remains explicitly rejected for dLLM block
  mode in this MVP path.
