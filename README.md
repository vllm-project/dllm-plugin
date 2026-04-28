# vllm-dllm-plugin

[![CI](https://github.com/vllm-project/dllm-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/vllm-project/dllm-plugin/actions/workflows/ci.yml)

**vllm-dllm-plugin** is a [vLLM](https://github.com/vllm-project/vllm) plugin for **block-based diffusion language models (dLLMs)**. The package provides a `vllm.general_plugins` entry point (`dllm`), Phase 1 contracts (`config`, `remasking`), a **mock registered model** for stack testing (Phases 2–6), and Phase 4 scheduler/worker helpers that encode block scheduling, commit-0 rollback, draft handoff, and grammar-safety guardrails. Production LLaDA2.0 model logic remains in progress (see [docs/ROADMAP.md](docs/ROADMAP.md)).

**Important:** `register_dllm()` first checks `importlib.util.find_spec("vllm")`; if `vllm` is not discoverable on `sys.path`, it returns without registering. If the spec exists but `from vllm import ModelRegistry` still fails, registration is skipped and a **DEBUG** traceback is logged. When that import succeeds, **`register_dllm()` registers two architecture names** with vLLM’s `ModelRegistry`, both targeting the **mock** in `vllm_dllm_plugin.models.mock_llada2` (not real inference—see [docs/MOCK_STACK_MODEL.md](docs/MOCK_STACK_MODEL.md)). Schedulers and workers are not registered yet.

The approach follows the public RFC discussion [vllm#36155](https://github.com/vllm-project/vllm/issues/36155) and reuses spec-decode-shaped fields so batching and executor paths stay aligned.

**AI-assisted work:** If tools materially helped with your change, disclose that (PR description is the default; commits may carry a short factual note when appropriate). See **AI-assisted contributions** in [CONTRIBUTING.md](CONTRIBUTING.md).

**Optional `vllm` vs bart-style plugins:** Many vLLM plugins (for example [bart-plugin](https://github.com/vllm-project/bart-plugin)) declare **vLLM as a hard install-time dependency**. This repo keeps **`vllm` in an optional extra** on purpose so contributors on macOS or without CUDA can run the default dev hooks and tests (`uv sync --group dev`). Full integration with a real vLLM install remains **`uv sync --group dev --extra vllm`** (and the optional CI workflow).

## Install (development)

Requires Python **3.10–3.13** (metadata: `requires-python = ">=3.10,<3.14"`, aligned with current vLLM expectations) and [uv](https://docs.astral.sh/uv/). CI tests **3.10** through **3.13**. A [`.python-version`](.python-version) file pins the default local interpreter for `uv`/pyenv; override as needed for the matrix.

```bash
git clone https://github.com/vllm-project/dllm-plugin.git
cd dllm-plugin
uv sync --group dev
uv run pre-commit install
```

**vLLM** is an optional extra so environments without CUDA wheels (e.g. many macOS setups) can still sync and run tests:

```bash
# Linux / CUDA-capable environments
uv sync --group dev --extra vllm
```

The optional extra pins **`vllm>=0.14.0,<0.15`** (API-churn guard, bart-plugin-style). **`uv.lock`** resolves a concrete version in that range when you sync with **`--extra vllm`** (see **Lockfile** in [CONTRIBUTING.md](CONTRIBUTING.md)). Widen the bound only after testing (e.g. optional smoke workflow + lock refresh).

**Upstream hook status (Phase 0):** dLLM draft-token hook alignment is tracked in [vllm#36155](https://github.com/vllm-project/vllm/issues/36155). Until the hook is confirmed in a pinned release for this plugin path, treat this bound as a compatibility guard and keep docs + `pyproject.toml` synchronized through issue [#2](https://github.com/vllm-project/dllm-plugin/issues/2).

See [CONTRIBUTING.md](CONTRIBUTING.md) for pre-commit, CI parity, and contribution norms.

## Using the plugin (future)

For MVP stack bring-up, enable the plugin by name and point vLLM at the plugin scheduler and worker classes, for example:

```bash
export VLLM_PLUGINS=dllm
vllm serve <model> \
  --scheduler-cls vllm_dllm_plugin.runtime_scheduler:DllmRuntimeScheduler \
  --worker-cls vllm_dllm_plugin.runtime_worker:DllmRuntimeWorker
```

`DllmRuntimeScheduler` and `DllmRuntimeWorker` are runtime adapter classes intended for CLI overrides. `DllmScheduler` and `DllmWorker` remain helper/contract implementations used by the adapters. MVP expects `VLLM_USE_V2_MODEL_RUNNER=1`; grammar-constrained draft rewriting is intentionally rejected for dLLM block mode to avoid silent block-shape corruption. Block size is configured globally via `vllm_dllm_plugin.config.DRAFT_SIZE` (override with `VLLM_DLLM_DRAFT_SIZE` before importing the plugin) so scheduler/worker/remasking share one value. This completes Phase 4 runtime wiring for mock bring-up, while Phase 6 integration confidence work remains separate. `register_dllm()` continues to register the **mock** model architectures when `vllm` imports successfully.

## Docs

- [docs/DESIGN_MVP.md](docs/DESIGN_MVP.md) — MVP architecture, field mapping, diagrams (public references only).
- [docs/MOCK_STACK_MODEL.md](docs/MOCK_STACK_MODEL.md) — mock registered model ids and HF config surface (Phases 2–6).
- [docs/CONTRACTS.md](docs/CONTRACTS.md) — copy-friendly field mapping / invariants for contributors (see DESIGN_MVP section 7).
- [docs/ROADMAP.md](docs/ROADMAP.md) — phased future work.
- [docs/TOOLING.md](docs/TOOLING.md) — accurate tooling summary (pre-commit uses **`uv run`**, DCO/`sh`, run-from-root note, CI) for contributors and PR descriptions.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
