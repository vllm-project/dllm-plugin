# vllm-dllm-plugin

[![CI](https://github.com/vllm-project/dllm-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/vllm-project/dllm-plugin/actions/workflows/ci.yml)

**vllm-dllm-plugin** is a [vLLM](https://github.com/vllm-project/vllm) plugin for **block-based diffusion language models (dLLMs)**. The package provides a `vllm.general_plugins` entry point (`dllm`), Phase 1 contracts (`config`, `remasking`), a **mock registered model** for stack testing (Phases 2–6), runtime scheduler/worker adapters, strict stack validation, a **CPU CI smoke** (`EngineArgs` / `VllmConfig` / `assert_compatible_stack`) for integration drift, and a GPU-gated end-to-end mock-stack `LLM.generate` test for Phase 6 confidence. Production LLaDA2.0 model logic remains in progress (see [docs/ROADMAP.md](docs/ROADMAP.md)).

**Important:** `register_dllm()` first checks `importlib.util.find_spec("vllm")`; if `vllm` is not discoverable on `sys.path`, it returns without registering. If the spec exists but `from vllm import ModelRegistry` still fails, registration is skipped and a **DEBUG** traceback is logged. When that import succeeds, **`register_dllm()` registers two architecture names** with vLLM’s `ModelRegistry`, both targeting the **mock** in `dllm_plugin.models.mock_llada2` (not real inference—see [docs/MOCK_STACK_MODEL.md](docs/MOCK_STACK_MODEL.md)). **Scheduler/worker runtime adapters are not registered like models:** only architectures go through `ModelRegistry`; operators supply **`DllmRuntimeScheduler`** / **`DllmRuntimeWorker`** via **`--scheduler-cls`** / **`--worker-cls`** (Phase 4+ adapters already exist in-repo).

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

The optional extra pins **`vllm>=0.20.0,<0.21`** (API-churn guard, bart-plugin-style). **`uv.lock`** resolves a concrete version in that range when you sync with **`--extra vllm`** (see **Lockfile** in [CONTRIBUTING.md](CONTRIBUTING.md)). Widen the bound only after testing (e.g. optional smoke workflow + lock refresh).

**Upstream hook status (Phase 0):** dLLM draft-token hook alignment is tracked in [vllm#36155](https://github.com/vllm-project/vllm/issues/36155). Until the hook is confirmed in a pinned release for this plugin path, treat this bound as a compatibility guard and keep docs + `pyproject.toml` synchronized through issue [#2](https://github.com/vllm-project/dllm-plugin/issues/2).

See [CONTRIBUTING.md](CONTRIBUTING.md) for pre-commit, CI parity, and contribution norms.

## Using the plugin (future)

**PyPI vs Python package:** install the distribution as **`vllm-dllm-plugin`** (`pip install vllm-dllm-plugin` / `uv add vllm-dllm-plugin`). Import and CLI class paths use the **`dllm_plugin`** package name.

For MVP stack bring-up, enable the plugin by name and point vLLM at the runtime adapters. **Preferred** short flags:

```bash
export VLLM_PLUGINS=dllm
vllm serve <model> \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker
```

`Scheduler` / `Worker` are aliases on `dllm_plugin` for `DllmRuntimeScheduler` / `DllmRuntimeWorker`. vLLM resolves **dotted** qualnames (`module.sub.Class`); use `dllm_plugin.Scheduler` / `dllm_plugin.Worker` or full paths such as `dllm_plugin.runtime_scheduler.DllmRuntimeScheduler`. `DllmScheduler` and `DllmWorker` remain helper contracts used by the adapters. MVP expects `VLLM_USE_V2_MODEL_RUNNER=1`; grammar-constrained draft rewriting is intentionally rejected for dLLM block mode to avoid silent block-shape corruption. Block size is configured globally via `dllm_plugin.config.DRAFT_SIZE` (override with `VLLM_DLLM_DRAFT_SIZE` before importing the plugin) so scheduler/worker/remasking share one value. Strict stack validation (`assert_compatible_stack`) runs in runtime adapter constructors and mock-model runtime init to reject incompatible scheduler/worker/model combinations early. Runtime remask handoff consumes model score rows when available and mock fallback is restricted to the explicit mock architecture path. `register_dllm()` continues to register the **mock** model architectures when `vllm` imports successfully.

## Docs

- [docs/DESIGN_MVP.md](docs/DESIGN_MVP.md) — MVP architecture, field mapping, diagrams (public references only).
- [docs/MOCK_STACK_MODEL.md](docs/MOCK_STACK_MODEL.md) — mock registered model ids and HF config surface (Phases 2–6).
- [docs/CONTRACTS.md](docs/CONTRACTS.md) — copy-friendly field mapping / invariants for contributors (see DESIGN_MVP section 7).
- [docs/ROADMAP.md](docs/ROADMAP.md) — phased future work.
- [docs/OPERATOR_LLaDA2.md](docs/OPERATOR_LLaDA2.md) — Phase 6 operator runbook (`VLLM_PLUGINS`, CLI flags, v2 runner, integration test).
- [docs/TOOLING.md](docs/TOOLING.md) — accurate tooling summary (pre-commit uses **`uv run`**, DCO/`sh`, run-from-root note, CI) for contributors and PR descriptions.
- [docs/MILESTONE_PR_CHECKLIST.md](docs/MILESTONE_PR_CHECKLIST.md) — optional PR description checklist aligned with milestone issue [#19](https://github.com/vllm-project/dllm-plugin/issues/19).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
