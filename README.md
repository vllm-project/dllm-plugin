# vllm-dllm-plugin

[![CI](https://github.com/vllm-project/dllm-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/vllm-project/dllm-plugin/actions/workflows/ci.yml)

**vllm-dllm-plugin** is a [vLLM](https://github.com/vllm-project/vllm) plugin for **block-based diffusion language models (dLLMs)**. This repository is currently a **skeleton**: an importable package, a `vllm.general_plugins` entry point (`dllm`), tests, and public design/roadmap docs. Scheduler, worker, and LLaDA2.0 model logic will land in later milestones (see [docs/ROADMAP.md](docs/ROADMAP.md)).

**Important:** Even if you set `VLLM_PLUGINS=dllm` and a `vllm` distribution is on your `sys.path`, **`register()` does not register models, schedulers, or workers yet** — it uses `importlib.util.find_spec("vllm")` (discoverability, not a full import), may emit a **DEBUG** log, and returns. That can succeed even when `import vllm` would fail (e.g. broken native deps). Do not expect inference behavior until the MVP stack lands.

The approach follows the public RFC discussion [vllm#36155](https://github.com/vllm-project/vllm/issues/36155) and reuses spec-decode-shaped fields so batching and executor paths stay aligned.

## Install (development)

Requires Python **3.10+** and [uv](https://docs.astral.sh/uv/). CI tests **3.10** through **3.13**. A [`.python-version`](.python-version) file pins the default local interpreter for `uv`/pyenv; override as needed for the matrix.

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

`pyproject.toml` requires **`vllm>=0.14`**; **`uv.lock`** may resolve a **newer** compatible release when you use the extra (see **Lockfile** in [CONTRIBUTING.md](CONTRIBUTING.md)).

See [CONTRIBUTING.md](CONTRIBUTING.md) for pre-commit, CI parity, and contribution norms.

## Using the plugin (future)

Once the MVP stack exists, you will enable the plugin by name and point vLLM at the plugin scheduler and worker classes (FQCNs will match the implemented modules), for example:

```bash
export VLLM_PLUGINS=dllm
vllm serve <model> \
  --scheduler-cls vllm_dllm_plugin.scheduler:DllmScheduler \
  --worker-cls vllm_dllm_plugin.worker:DllmWorker
```

Until those classes exist, this is **documentation-only**; the `register()` entry point is a no-op when no `vllm` spec is found and a **silent stub** (plus a **DEBUG** log line) when `find_spec("vllm")` succeeds.

## Docs

- [docs/DESIGN_MVP.md](docs/DESIGN_MVP.md) — MVP architecture, field mapping, diagrams (public references only).
- [docs/ROADMAP.md](docs/ROADMAP.md) — phased future work.
- [docs/TOOLING.md](docs/TOOLING.md) — accurate tooling summary (pre-commit uses **`uv run`**, DCO/`sh`, run-from-root note, CI) for contributors and PR descriptions.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
