# Contributing to vllm-dllm-plugin

Thank you for helping improve the dLLM plugin. This project follows the same spirit as **[Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/)**: clear PRs, review before merge, and **Developer Certificate of Origin (DCO)** sign-off where this repository adopts it.

## Environment

- Install **[uv](https://docs.astral.sh/uv/)** and Python **3.10+** (CI covers **3.10–3.13**).
- Create a venv and install dev dependencies:

  ```bash
  uv sync --group dev
  ```

- **Runtime vLLM** is an **optional extra** (`vllm>=0.14`) so macOS and other environments without CUDA wheels can still develop and run tests that skip vLLM-specific cases. On Linux (e.g. CI or GPU dev boxes), install with:

  ```bash
  uv sync --group dev --extra vllm
  ```

- Install commit hooks:

  ```bash
  uv run pre-commit install
  ```

## Pre-commit (single source of truth for lint / format / types / lockfile)

Hooks use **`uv`** on your `PATH`:

1. **`uv sync --locked --group dev`** when `pyproject.toml` or `uv.lock` changes — fails if the lockfile is out of date. After dependency edits, run **`uv lock`** and commit **`uv.lock`**.
2. **`uvx ruff check`** and **`uvx ruff format --check`** on `vllm_dllm_plugin/` and `tests/`.
3. **`uvx ty check`** for the project (`ty check --project dllm-plugin` when this repo lives under a parent git tree as `dllm-plugin/`).

Optional: **`rhysd/actionlint`** on workflow YAML.

If you clone **only** this repository, hooks run from the repo root. If you work inside a **parent** repository (e.g. the main `vllm` tree with `dllm-plugin/` as a subdirectory), the same `.pre-commit-config.yaml` detects that layout and only touches the plugin paths.

Run all hooks on every file (matches CI):

```bash
uv run pre-commit run --all-files
```

## CI parity

GitHub Actions runs **`uv sync --locked --group dev`**, then **`uv run pre-commit run --all-files`**, then **`uv run pytest`**. Do not duplicate Ruff/`ty` in the workflow; keep checks in pre-commit.

## Manual checks (optional)

From the synced environment (lockfile-pinned tool versions):

```bash
uv run ruff check vllm_dllm_plugin tests
uv run ruff format --check vllm_dllm_plugin tests
uv run ty check
uv run pytest
```

Use `# ty: ignore` sparingly; prefer fixing types or narrowing stubs.

## Upstream alignment

- **[Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/)** — DCO / sign-off, PR title prefixes (`[Doc]`, `[CI/Build]`, `[Misc]`, etc.), review expectations.
- **[Plugin system](https://docs.vllm.ai/en/latest/design/plugin_system.html)** — how plugins are discovered and loaded.
- **Examples:** [bart-plugin](https://github.com/vllm-project/bart-plugin) (model-oriented plugin), [vllm-metal](https://github.com/vllm-project/vllm-metal) (worker/platform-oriented reference).

## AI-assisted contributions

There is no separate upstream “AI policy” file in the vLLM tree. If you use AI tools for a **substantive** change, **say so in the PR description**. You remain responsible for correctness, security, and licensing. If this repo uses **Signed-off-by** (DCO), your sign-off must still reflect that you are submitting the work you understand you have the right to submit.

## RFC / design context

- Discussion: [vllm#36155](https://github.com/vllm-project/vllm/issues/36155)
- Spec (public): [`rfc-dllm-plugin-standalone-v2.md`](https://github.com/vllm-project/vllm/blob/main/specs/002-dllm-plugin/rfc-dllm-plugin-standalone-v2.md)
- In-repo MVP design: [docs/DESIGN_MVP.md](docs/DESIGN_MVP.md)

<!-- Assisted by: Cursor AI -->
