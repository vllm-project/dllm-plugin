# Contributing to vllm-dllm-plugin

Thank you for helping improve the dLLM plugin. This project follows the same spirit as **[Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/)**: clear PRs, review before merge, and **Developer Certificate of Origin (DCO)** sign-off where this repository adopts it.

## DCO and commit sign-off

Commits should include a **`Signed-off-by:`** line (per [DCO 1.1](https://developercertificate.org/)). After installing pre-commit (below), a **`prepare-commit-msg`** hook runs **`scripts/add-signoff.sh`** to append that trailer automatically using `git config user.name` / `user.email`.

- Ensure **`git config user.name`** and **`git config user.email`** are set correctly before committing (non-empty). The **`prepare-commit-msg`** hook **fails** if either is missing so you do not get a malformed **`Signed-off-by:`** trailer. Example: **`git config --global user.name 'Your Name'`** and **`git config --global user.email 'you@example.com'`**.
- For one-off commits without the hook: **`git commit -s`**.
- If you amend or squash, re-check that the final commit message still contains **`Signed-off-by:`** so DCO checks pass.

`default_install_hook_types` in `.pre-commit-config.yaml` includes **`prepare-commit-msg`** so **`pre-commit install`** registers both **`pre-commit`** and **`prepare-commit-msg`** hooks.

The **vllm-project** org may require the **DCO** GitHub check to pass on PRs. The local hook helps consistency but does **not** replace that check — ensure every commit on the PR branch includes **`Signed-off-by:`**.

## Environment

- Install **[uv](https://docs.astral.sh/uv/)** and Python **3.10+** (CI covers **3.10–3.13**). The repo includes **`.python-version`** (default **3.12**) so local `uv`/pyenv pick a consistent interpreter; change it locally if needed.
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

## Lockfile (`uv.lock`)

`uv lock` resolves **all** optional dependency groups into **`uv.lock`**, so the file is large: it pins the full transitive graph of the **`vllm`** extra (e.g. many wheels), even though **CI** only runs **`uv sync --locked --group dev`** without that extra.

That is **intentional** for reproducible **`uv sync --extra vllm`** installs. Expect large diffs when dependencies change; reviewers can skim lock updates and rely on CI + `uv lock --check` semantics via the pre-commit **`uv sync --locked`** hook.

**Alternative (not used here):** a smaller lock that omits optional extras trades away reproducibility for `uv sync --locked --extra vllm` unless maintainers adopt a separate lock or install-time resolution policy.

## Pre-commit (single source of truth for lint / format / types / lockfile)

Hooks use **`uv`** on your `PATH`:

1. **`uv sync --locked --group dev`** when `pyproject.toml` or `uv.lock` changes — fails if the lockfile is out of date. After dependency edits, run **`uv lock`** and commit **`uv.lock`**.
2. **`uv run ruff check`** and **`uv run ruff format --check`** on `vllm_dllm_plugin/` and `tests/` when those paths or **`pyproject.toml`** change (same versions as **`uv.lock`**; avoids running on doc-only commits).
3. **`uv run ty check`** when **`vllm_dllm_plugin/**/*.py`** or **`pyproject.toml`** changes.

Run **`uv sync --group dev`** once locally so **`uv run`** can resolve tools without extra network work during commits.

Optional: **`rhysd/actionlint`** on workflow YAML.

Install and run pre-commit from this repository’s root. If the plugin lives inside a parent tree (e.g. `vllm/dllm-plugin/`), **`cd` into that directory** before **`pre-commit install`** or **`uv run pre-commit run`** so hooks resolve `uv`, `pyproject.toml`, and paths correctly.

Run all hooks on every file (matches CI):

```bash
uv run pre-commit run --all-files
```

## CI parity

GitHub Actions runs **`uv sync --locked --group dev`**, then **`uv run pre-commit run --all-files`**, then **`uv run pytest`**. Do not duplicate Ruff/`ty` in the workflow; keep checks in pre-commit.

**Default CI does not install `--extra vllm`**, so tests that need an importable **`vllm`** package are skipped there. That avoids CUDA/wheel pain on every PR. For an optional integration check, maintainers can run the **Optional vLLM smoke** workflow ( **`workflow_dispatch`** in `.github/workflows/optional-vllm-smoke.yml` ), which syncs with **`--extra vllm`** and runs pytest.

**Optional vLLM smoke** is **best-effort** on **`ubuntu-latest`**: **`uv sync --extra vllm`** may fail if PyPI has no suitable wheel or pins break on that image (see header comments in the workflow YAML). Dispatch it after lock/extra changes and note success or failure for the team until behavior is stable.

## Shell for pre-commit

The **DCO** hook runs **`sh scripts/add-signoff.sh`** (POSIX **`sh`**). Use Linux, macOS, **Git Bash**, or **WSL** so **`sh`** is available. Ruff, `ty`, and **`uv sync`** use plain **`entry`** commands (no bash).

## PR descriptions

Keep GitHub PR bodies aligned with the branch — see **[docs/TOOLING.md](docs/TOOLING.md)** for an accurate summary (e.g. pre-commit uses **`uv run`**, not **`uvx`**, for Ruff and `ty`).

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
