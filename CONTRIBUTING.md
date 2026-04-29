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

- Install **[uv](https://docs.astral.sh/uv/)** and Python **3.10–3.13** (`requires-python` is **`>=3.10,<3.14`**, consistent with current vLLM). CI covers **3.10–3.13**. The repo includes **`.python-version`** (default **3.12**) so local `uv`/pyenv pick a consistent interpreter; change it locally if needed.
- Create a venv and install dev dependencies:

  ```bash
  uv sync --group dev
  ```

- **Runtime vLLM** is an **optional extra** with a **bounded** spec (`vllm>=0.14,<0.15`) so installs stay on tested API lines; widen only with lock refresh + optional smoke. macOS and other environments without CUDA wheels can still develop and run tests that skip vLLM-specific cases. On Linux (e.g. CI or GPU dev boxes), install with:

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
2. **`uv run ruff check`** and **`uv run ruff format --check`** on `dllm_plugin/` and `tests/` when those paths or **`pyproject.toml`** change (same versions as **`uv.lock`**; avoids running on doc-only commits).
3. **`uv run ty check`** when **`dllm_plugin/**/*.py`** or **`pyproject.toml`** changes.

Run **`uv sync --group dev`** once locally so **`uv run`** can resolve tools without extra network work during commits.

Optional: **`rhysd/actionlint`** on workflow YAML.

Install and run pre-commit from this repository’s root. If the plugin lives inside a parent tree (e.g. `vllm/dllm-plugin/`), **`cd` into that directory** before **`pre-commit install`** or **`uv run pre-commit run`** so hooks resolve `uv`, `pyproject.toml`, and paths correctly.

Run all hooks on every file (matches CI):

```bash
uv run pre-commit run --all-files
```

## Milestone PR descriptions

Orchestration issue **[#19](https://github.com/vllm-project/dllm-plugin/issues/19)** suggests structured PR descriptions (phase label, HARD/SOFT dependencies, checklist). Optional copy-paste starter: **`docs/MILESTONE_PR_CHECKLIST.md`**.

## CI parity

GitHub Actions runs **`uv sync --locked --group dev`**, then **`uv run pre-commit run --all-files`**, then **`uv run pytest`**. Do not duplicate Ruff/`ty` in the workflow; keep checks in pre-commit.

**Default CI does not install `--extra vllm`**, so tests that need an importable **`vllm`** package are skipped there. That avoids CUDA/wheel pain on every PR. For an optional integration check, maintainers can run the **Optional vLLM smoke** workflow ( **`workflow_dispatch`** in `.github/workflows/optional-vllm-smoke.yml` ), which syncs with **`--extra vllm`** and runs pytest.

**Optional vLLM smoke** is **best-effort** on **`ubuntu-latest`**: **`uv sync --extra vllm`** may fail if PyPI has no suitable wheel or pins break on that image (see header comments in the workflow YAML). Dispatch it after lock/extra changes and note success or failure for the team until behavior is stable.

**Before the workflow exists on `main`**, **`workflow_dispatch`** cannot be triggered (GitHub has no registered workflow yet). To get an **`ubuntu-latest`** run from a PR, maintainers may **temporarily** add to **`optional-vllm-smoke.yml`**:

```yaml
pull_request:
  branches: [main, master]
  paths:
    - ".github/workflows/optional-vllm-smoke.yml"
```

so only edits to that workflow file start the job (not every doc commit). Push, copy the **Actions** run URL/conclusion into the PR description, then **remove** the `pull_request` block and leave **`workflow_dispatch`** only. The run may stay **queued** until org runners pick it up—check the **Actions** tab if it does not start.

## Shell for pre-commit

The **DCO** hook runs **`sh scripts/add-signoff.sh`** (POSIX **`sh`**). Use Linux, macOS, **Git Bash**, or **WSL** so **`sh`** is available. Ruff, `ty`, and **`uv sync`** use plain **`entry`** commands (no bash).

## PR descriptions

Keep GitHub PR bodies aligned with the branch — see **[docs/TOOLING.md](docs/TOOLING.md)** for an accurate summary (e.g. pre-commit uses **`uv run`**, not **`uvx`**, for Ruff and `ty`). Do **not** paste paths to **local-only** review or planning files; link **public** issues, RFCs, or this repo’s **`docs/`** instead.

For milestone work, use the repository PR template at **`.github/PULL_REQUEST_TEMPLATE.md`** and fill the **Issue + Phase**, **HARD/SOFT dependencies**, **scope**, **validation evidence**, and **docs impact** sections.

## Manual checks (optional)

From the synced environment (lockfile-pinned tool versions):

```bash
uv run ruff check dllm_plugin tests
uv run ruff format --check dllm_plugin tests
uv run ty check
uv run pytest
```

Use `# ty: ignore` sparingly; prefer fixing types or narrowing stubs.

## Upstream alignment

- **[Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/)** — DCO / sign-off, PR title prefixes (`[Doc]`, `[CI/Build]`, `[Misc]`, etc.), review expectations.
- **[Plugin system](https://docs.vllm.ai/en/latest/design/plugin_system.html)** — how plugins are discovered and loaded.
- **Examples:** [bart-plugin](https://github.com/vllm-project/bart-plugin) (model-oriented plugin), [vllm-metal](https://github.com/vllm-project/vllm-metal) (worker/platform-oriented reference).

## AI-assisted contributions

There is no separate upstream “AI policy” file in the vLLM tree. If you use AI tools for a **substantive** change (material help with design, code, tests, or docs), **disclose that clearly** in at least one appropriate place:

- **Pull request:** Prefer a short **PR description** section (what you used, for which parts, and how you validated the result). This is the default place for narrative disclosure.
- **Commits:** When a **single commit** is substantially AI-assisted, you **may** add a brief factual note in the commit body (e.g. tool + scope). Keep the **subject line** human-focused (**what** / **why**); avoid empty marketing or unrelated trailers.
- **Issues / discussions:** When filing bugs, questions, or design threads that rely on AI-generated analysis, note that so reviewers can calibrate.

You remain responsible for correctness, security, and licensing. If this repo uses **Signed-off-by** (DCO), your sign-off must still reflect that you are submitting the work you understand you have the right to submit.

**Optional:** maintainers may squash or rewrite history for bisect-friendly narrative; that is **not** required solely to remove AI disclosure.

## Versioning (setuptools-scm)

The distribution version is **derived from git** via **[setuptools-scm](https://github.com/pypa/setuptools_scm)** (no static `version` field in `pyproject.toml`).

- **Releases:** tag with **`vMAJOR.MINOR.PATCH`** (e.g. **`v0.1.0`**). Builds and **`importlib.metadata.version("vllm-dllm-plugin")`** use that tag.
- **Between tags:** installs report **PEP 440** dev/local versions (e.g. **`0.2.0.dev3+gabcdef`**) from commit distance and hash — expected for development.
- **CI / clones:** shallow checkouts can break SCM version resolution; this repo’s **CI** fetches **full git history** for the workflow so **`setuptools-scm`** can run.
- Until the **first** release tag exists, versions are **development-shaped**; that is normal for a pre-1.0 skeleton.

## SPDX headers (Python)

New **`.py`** files under **`dllm_plugin/`** and **`tests/`** should start with the same SPDX lines as vLLM core:

```text
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

## Maintainer alignment notes

- **`ty` without `mypy`:** intentional for this small package; see **[docs/TOOLING.md](docs/TOOLING.md)**.
- **Tracked `uv.lock`:** includes optional **`vllm`** resolution on purpose (large lockfile tradeoff documented under **Lockfile** above).

## CODEOWNERS (org routing)

If **`.github/CODEOWNERS`** is not present yet, add one once **vllm-project** provides a **GitHub team** or **maintainer list** for this repository (see [GitHub CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)). Until then, use **PR reviewers** explicitly.

## RFC / design context

- Discussion: [vllm#36155](https://github.com/vllm-project/vllm/issues/36155)
- Spec (public): [`rfc-dllm-plugin-standalone-v2.md`](https://github.com/vllm-project/vllm/blob/main/specs/002-dllm-plugin/rfc-dllm-plugin-standalone-v2.md)
- In-repo MVP design: [docs/DESIGN_MVP.md](docs/DESIGN_MVP.md)
