# Tooling summary (keep docs and PR descriptions accurate)

Use this checklist when opening or updating pull requests so the **GitHub PR description** matches the branch (e.g. avoid stale mentions of **`uvx`** for Ruff/`ty`).

| Topic | Current behavior |
|-------|------------------|
| **Lint / format / types in pre-commit** | **`uv run ruff`** and **`uv run ty`** from the project environment — versions come from **`uv.lock`** (same as CI after `uv sync --group dev`). Ruff enables **E, F, I, UP, W, B, G, SIM, ISC** (closer to vLLM hygiene than a minimal skeleton set; not identical to core). Hooks are **scoped by `files:`** (Python under **`vllm_dllm_plugin/`** / **`tests/`** and **`pyproject.toml`**) so doc-only commits skip Ruff/`ty`; **`pre-commit run --all-files`** still runs them when matching files exist. |
| **Type checking** | This repo uses **`ty` only**. **vLLM core** is **mypy-centric** in its own tree; do not assume core publishes a canonical **`ty`** config to copy. Revisit **`mypy`** (or dual checks) if maintainers want stricter parity when the plugin grows vLLM-facing typed surface. |
| **Lockfile check** | **`uv sync --locked --group dev`** when `pyproject.toml` / `uv.lock` change. **`uv.lock` stays tracked** and includes resolution for the optional **`vllm`** extra (large file by design — reproducible **`uv sync --extra vllm`**). |
| **Version** | **`setuptools-scm`** from **git tags**; see **Versioning** in **[CONTRIBUTING.md](CONTRIBUTING.md)**. |
| **Commit messages** | Prefer clear **subject lines** (**what** / **why**). For **substantively AI-assisted** commits, a **short factual note** in the body is allowed; see **AI-assisted contributions** in **[CONTRIBUTING.md](CONTRIBUTING.md)**. |
| **Shell for hooks** | Ruff, `ty`, and **`uv sync`** use single-command **`entry`** lines. The DCO hook runs **`sh scripts/add-signoff.sh`** — need **`sh`** (Linux, macOS, Git Bash, WSL). |
| **Where to run pre-commit** | From this repo root; if embedded under another tree, **`cd`** into the plugin directory first. |
| **DCO sign-off** | Local **`prepare-commit-msg`** hook appends **`Signed-off-by:`**; the **vllm-project** org may still enforce the **DCO** status check on GitHub — both matter. |
| **CI default** | **`uv sync --locked --group dev`** only (no **`--extra vllm`**). Optional **`workflow_dispatch`** job can install **`vllm`** for manual runs (may fail on stock runners depending on wheels; see workflow comments). |
| **Optional vLLM bound** | `pyproject.toml` is the canonical source for the optional `vllm` bound. Keep docs consistent with that file; keep upstream hook tracking context in README and plugin issue [#2](https://github.com/vllm-project/dllm-plugin/issues/2), with upstream reference [vllm#36155](https://github.com/vllm-project/vllm/issues/36155). |
