# Tooling summary (keep docs and PR descriptions accurate)

Use this checklist when opening or updating pull requests so the **GitHub PR description** matches the branch (e.g. avoid stale mentions of **`uvx`** for Ruff/`ty`).

| Topic | Current behavior |
|-------|------------------|
| **Lint / format / types in pre-commit** | **`uv run ruff`** and **`uv run ty`** from the project environment — versions come from **`uv.lock`** (same as CI after `uv sync --group dev`). |
| **Lockfile check** | **`uv sync --locked --group dev`** when `pyproject.toml` / `uv.lock` change. |
| **Shell for hooks** | Hook **`entry`** values use **`bash -c '…'`** (and **`sh`** for `scripts/add-signoff.sh`). **Bash** must be available (Linux, macOS, Git Bash on Windows; native Windows cmd without Bash is unsupported for these hooks). |
| **DCO sign-off** | Local **`prepare-commit-msg`** hook appends **`Signed-off-by:`**; the **vllm-project** org may still enforce the **DCO** status check on GitHub — both matter. |
| **CI default** | **`uv sync --locked --group dev`** only (no **`--extra vllm`**). Optional **`workflow_dispatch`** job can install **`vllm`** for manual runs. |
