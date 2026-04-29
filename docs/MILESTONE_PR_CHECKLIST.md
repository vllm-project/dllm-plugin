# Milestone PR checklist (issue #19 alignment)

Use this when opening or updating a PR that tracks **[issue #19](https://github.com/vllm-project/dllm-plugin/issues/19)** (LLaDA2.0 milestone orchestration). Copy sections into the GitHub PR description as needed; nothing here is enforced by CI.

## Labels

- **Phase:** e.g. `Phase 5–6` — match the orchestration issue phase table.

## Dependencies

| Kind | Examples |
|------|----------|
| **HARD** | Must merge / land before this PR is correct (e.g. upstream vLLM hook release tracked in issue #2). |
| **SOFT** | Parallel work or documentation-only dependencies that do not block merge but should be tracked. |

## Structured checklist (example)

- [ ] Maps to issue #19 phase exit criteria (cite subsection).
- [ ] Tests / docs updated per phase gate (unit, operator doc, integration path).
- [ ] If GPU-only evidence: link job status + persisted logs (see `docs/OPERATOR_LLaDA2.md`).
- [ ] Closes / updates linked GitHub issues with accurate scope (avoid claiming CI exercises GPU paths when only CPU smoke runs on default runners).

For mock-stack vs Phase 7 real weights, keep closure language precise so **Phase 6 “integration confidence”** is not overstated relative to what CI actually runs.
