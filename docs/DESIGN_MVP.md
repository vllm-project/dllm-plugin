# dLLM plugin — MVP design

This document describes the **MVP architecture** for [`vllm-project/dllm-plugin`](https://github.com/vllm-project/dllm-plugin). It aligns with the public design discussion in [vllm#36155](https://github.com/vllm-project/vllm/issues/36155) (spec-decode path reuse, minimal core change).

**Audience:** implementers and reviewers of the plugin and the minimal vLLM core hook.

---

## 1. MVP goals

| Goal | Notes |
|------|--------|
| **One diffusion step = one worker schedule = one model forward** | Same abstraction as in that discussion; continuous batching stays aligned across requests. |
| **Block size `DRAFT_SIZE`** | Fixed per model (e.g. 32 for LLaDA2.0); one input block in, variable **Committed** (0..DRAFT_SIZE) + fixed **next-step input block** out. |
| **Reuse spec-decode fields** | No new core tensor types; overload meaning when plugin scheduler + worker are active. |
| **Custom scheduler + worker + registered model** | Loaded via `--scheduler-cls` / `--worker-cls` and `vllm.general_plugins` model registration. |
| **Commit-0** | Plugin scheduler rolls back `num_computed_tokens` when no tokens are committed in a step. |
| **Composable remasking (MVP scope)** | Pluggable **remasking policy** interface after forward (threshold / top-k style); LLaDA2.0 can ship with one default implementation. |
| **First architecture** | LLaDA2.0 inference path end-to-end. |
| **Validation** | Fail fast if a dLLM model is used without the plugin scheduler/worker (or wrong classes). |

**Out of MVP** (see [ROADMAP.md](ROADMAP.md)): grammar/structured outputs beyond “do not break AR grammar on next block”, bespoke CUDA attention kernels where **virtual non-causal chunks** on existing FlashAttention paths are insufficient ([§9](#9-attention-and-execution-mvp)), prefix caching under semi-causal masks, extra architectures, draft streaming UX, and advanced grammar integrations.

---

## 2. Design principles

1. **Thin core, fat plugin** — vLLM change is only the draft-token hook guard; dLLM semantics live in the plugin.
2. **Strict stack** — Model + scheduler + worker are **one supported configuration**; no mixing with default scheduler/worker for dLLM models.
3. **Spec-decode-shaped I/O** — Scheduler and worker agree on overloaded fields so existing batching and executor paths stay exercised.
4. **Remasking behind an interface** — Model forward produces logits/hidden state; **RemaskingPolicy** (or equivalent) updates draft state and decides commit candidates.

---

## 3. Suggested package layout (MVP)

```text
dllm_plugin/
  __init__.py              # register_dllm() entry for vllm.general_plugins
  config.py                # DRAFT_SIZE, model id constants, feature flags
  validation.py            # assert_compatible_stack(vllm_config)
  scheduler.py             # DllmScheduler (v1 scheduler interface)
  worker.py                # DllmWorker (WorkerBase subclass)
  remasking/
    __init__.py
    base.py                # RemaskingPolicy protocol / ABC
    llada2_default.py       # MVP default for LLaDA2.0
  models/
    __init__.py
    mock_llada2.py         # stack-test stub (Phases 2–6); not production inference
    llada2.py              # real vLLM model module (HF mapping) — Phase 7 / issue #12
```

**Implemented defaults:** `DRAFT_SIZE` (32 for LLaDA2.0 MVP), model identifier
constants, and feature flags live in `dllm_plugin.config` with docstrings as
the implementer-facing source of truth (milestone issue #3).

Naming is illustrative; the PyPI distribution is **`vllm-dllm-plugin`**.

---

## 4. vLLM core vs plugin boundary

```mermaid
flowchart TB
  subgraph core [vLLM core]
    Engine[EngineCore]
    SchedIface[Scheduler interface]
    Exec[ModelExecutor]
    Hook[post_step draft hook]
  end
  subgraph plugin [dllm-plugin]
    DllmSched[DllmScheduler]
    DllmWork[DllmWorker]
    DllmModel[LLaDA2 model]
    Remask[RemaskingPolicy]
  end
  Engine --> SchedIface
  SchedIface --> DllmSched
  Exec --> DllmWork
  DllmWork --> DllmModel
  DllmModel --> Remask
  Hook -->|"take_draft_token_ids update_draft_token_ids"| DllmSched
  DllmWork -->|"ModelRunnerOutput"| Engine
```

**Core dependency:** After the upstream hook lands in vLLM, `Hook` runs whenever a model step executed and draft IDs exist—not only when `speculative_config` is set. Until then, document a **minimum vLLM version or git SHA** once integration tests pin it (the canonical optional-extra bound lives in `pyproject.toml`; unlike bart-style plugins that often require vLLM at install time, this repo keeps vLLM optional for contributor ergonomics). The exact release containing the hook is tracked via [vllm#36155](https://github.com/vllm-project/vllm/issues/36155), with human-readable tracking context maintained in README and plugin issue [#2](https://github.com/vllm-project/dllm-plugin/issues/2).

---

## 5. Registration and runtime

```mermaid
flowchart LR
  subgraph registration [Discovery]
    EP[vllm.general_plugins]
    MR[ModelRegistry]
    EP --> RegFn[register_dllm]
    RegFn --> MR
  end
  subgraph runtime [Runtime stack]
    DllmSched[DllmScheduler]
    DllmWork[DllmWorker]
    Runner[Model runner GPU path]
    DllmSched -->|"SchedulerOutput"| DllmWork
    DllmWork --> Runner
    Runner -->|"logits + hidden"| Remask[RemaskingPolicy]
    Remask -->|"committed_ids next_block"| DllmWork
  end
  Val[validation.py] --> DllmWork
  Val --> DllmSched
```

- **Registration** mirrors [bart-plugin](https://github.com/vllm-project/bart-plugin): one entry point that registers architecture names → qualified model class strings.
- **Runtime** uses the same split of responsibilities: scheduler owns request state for `spec_token_ids`; worker maps `scheduled_spec_decode_tokens` to the forward and fills `sampled_token_ids` + draft return path.

---

## 6. One decode step (sequence)

```mermaid
sequenceDiagram
  participant Engine
  participant DllmSched as DllmScheduler
  participant DllmWork as DllmWorker
  participant Model as LLaDA2Model
  participant Remask as RemaskingPolicy

  DllmSched->>DllmSched: schedule read spec_token_ids
  DllmSched->>DllmSched: set scheduled_spec_decode_tokens num_scheduled_tokens equals DRAFT_SIZE
  Engine->>DllmWork: SchedulerOutput
  DllmWork->>DllmWork: build batch from input block
  DllmWork->>Model: forward one block plus KV context
  Model->>Remask: logits or per-position scores
  Remask->>DllmWork: committed_token_ids zero_to_DRAFT_SIZE
  Remask->>DllmWork: next_step_input_block length DRAFT_SIZE
  DllmWork->>DllmWork: set sampled_token_ids to Committed
  DllmWork->>DllmWork: prepare DraftTokenIds for next block
  DllmWork->>Engine: ModelRunnerOutput
  Engine->>DllmSched: update_from_output commit minus zero rollback
  Engine->>Engine: post_step take_draft_token_ids update_draft_token_ids
  DllmSched->>DllmSched: spec_token_ids equals next block
```

**Commit-0:** In `update_from_output`, if `sampled_token_ids` is empty for a request, the scheduler rolls back `num_computed_tokens` by the number of tokens scheduled that step (typically `DRAFT_SIZE` in this MVP design).

---

## 7. Field mapping (MVP contract)

| vLLM field / API | Role when plugin stack is active |
|------------------|----------------------------------|
| `Request.spec_token_ids` | **Next-step input block** (length `DRAFT_SIZE`) for the upcoming schedule. |
| `SchedulerOutput.scheduled_spec_decode_tokens` | **Input block** (length `DRAFT_SIZE`) for this step’s forward. |
| `SchedulerOutput.num_scheduled_tokens` (per request) | Set to `DRAFT_SIZE` for decode steps using the block path. |
| `ModelRunnerOutput.sampled_token_ids` | **Committed** token IDs only, length 0..`DRAFT_SIZE` (may be empty). |
| Worker `take_draft_token_ids()` | Returns **next-step input block** packaged as `DraftTokenIds` for engine → scheduler. |
| Scheduler `update_draft_token_ids` / `update_draft_token_ids_in_output` | Store next block into `spec_token_ids`; **must not** apply AR draft grammar to dLLM blocks (override for structured output / async). |

Mutually exclusive with true speculative decoding on the same requests: operators must not enable spec-decode + dLLM plugin stack together for the same run mode.

**Contributor copy:** The ASCII summary `docs/CONTRACTS.md` tracks this section (and related timing in section 6). Update both places together when field names or semantics change so they do not drift.

**Upstream drift:** vLLM API identifiers above are accurate for the revision range implied by `pyproject.toml` (optional `vllm` extra). They are not continuously validated against vLLM `main`; when the pin moves, reconcile this table and `docs/CONTRACTS.md` (plugin issue #2 tracks minimum-version / hook context).

---

## 8. Remasking composability (MVP)

```mermaid
flowchart TB
  Forward[Model forward one block]
  State[Draft state per position]
  Forward --> State
  State --> Policy[RemaskingPolicy.apply]
  Policy --> Committed[Committed subset]
  Policy --> NextInput[Next input block MASK plus decoded]
  Committed --> OutSched[sampled_token_ids]
  NextInput --> OutDraft[DraftTokenIds]
```

**MVP contract (conceptual):**

- **Input:** Current input block, logits (or equivalent), optional request config (e.g. threshold).
- **Output:** `committed_token_ids: list[int]` (0..N), `next_input_block: list[int]` (length `DRAFT_SIZE`), and internal mask/draft state for logging.

**Shape checks:** `RemaskStepResult` (see `dllm_plugin.remasking`) does not validate lengths at construction. After `RemaskingPolicy.apply`, the worker or policy boundary should run `validate_remask_step_result()` (same package) or the concrete policy should raise `ValueError` for invalid shapes, consistent with the protocol docstring on `apply`.

**Lists vs tuples:** Conceptual output above uses `list[int]`; the implemented `RemaskStepResult` uses immutable `tuple[int, ...]`. Worker code should convert where vLLM/engine APIs require lists.

**Protocol runtime checks:** `RemaskingPolicy` is `@runtime_checkable`; `isinstance(obj, RemaskingPolicy)` only checks for a callable `apply`, not full signature compliance or return types. Use tests and static typing for the real contract.

**LLaDA2.0 default** implements one concrete policy (e.g. confidence-based commit + remask rest); additional policies can plug in as new `RemaskingPolicy` implementations without changing the worker’s engine contract.

### 8.1 Phase 3–4 bridge (#13 / #10)

Phase 6 documentation **does not** replace Phase 3–4 obligations; it layers validation and runtime adapters on top:

- **Issue #13 (`remasking/handoff.py`):** After the last pipeline-parallel forward, `remask_after_block_forward` consumes **2-D** block logits `(DRAFT_SIZE, vocab_size)` and a concrete `RemaskingPolicy`. Non-last PP ranks do not run remasking (`logits is None`).
- **Issue #10 (`worker.py`):** `DllmWorker.run_one_block` wires forward outputs through that helper and maps `RemaskStepResult` into scheduler-visible fields (`DllmWorkerStep` / `take_draft_token_ids`).
- **Commit-0 vs policy stepping:** Scheduler **commit-0** rollback when `sampled_token_ids` is empty is independent of **`Llada2DefaultRemaskingPolicy`** inner transfer scheduling (`denoise_steps`, `denoise_step_index`, …). Copy-friendly tables live in `docs/CONTRACTS.md` (forward → remasking + `remasking_config` keys).

---

## 9. Attention and execution (MVP)

### 9.1 Virtual sub-requests (reference pattern in vLLM)

vLLM’s **chunked local attention** implements a non-standard attention layout by decomposing a logical sequence into several **virtual requests**. Each virtual request runs **ordinary causal** attention on its own contiguous key range, so existing causal kernels apply. Implementation entry points:

- [`vllm/model_executor/layers/attention/chunked_local_attention.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/chunked_local_attention.py)
- Backend wiring (commit-pinned line range): [`vllm/v1/attention/backends/utils.py` @ `4ed51308`](https://github.com/vllm-project/vllm/blob/4ed51308c8826619459be858a6dc4333206f41c1/vllm/v1/attention/backends/utils.py#L167-L359)

The dLLM plugin can mirror that **decomposition idea** with a different per-chunk mask: see below.

### 9.2 Mask shapes (schematic, model-dependent)

Exact geometry is architecture-specific; the following ASCII sketches contrast the **chunked-local** staggered window with a **dense block-style** mask common among dLLMs (both shown for six token positions; `1` = allowed attention).

**Chunked local (staggered local windows):**

```text
k_toks >   0 1 2 3 4 5
q_toks v  _____________
       0 | 1
       1 | 1 1
       2 |     1
       3 |     1 1
       4 |         1
       5 |         1 1
```

**Many dLLMs (prefix / block visibility grows by step):**

```text
k_toks >   0 1 2 3 4 5
q_toks v  _____________
       0 | 1 1
       1 | 1 1
       2 | 1 1 1 1
       3 | 1 1 1 1
       4 | 1 1 1 1 1 1
       5 | 1 1 1 1 1 1
```

### 9.3 Decomposition: causal chunks vs non-causal chunks

**Chunked local** effectively splits into virtual requests that each look like a tiny **causal** problem, e.g. keys `{0,1}` for queries `{0,1}`, then keys `{2,3}` for `{2,3}`, then `{4,5}` for `{4,5}`—each sub-matrix is lower-triangular.

**Block-style dLLM masks** can be split analogously into virtual requests where each sub-problem is **fully connected among its allowed (q, k) pairs**—i.e. **standard non-causal** attention on that key/query subset—for example:

```text
virtual req 0 (q,k over {0,1}):   virtual req 1 (over {2,3}):   virtual req 2 (over {4,5}):
       0 | 1 1                         2 | 1 1 1 1                     4 | 1 1 1 1 1 1
       1 | 1 1                         3 | 1 1 1 1                     5 | 1 1 1 1 1 1
```

**FlashAttention** is used with **`is_causal=False`** on these chunks; that path is a normal non-causal workload and is **not** inherently less optimized than other non-causal attention (per upstream attention/maintainer discussion). A **blocked** or arbitrary sparse mask can therefore often be served by **composition of virtual non-causal (and, where needed, causal) chunks** plus FlexAttention or explicit mask metadata—**before** investing in bespoke CUDA.

### 9.4 MVP baseline

- Prefer **FlexAttention**, **FlashAttention with non-causal virtual chunks**, and/or **custom masks** consistent with the public design thread [#36155](https://github.com/vllm-project/vllm/issues/36155).
- **Worker responsibility:** Keep **`num_spec_tokens` / draft buffers** consistent with what `take_draft_token_ids` expects.

---

## 10. Operator configuration (illustrative)

```bash
export VLLM_PLUGINS=dllm
vllm serve <model> \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker
```

Current MVP runtime adapters are `DllmRuntimeScheduler` / `DllmRuntimeWorker`, exposed for CLI as **`dllm_plugin.Scheduler`** / **`dllm_plugin.Worker`** or full dotted paths under `dllm_plugin.runtime_*` (vLLM resolves dotted qualnames). Helper classes (`dllm_plugin.scheduler:DllmScheduler`, `dllm_plugin.worker:DllmWorker`) remain the contract core used by adapters. Before the first decode schedule, `request.spec_token_ids` must hold the first input block (`DRAFT_SIZE` tokens); the plugin scheduler initializes it (prompt suffix + mask padding per this MVP design). Strict stack validation (`dllm_plugin.validation.assert_compatible_stack(...)`) runs in runtime adapter constructors and at mock-model runtime initialization, so dLLM architecture + incompatible scheduler/worker fails fast.

Phase 6 integration confidence includes a concrete runtime integration test (`tests/test_vllm_mock_integration.py`) that instantiates vLLM runtime objects with the plugin adapters and executes one mock-stack generation step (GPU-gated).

---

## 11. Risks (MVP)

| Risk | Mitigation |
|------|------------|
| Custom scheduler API not stable | Pin max tested vLLM version; integration tests in CI. |
| Draft hook not in release | Document minimum vLLM from SHA or nightly until released. |
| Structured output + async queue | Implement scheduler overrides early; defer full PDA post-MVP where possible. |
| Wrong worker/scheduler pairing | Implemented strict stack check via `validation.py` in runtime adapter init paths. |
