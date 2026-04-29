# MVP field mapping (contributor reference)

Copy-friendly summary of **`docs/DESIGN_MVP.md` section 7** (field mapping) and
related invariants. Keep this file ASCII-only for terminals and PR review.

**Keeping docs aligned:** Edits to the field-mapping table or invariants in
`DESIGN_MVP.md` section 7 (and closely related sections 6 and 8) should be
mirrored here, and substantive changes here should be reflected back in
`DESIGN_MVP.md`, so contributor copy and the canonical design doc do not drift
silently.

**Upstream vLLM identifiers:** The table below uses vLLM type and member names as
they appear for the **bounded** optional `vllm` dependency in `pyproject.toml`.
Those names can change in newer vLLM releases. When raising or widening that
pin, re-check upstream APIs and update this file and `DESIGN_MVP.md` together
(see plugin issue #2 for hook / minimum-version tracking).

## Spec-decode-shaped fields (plugin stack active)

| vLLM field / API | Role when dLLM plugin scheduler + worker are active |
|------------------|-----------------------------------------------------|
| `Request.spec_token_ids` | **Next-step input block** for the upcoming schedule. Length **`DRAFT_SIZE`** (see `dllm_plugin.config.DRAFT_SIZE`). |
| `SchedulerOutput.scheduled_spec_decode_tokens` | **Input block** for **this** step's forward. Length **`DRAFT_SIZE`**. |
| `SchedulerOutput.num_scheduled_tokens` (per request) | Set to **`DRAFT_SIZE`** for decode steps on the block path. |
| `ModelRunnerOutput.sampled_token_ids` | **Committed** token IDs only; length **0..`DRAFT_SIZE`** (may be empty). |
| Worker `take_draft_token_ids()` | Returns the **next-step input block** as `DraftTokenIds` for engine -> scheduler. |
| Scheduler `update_draft_token_ids` / `update_draft_token_ids_in_output` | Store the next block into `spec_token_ids`. **Must not** apply AR draft grammar to dLLM blocks (scheduler overrides for structured output / async). |

**Runtime wiring note:** CLI overrides use runtime adapters (preferred:
`dllm_plugin.Scheduler` and `dllm_plugin.Worker`) or full dotted paths such as
`dllm_plugin.runtime_scheduler.DllmRuntimeScheduler` and
`dllm_plugin.runtime_worker.DllmRuntimeWorker`. The helper classes in
`dllm_plugin.scheduler` / `dllm_plugin.worker` are contract-core
implementations used internally by those adapters. Runtime adapter constructors
call `dllm_plugin.validation.assert_compatible_stack(...)` and fail fast
for incompatible scheduler/worker/model combinations unless validation is turned
off via `strict=False` or `VLLM_DLLM_STRICT_STACK_VALIDATION=0` (see
`docs/OPERATOR_LLaDA2.md`).

## Commit-0 rollback

If no tokens are committed in a step, the plugin scheduler rolls back
`num_computed_tokens` by the number of tokens scheduled that step (typically
**`DRAFT_SIZE`**). See `DESIGN_MVP.md` section 6 (sequence diagram) and section 1
(Commit-0 goal).

## Mutual exclusion

True speculative decoding on the same requests must **not** be combined with the
dLLM plugin stack (same run mode). See `DESIGN_MVP.md` section 7 (last paragraph).
In the Phase 4 scheduler implementation, grammar-constrained draft rewriting is
treated as an explicit error for dLLM block mode (fail fast, no silent rewrite).

## Forward → remasking handoff (issues #13 / #10)

Normative bridge between **one block forward** and **policy output**, preserved from Phase 3–4 milestones:

| Obligation | Detail |
|------------|--------|
| **Entry point** | `remask_after_block_forward()` in `dllm_plugin.remasking.handoff`, invoked from `DllmWorker.run_one_block()` in `dllm_plugin.worker` (issue #10). |
| **Logits shape** | Per-block **2-D** `(DRAFT_SIZE, vocab_size)` (or equivalent nested rows). Validated by `assert_block_logits_shape`. Row `i` aligns with `input_draft[i]` and `scheduled_spec_decode_tokens[i]`. |
| **Pipeline parallel** | Only the **last** PP stage yields non-`None` logits for remasking; non-last ranks must **not** call `remask_after_block_forward` (`logits is None`). |
| **Worker ↔ engine mapping** | Policy output maps to `sampled_token_ids` (committed subset, length 0..`DRAFT_SIZE`) and the draft return path for the next block (length **`DRAFT_SIZE`**). See `DESIGN_MVP.md` sections 6–7. |
| **Batch dimension** | Helper assumes **no** leading batch axis `(batch, DRAFT_SIZE, vocab)`; batched vLLM runners slice per request upstream (issue #10 notes). |

Runtime adapters (`runtime_worker.py`) may consume **`dllm_block_logits`** from model output with **equivalent per-row semantics** to the above when models expose scores without storing raw tensors on `ModelRunnerOutput`.

## Commit-0 vs inner denoise stepping

These are **different** controls and must not be conflated in reviews:

- **Commit-0 (scheduler, `DESIGN_MVP` §6):** If a decode step produces **no** committed tokens in `sampled_token_ids`, the **plugin scheduler** rolls back `num_computed_tokens` by the number of tokens scheduled that step (typically `DRAFT_SIZE`). This is an **engine/schedule** contract.
- **Inner denoise (policy, `Llada2DefaultRemaskingPolicy`):** A **single** `RemaskingPolicy.apply` call may transfer up to a **transfer count** derived from `denoise_steps` / `denoise_step_index` / `num_transfer` in `remasking_config` (see table below). That is **remasking policy** schedule, not the scheduler’s commit-0 rule.

## `remasking_config` for `Llada2DefaultRemaskingPolicy`

Optional mapping passed into `remask_after_block_forward(..., remasking_config=...)`
and forwarded to `Llada2DefaultRemaskingPolicy.apply` (see
`dllm_plugin.remasking.llada2_default` for full semantics):

| Key | Type | Role |
|-----|------|------|
| `commit_confidence_threshold` | `float` | Masked positions are high-confidence when softmax prob at argmax is **strictly** greater than this threshold. |
| `mask_token_id` | `int` | Mask placeholder token id in drafts. |
| `denoise_steps` | `int` | Schedule length for transfer counts. |
| `denoise_step_index` | `int` | Zero-based index into that schedule (production should advance per step). |
| `num_transfer` | `int` | If set, overrides schedule-derived transfer count. |

## One decode step (ASCII)

```text
Engine -> DllmScheduler: read spec_token_ids (next block)
DllmScheduler -> DllmScheduler: set scheduled_spec_decode_tokens,
                num_scheduled_tokens == DRAFT_SIZE
Engine -> DllmWorker: SchedulerOutput
DllmWorker -> Model: forward one block (+ KV)
Model -> RemaskingPolicy: logits / scores
RemaskingPolicy -> DllmWorker: committed_token_ids (0..DRAFT_SIZE),
                     next_input_block (length DRAFT_SIZE)
DllmWorker -> Engine: sampled_token_ids = Committed;
              DraftTokenIds for next block
Engine -> DllmScheduler: update_from_output; commit-0 rollback if empty commit
Engine -> Engine: post_step take_draft_token_ids / update_draft_token_ids
DllmScheduler -> DllmScheduler: spec_token_ids := next block
```

## Remasking handoff (section 8)

`RemaskingPolicy.apply` consumes the current **input block** and model outputs,
and returns **committed** ids plus a **fixed-length** next input block
(`RemaskStepResult` from `dllm_plugin.remasking`). Length of
`next_input_block` must equal **`DRAFT_SIZE`**. The dataclass does not enforce
that at construction; call `validate_remask_step_result()` after `apply`
returns at the worker/policy boundary (see `DESIGN_MVP.md` section 8).

**Runtime score source:** Runtime worker remask handoff consumes
model-provided score rows from `ModelRunnerOutput` when available. For the
mock architecture path, fallback logits follow the mock model contract
(`compute_logits`: deterministic row where token id `0` has score `1.0`, others
`0.0`) instead of synthesizing from sampled token ids.

**Tuples vs lists:** `RemaskStepResult` fields are `tuple[int, ...]` for
immutability. Design prose may say `list[int]`; worker code should convert at
vLLM / engine boundaries when an API expects a mutable list.

**`isinstance` and `RemaskingPolicy`:** With `@runtime_checkable`, `isinstance(x,
RemaskingPolicy)` only confirms that `apply` exists and is callable. It does not
prove keyword-only calling conventions, return types, or correct behavior; rely
on tests and type checkers for that.

**Validator and dynamic block size:** `validate_remask_step_result()` compares
against `dllm_plugin.config.DRAFT_SIZE` only. If the stack ever uses a
per-request block length, this helper must gain an explicit length parameter (or
a replacement); otherwise it becomes misleading.

## See also

- `docs/DESIGN_MVP.md` sections 6-8 (canonical diagrams and tables)
- `dllm_plugin.config` for `DRAFT_SIZE` and related constants
- **`dllm_plugin/remasking/`** — normative implementations (`handoff.py`,
  `llada2_default.py`, `base.py`) for forward→remasking behavior beyond this
  summary; **`docs/ROADMAP.md`** (Phase 7) for real-model weights and extended
  policy work when it diverges from mock MVP docs here.
