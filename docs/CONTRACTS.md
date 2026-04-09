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
| `Request.spec_token_ids` | **Next-step input block** for the upcoming schedule. Length **`DRAFT_SIZE`** (see `vllm_dllm_plugin.config.DRAFT_SIZE`). |
| `SchedulerOutput.scheduled_spec_decode_tokens` | **Input block** for **this** step's forward. Length **`DRAFT_SIZE`**. |
| `SchedulerOutput.num_scheduled_tokens` (per request) | Set to **`DRAFT_SIZE`** for decode steps on the block path. |
| `ModelRunnerOutput.sampled_token_ids` | **Committed** token IDs only; length **0..`DRAFT_SIZE`** (may be empty). |
| Worker `take_draft_token_ids()` | Returns the **next-step input block** as `DraftTokenIds` for engine -> scheduler. |
| Scheduler `update_draft_token_ids` / `update_draft_token_ids_in_output` | Store the next block into `spec_token_ids`. **Must not** apply AR draft grammar to dLLM blocks (scheduler overrides for structured output / async). |

## Commit-0 rollback

If no tokens are committed in a step, the plugin scheduler rolls back
`num_computed_tokens` by the number of tokens scheduled that step (typically
**`DRAFT_SIZE`**). See `DESIGN_MVP.md` section 6 (sequence diagram) and section 1
(Commit-0 goal).

## Mutual exclusion

True speculative decoding on the same requests must **not** be combined with the
dLLM plugin stack (same run mode). See `DESIGN_MVP.md` section 7 (last paragraph).

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
(`RemaskStepResult` from `vllm_dllm_plugin.remasking`). Length of
`next_input_block` must equal **`DRAFT_SIZE`**. The dataclass does not enforce
that at construction; call `validate_remask_step_result()` after `apply`
returns at the worker/policy boundary (see `DESIGN_MVP.md` section 8).

**Tuples vs lists:** `RemaskStepResult` fields are `tuple[int, ...]` for
immutability. Design prose may say `list[int]`; worker code should convert at
vLLM / engine boundaries when an API expects a mutable list.

**`isinstance` and `RemaskingPolicy`:** With `@runtime_checkable`, `isinstance(x,
RemaskingPolicy)` only confirms that `apply` exists and is callable. It does not
prove keyword-only calling conventions, return types, or correct behavior; rely
on tests and type checkers for that.

**Validator and dynamic block size:** `validate_remask_step_result()` compares
against `vllm_dllm_plugin.config.DRAFT_SIZE` only. If the stack ever uses a
per-request block length, this helper must gain an explicit length parameter (or
a replacement); otherwise it becomes misleading.

## See also

- `docs/DESIGN_MVP.md` sections 6-8 (canonical diagrams and tables)
- `vllm_dllm_plugin.config` for `DRAFT_SIZE` and related constants
