# MVP field mapping (contributor reference)

Copy-friendly summary of **`docs/DESIGN_MVP.md` section 7** (field mapping) and
related invariants. Keep this file ASCII-only for terminals and PR review.

**Keeping docs aligned:** Edits to the field-mapping table or invariants in
`DESIGN_MVP.md` sections 5–8 (especially 6–8 and the forward→remasking note in
section 5) should be mirrored here, and substantive changes here should be
reflected back in `DESIGN_MVP.md`, so contributor copy and the canonical design
doc do not drift silently.

**Upstream vLLM identifiers:** The table below uses vLLM type and member names as
they appear for the **bounded** optional `vllm` dependency in `pyproject.toml`.
Those names can change in newer vLLM releases. When raising or widening that
pin, re-check upstream APIs and update this file and `DESIGN_MVP.md` together
(see plugin issue #2 for hook / minimum-version tracking).

## Spec-decode-shaped fields (plugin stack active)

| vLLM field / API | Role when dLLM plugin scheduler + worker are active |
|------------------|-----------------------------------------------------|
| `Request.spec_token_ids` | Next-step **`input_draft`** for the upcoming schedule. Length **`DRAFT_SIZE`** (see `vllm_dllm_plugin.config.DRAFT_SIZE`). |
| `SchedulerOutput.scheduled_spec_decode_tokens` | This step's **`input_draft`** for the forward. Length **`DRAFT_SIZE`**. |
| `SchedulerOutput.num_scheduled_tokens` (per request) | Set to **`DRAFT_SIZE`** for decode steps on the block path. |
| `ModelRunnerOutput.sampled_token_ids` | **Committed** token IDs only; length **0..`DRAFT_SIZE`** (may be empty). |
| Worker `take_draft_token_ids()` | Returns the next-step **`input_draft`** as `DraftTokenIds` for engine -> scheduler. |
| Scheduler `update_draft_token_ids` / `update_draft_token_ids_in_output` | Store the next block into `spec_token_ids`. **Must not** apply AR draft grammar to dLLM blocks (scheduler overrides for structured output / async). |

## Commit-0 rollback

If no tokens are committed in a step, the plugin scheduler rolls back
`num_computed_tokens` by the number of tokens scheduled that step (typically
**`DRAFT_SIZE`**). See `DESIGN_MVP.md` section 6 (sequence diagram) and section 1
(Commit-0 goal).

**dLLM default remasking:** `Llada2DefaultRemaskingPolicy` may return **empty**
`committed_token_ids` on inner denoise steps while the draft still contains the
configured mask token. That is **not** a failed step; the worker must not surface
those inner steps to commit-0 without an explicit contract exception, or it must
batch the inner loop into one engine step (see `DESIGN_MVP.md` section 6).

## Mutual exclusion

True speculative decoding on the same requests must **not** be combined with the
dLLM plugin stack (same run mode). See `DESIGN_MVP.md` section 7 (last paragraph).

## Forward to remasking handoff (issue #13)

After last-rank `compute_logits`, the worker maps block logits plus this step's
`input_draft` into `RemaskStepResult` via
`vllm_dllm_plugin.remasking.handoff.remask_after_block_forward` (must pass an
explicit `RemaskingPolicy`, e.g. `Llada2DefaultRemaskingPolicy` for the LLaDA2
MVP; see `DESIGN_MVP.md` section 5, forward outputs subsection).

- **Logits shape:** 2-D ``(DRAFT_SIZE, vocab_size)`` (or equivalent nested
  sequence). Row ``i`` matches ``input_draft[i]``. Non-last PP ranks return
  ``logits is None``; do not call the handoff there.
- **Dtype / device:** follow the runner logits tensor; the default policy reads
  float-like values per row.
- **Mock path:** deterministic stub outputs for stack tests are defined in
  issue #24 / ``docs/MOCK_STACK_MODEL.md`` and ``vllm_dllm_plugin.models.mock_llada2``.
- **Batched 3-D logits** (leading batch axis) are **out of scope** for the MVP
  helper; issue #10 may slice or wrap.

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

`RemaskingPolicy.apply` consumes the current **input draft** (`input_draft`) and model outputs,
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

**LLaDA2 default policy (`Llada2DefaultRemaskingPolicy`, issue #7):** optional
`remasking_config` keys: `commit_confidence_threshold` (float; a masked position
counts as **high-confidence** only when softmax probability at the argmax token
is **strictly greater** than this value — the same comparator as Hugging Face
Diffusers [`BlockRefinementScheduler.step`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_block_refinement.py)),
`mask_token_id` (int), `denoise_steps` (int), `denoise_step_index` (int),
`num_transfer` (int override of the per-step transfer budget; **not** in the
Diffusers API, for tests and tooling). Defaults in `vllm_dllm_plugin.config`. See
module docstring in `vllm_dllm_plugin.remasking.llada2_default`.

## See also

- `docs/DESIGN_MVP.md` sections 6-8 (canonical diagrams and tables)
- `vllm_dllm_plugin.config` for `DRAFT_SIZE` and related constants
