# Phase 9: Numerical Validation and Generation Quality

## Summary

This document describes the findings from the Phase 9 numerical investigation
comparing dInfer (reference) and vLLM dllm-plugin implementations of
LLaDA2.0-mini, and the fixes applied to achieve coherent text generation.

## Bugs Found and Fixed

### 1. MoE Routing (Critical)

**File:** `dllm_plugin/models/llada2.py`

`FusedMoE` was initialized without LLaDA2-specific routing parameters. It
defaulted to softmax top-k instead of LLaDA2's sigmoid group-limited routing
with 2.5x scaling. The pre-computed routing in `LLaDA2MoE.forward()` was
passed as `router_logits` to `FusedMoE`, which re-routed with its own softmax
defaults — completely ignoring the plugin's sigmoid/group-limited logic.

**Fix:** Pass `scoring_func="sigmoid"`, `use_grouped_topk=True`,
`num_expert_group`, `topk_group`, `routed_scaling_factor=2.5` to `FusedMoE`
constructor. Simplify forward to pass raw gate logits.

**Impact:** E2E logits cos_sim went from 0.85 to 0.98. Top-1 token match
went from 0% to 100%. KL divergence dropped from 4.78 to 0.08.

### 2. FlashInfer Attention Backend (Critical)

**File:** `dllm_plugin/models/llada2_attention.py`, `llada2_attention_backend.py`

The `_forward_concatenated()` method injected raw `CommonAttentionMetadata`
into the per-layer metadata dict, but `Attention` expects already-built
backend-specific metadata (e.g., `FlashInferMetadata` with `.plan()`-
initialized wrappers). This crashed with FlashInfer.

**Fix:** Moved concatenation into the builder's `build()` method where it
operates on `CommonAttentionMetadata` before the backend-specific
transformation. The builder now handles both `causal=False` and prefix+block
page concatenation. The attention forward always uses the simple
`self.attn(q, k, v)` path.

### 3. Mask Token ID (Critical)

**File:** `dllm_plugin/config.py`

`LLADA2_DEFAULT_MASK_TOKEN_ID` was `1` (an arbitrary token). The correct
LLaDA2.0 mask token is `156895` (`<|mask|>`). With the wrong mask token,
the model never saw the correct mask pattern during diffusion denoising.

**Fix:** Changed to `156895`. Confirmed via `tokenizer.decode([156895])` =
`'<|mask|>'` and 6+ validation scripts in `tools/validation/`.

### 4. Confidence Threshold

**File:** `dllm_plugin/config.py`

`LLADA2_DEFAULT_COMMIT_CONFIDENCE_THRESHOLD` was `0.01` (1%). dInfer uses
`0.9` (90%) in its `ThresholdParallelDecoder`. With 0.01, all positions
commit on the first forward pass — no iterative refinement.

**Fix:** Changed to `0.9` to match dInfer.

### 5. Commit-0 Rollback (Critical)

**File:** `dllm_plugin/runtime_scheduler.py`

Per DESIGN_MVP.md §6.1, when `sampled_token_ids` is empty (Commit-0), the
scheduler must roll back `num_computed_tokens` by the number of scheduled
tokens. Without this, vLLM's optimistic accounting diverges and the engine
stalls — `num_computed_tokens` keeps growing but no tokens are actually
committed, so the scheduler never re-schedules the block.

**Fix:** In `update_from_output()`, after calling `super()` (which increments
optimistically), detect empty `sampled_token_ids` for requests with
`scheduled_spec_decode_tokens` and roll back `request.num_computed_tokens`.

### 6. Draft Token Buffer (Critical)

**File:** `dllm_plugin/gpu_model_runner.py`

vLLM's `req_states.draft_tokens` buffer is only populated by the speculator
(standard spec-decode). The dLLM scheduler provides draft tokens via
`scheduled_spec_decode_tokens`, but these were never copied into the buffer.
The `combine_sampled_and_draft_tokens` Triton kernel read zeros instead of
mask tokens.

**Fix:** Override `prepare_inputs()` to copy scheduler's spec tokens into
`req_states.draft_tokens` before the kernel runs. Also resize the buffer
from `[max_reqs, 0]` to `[max_reqs, DRAFT_SIZE]` via `_resize_for_draft_blocks()`.

### 7. Prompt Prefix Stripping

**File:** `dllm_plugin/gpu_model_runner.py`

The first block contains `[prompt_suffix, mask, mask, ...]`. When committed,
prompt positions were included in the output as "generated" tokens.

**Fix:** Record the initial prompt length at the start of each block's
denoising. At commit time, strip leading prompt tokens from `committed_token_ids`.

### 8. AR Prefill Suppression

**File:** `dllm_plugin/gpu_model_runner.py`

The first engine step is a standard AR prefill (no draft tokens) that
produces 1 spurious token before block diffusion starts.

**Fix:** Zero out `sampled_token_ids` and `num_sampled` for dLLM prefill
steps (when no `scheduled_spec_decode_tokens` exist).

## Current Performance

Tested on A100-40GB with LLaDA2.0-mini (30GB BF16):

| Metric | Value | Notes |
|--------|-------|-------|
| Output TPS | 1.5 tok/s | Limited by iterative denoising |
| TTFT | ~43s | ~28 forward passes per first block |
| Forward passes/block | ~28 | 1 token transferred per step |
| Quality | Coherent English | Comparable to dInfer |

## Required CLI Flags

```bash
vllm serve inclusionAI/LLaDA2.0-mini \
  --trust-remote-code --enforce-eager \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --no-async-scheduling
```

## Remaining Optimizations

- **CUDA graphs / torch.compile:** Would reduce forward pass from ~1s to
  ~10ms, improving TPS by ~100x. Not yet validated with dLLM block path.
- **Async scheduling:** Commit-0 rollback not yet implemented for async mode.
- **FusedMoE tuned config:** Missing A100-specific config file for 256-expert MoE.

## vLLM Fork Dependency

Branch `dllm-fork` on `AlonKellner-RedHat/vllm` (4 commits on v0.20.2):
- FlashInfer `causal=False` support (removed hardcoded assertions)
- `CommonAttentionMetadata.causal` flag propagation from `attention_config`
- `gpu_worker._model_runner_cls` support for plugin model runners
- Scheduler spec_token_ids handling for dLLM blocks
