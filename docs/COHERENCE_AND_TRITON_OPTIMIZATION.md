# Coherence Fix + Triton Remasking Optimization

## Summary

Two issues were identified and fixed that produced incoherent multi-block
output from the dllm-plugin. A Triton remasking kernel was also integrated
as a performance optimization with verified numerical equivalence.

## Issue 1: Causal attention in FlashInfer (root cause of multi-block degradation)

**Symptom**: Block 1 produced different tokens from dInfer at temperature=0.
Blocks 2+ degraded into repetition ("A violin of gold" repeated, "of of of...").

**Root cause**: FlashInfer's `_use_causal` flag was `True` because
`attention_config.use_non_causal` was never set for diffusion models.
The plugin set `causal=False` in `CommonAttentionMetadata`, but FlashInfer
ignores per-step metadata and uses its init-time `_use_causal` flag.

**Evidence**: Per-token attention output comparison showed the classic
causal signature — position 0 (cos=0.77, worst) vs position N-1
(cos=0.9999, near-exact). In bidirectional attention, all positions
should have similar divergence.

**Fix** (fork `vllm/config/vllm.py`): Set `model_config._use_non_causal`
and `attention_config.use_non_causal` when `DiffusionConfig` is detected.
Committed as `4019579` on `dllm-fork-coherent`.

## Issue 2: Missing fork patches for KV cache pipeline

**Symptom**: First-block output was garbled ("TheTheThe of of of...").

**Root cause**: Two patches from `docs/NUMERICAL_INVESTIGATION.md` were
documented but never committed to the fork:

- **Patch 1** (`take_draft_token_ids`): Write draft tokens to
  `req_states.draft_tokens` GPU buffer so the combine kernel gets
  correct input on subsequent decode steps.
- **Patch 2** (`execute_model`): Rebuild `slot_mappings_by_layer` from
  `remapped_slot_mappings` returned by `prepare_attn()`, enabling
  first-block full recomputation to write KV cache at correct positions.

**Fix** (fork `vllm/v1/worker/gpu/model_runner.py`): Both patches
applied. Committed as `3f3b89a` on `dllm-fork-coherent`.

## Triton remasking kernel optimization

**What**: Fused two-kernel design replaces 7 separate PyTorch kernel
launches (softmax, argmax, gather, squeeze, where×2, clone) with:
- Kernel 1: Online softmax + argmax per (batch, position) over 157K vocab
- Kernel 2: Cross-position max reduction + threshold + commit per batch

**Integration**: `DiffusionSampler.__init__` selects the Triton kernel
via `try/except` import, falling back to PyTorch if unavailable. The
call site in `_denoise()` uses `self._remask_fn()`.

**Equivalence**: 8 test cases verify exact bit-equality of all 3 outputs
(draft tokens, all_done, num_transferred) between PyTorch and Triton
implementations. Tests cover: random inputs (20 seeds), all-masked,
no-masked, single-mask, argmax-is-mask-token, batch size 8, low/high
threshold. All pass on A100.

**E2E verification**: Output is identical with Triton kernel enabled —
same token counts, same text, same coherence across 3 test prompts.

## Capture-replay methodology

The investigation used the existing capture-replay infrastructure:

1. `capture_e2e_dinfer.py` — captures all layer sub-operations from
   dInfer's first denoising iteration
2. `capture_e2e_vllm.py` — captures the same from the vLLM plugin
3. `compare_e2e.py` — systematic comparison at 245 checkpoints
4. `capture_attention_deep.py` — A1-A9 attention sub-operation captures
   (Q/K norm, post-RoPE, attention output, o_proj)

Per-token attention output comparison confirmed the causal attention
signature and verified the fix produced bidirectional behavior.

## Fork branch

All fork changes are on `dllm-fork-coherent`:
- `6b08edd` — DiffusionConfig, bonus=0, non-causal attention support
- `3f3b89a` — Patch 1 + 2 (draft_tokens write, slot_mappings remap)
- `4019579` — Non-causal fix (use_non_causal for diffusion)
