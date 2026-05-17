# Numerical Investigation: dllm-plugin vs dInfer

## Summary

A systematic E2E investigation compared the dllm-plugin (vLLM MRV2) against
dInfer's `BlockDiffusionLLM` reference implementation for LLaDA2.0-mini.

**Result:** The plugin's remasking logic is **bit-identical** to dInfer across
all denoising steps when run on the same model. The only remaining difference
is a single BF16 rounding bit in one fused RMSNorm kernel value (1 out of
16,384 values), which washes out after 1 denoising iteration and does not
affect token decisions.

## Bugs Found and Fixed

### Bug 1: Draft tokens written after combine kernel reads them

`model_state.prepare_inputs()` wrote draft tokens to `req_states.draft_tokens`
AFTER the `combine_sampled_and_draft_tokens` kernel had already read zeros.

**Fix (fork):** Write draft tokens in `take_draft_token_ids()` at
`model_runner.py:~1286`, before the next step's combine kernel runs.

### Bug 2: Bootstrap `num_sampled` off-by-one

`_bootstrap()` returned `num_sampled=0`. The `post_update` kernel computed
`num_rejected = num_logits - 0 = 1`, advancing `nct_gpu` by `query_len - 1`
instead of `query_len`. Positions started at 3 instead of 4.

**Fix:** `_bootstrap()` uses the base sampler's `num_sampled` instead of zeros.
([diffusion_sampler.py](../dllm_plugin/sampling/diffusion_sampler.py))

### Bug 3: Commit-0 rollback during prefill

`_pre_schedule_nct` captured `nct=0` before prefill. After the prefill step
(no committed tokens from DiffusionSampler), the rollback restored `nct=0`,
causing prompt re-processing (36 tokens instead of 32 on the next step).

**Fix:** Only capture `_pre_schedule_nct` for requests where
`request.num_computed_tokens >= num_prompt_tokens`.
([runtime_scheduler.py](../dllm_plugin/runtime_scheduler.py))

### Bug 4: Frozen prefix KV on first block

The prefill computes KV for positions 0-3 with 4-token context. dInfer
recomputes the full 32-token block on every iteration via `replace_position`.
Without this, the model repeats prompt tokens ("TheTheTheThe of of of...").

**Fix:** Slot-mapping remap in `prepare_attn()` targets positions `[0..31]`
instead of `[prefix..prefix+31]`, overwriting the frozen prefix KV. The
remapped slot_mappings are stored as `_remapped_slot_mappings` for the fork
to rebuild `slot_mappings_by_layer` (the KV cache write path).
([llada2_model_state.py](../dllm_plugin/models/llada2_model_state.py))

**Fork patch:** After `model_state.prepare_attn()`, check for
`_remapped_slot_mappings` and rebuild `slot_mappings_by_layer`:
```python
# In execute_model(), after attn_metadata = self.model_state.prepare_attn(...):
remapped = getattr(self.model_state, '_remapped_slot_mappings', None)
if remapped is not None:
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        remapped, self.kv_cache_config)
    self.model_state._remapped_slot_mappings = None
```

### Bug 5: Off-by-one in `num_computed_tokens` after block commit

The parent scheduler's spec-decode logic assumes `bonus_tokens=1`, causing
`num_computed_tokens` to be 31 instead of 32 after committing 28 tokens.
This produced 33-token batches and position drift at block 2+.

**Fix:** Override `num_computed_tokens = prompt_len + dllm_state.committed`
after `super().update_from_output()`.
([runtime_scheduler.py](../dllm_plugin/runtime_scheduler.py))

### Additional: KV refresh step and matched remasking

- **KV refresh:** After all masks resolve, one extra Commit-0 forward refreshes
  the KV cache with the final committed tokens before the next block starts.
  This matches dInfer's cross-block update.
  ([diffusion_sampler.py](../dllm_plugin/sampling/diffusion_sampler.py))

- **Matched remasking:** The plugin's `batched_remask()` now exactly reproduces
  dInfer's `get_transfer_index_threshold()`: adaptive threshold with
  `>=` comparison, mask-ID prediction guard, and multi-position force-commit.
  Verified bit-identical across all 19 denoising steps on the same model.

## Required Fork Patches

Two patches to `vllm/v1/worker/gpu/model_runner.py`:

### Patch 1: `take_draft_token_ids` — write draft tokens to GPU buffer

```python
def take_draft_token_ids(self) -> DraftTokenIds | None:
    custom = self.model_state.take_draft_token_ids()
    if custom is not None:
        for i, req_id in enumerate(custom.req_ids):
            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is not None:
                tokens = custom.draft_token_ids[i]
                n = min(len(tokens), self.req_states.draft_tokens.shape[1])
                self.req_states.draft_tokens[req_idx, :n] = torch.tensor(
                    tokens[:n], dtype=self.req_states.draft_tokens.dtype,
                    device=self.req_states.draft_tokens.device)
        return custom
    return self.draft_tokens_handler.get_draft_tokens()
```

### Patch 2: `execute_model` — rebuild slot_mappings from remap

After `attn_metadata = self.model_state.prepare_attn(...)`, add:

```python
# attn_metadata is a dict[str, AttentionMetadata]. Check for remap key.
remapped = None
for group_meta in attn_metadata.values():
    if isinstance(group_meta, dict) and "remapped_slot_mappings" in group_meta:
        remapped = group_meta.pop("remapped_slot_mappings")
        break
if remapped is None:
    remapped = attn_metadata.pop("remapped_slot_mappings", None)
if remapped is not None:
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        remapped, self.kv_cache_config)
```

## Reproduction Guide

### Prerequisites

- Kubernetes cluster with A100 GPUs
- `kubectl` access
- Model: `inclusionAI/LLaDA2.0-mini`

### 1. Create pods

```bash
kubectl apply -f tools/investigation/pod-dinfer.yaml
kubectl apply -f tools/investigation/pod-vllm.yaml
kubectl wait --for=condition=Ready pod/investigation-dinfer --timeout=300s
kubectl wait --for=condition=Ready pod/investigation-vllm --timeout=300s
```

### 2. Set up pods

```bash
bash tools/investigation/setup_dinfer_pod.sh
bash tools/investigation/setup_vllm_pod.sh
```

Apply the two fork patches (§ Required Fork Patches) to the vLLM pod's
`model_runner.py`.

### 3. Capture E2E sub-operations

The capture scripts are in `tools/investigation/scripts/` (gitignored, kept
locally). Copy them to the pods via the setup scripts, then:

```bash
# dInfer pod (captures first denoising iteration at every layer)
kubectl exec investigation-dinfer -- python3 /workspace/scripts/capture_e2e_dinfer.py

# vLLM pod
kubectl exec investigation-vllm -- python3 /workspace/scripts/capture_e2e_vllm.py
```

### 4. Compare captures

Copy dInfer captures to the vLLM pod, then:
```bash
kubectl exec investigation-vllm -- python3 /workspace/scripts/compare_e2e.py \
    --captures-dir /workspace/captures/e2e
```

### 5. Verify remasking equivalence

Run both remasking implementations on the same model forward outputs (on the
dInfer pod with both `dinfer` and the plugin's `batched_remask` loaded).
Both should produce bit-identical token decisions at every denoising step.

## Equivalence Proof

### Remasking: Bit-identical

On the same model with the same input, both implementations produce
**identical token decisions at every denoising step** (verified across all
19 steps of block 0 for "The quick brown fox"). Drafts match at every step,
same tokens committed in the same order, same final output.

### 1-ULP perturbation impact

Perturbing layer 0 `k_norm` by 1 BF16 ULP (7.8e-3 at one position):

| Metric | Value |
|--------|-------|
| Max abs logit diff | 2.0 |
| Mean cosine similarity | 0.996 |
| Max KL divergence | 0.073 |
| Top-1 agreement | 93.8% (30/32) |
| Top-5 overlap | 86.9% |

The 2 disagreeing positions swap between EOS and newline tokens at ~5%
probability — low-confidence masked positions where the top prediction is
unstable regardless.

### Multi-step washout

Running dInfer twice (normal vs 1-ULP perturbed) across a full 2-block
generation (66 forwards):

- **Step 0** (perturbation injected): top-1 agreement 68.8%
- **Steps 1-65**: top-1 agreement **100%**, MaxDiff **0.000**

The KV refresh mechanism completely erases the perturbation after 1 iteration.
Both runs produce identical text.
