# Validation Instructions for Softmax Normalization Fix

**Status:** Implementation complete, ready for GPU validation  
**Date:** 2026-05-12

---

## Quick Start

### Option 1: Automated Validation Script

Run the complete validation suite:

```bash
# On a GPU-enabled environment with vLLM installed
./tools/validate_softmax_fix.sh
```

This script will:
1. Run attention checkpoint validation (C8)
2. Run token generation validation
3. Run full numerical validation suite
4. Generate summary report

### Option 2: Manual Step-by-Step Validation

#### Step 1: Unit Tests (No GPU Required)

Test metadata construction logic:

```bash
pytest tests/test_concatenated_virtual_batch.py -v
```

Expected output:
```
✅ Concatenated metadata constructed correctly
✅ First block (no prefix) handled correctly
✅ Heterogeneous prefix lengths handled correctly
```

#### Step 2: Attention Checkpoint Validation (Requires GPU)

Test that C8 attention divergence is fixed:

```bash
pytest tests/test_llada2_numerical_validation.py::TestAttentionValidation::test_attention_output_matches_reference -v -s
```

**Expected result:**
- **Before fix:** max_diff = 1.188 ❌
- **After fix:** max_diff < 0.01 ✅

#### Step 3: Token Generation Validation (Requires GPU)

Test that vLLM generates correct tokens matching dInfer:

```bash
pytest tests/test_llada2_numerical_validation.py::TestE2EValidation::test_e2e_generation_matches_reference -v -s
```

**Expected result:**
- **Before fix:** vLLM generates token 198 ("\n") ❌
- **After fix:** vLLM generates token 30 ("?") matching dInfer ✅

#### Step 4: Full Validation Suite (Requires GPU)

Run all numerical validation tests:

```bash
pytest tests/test_llada2_numerical_validation.py -v
```

Expected: All checkpoints pass within tolerance.

---

## Running in Kubernetes (GPU Environment)

### Setup Validation Pod

```bash
# Create validation pod with GPU
kubectl apply -f tools/k8s/debug-pod-a100-phase91.yaml

# Wait for pod to be ready
kubectl wait --for=condition=ready pod/dllm-plugin-validation-a100 --timeout=300s

# Exec into pod
kubectl exec -it dllm-plugin-validation-a100 -- /bin/bash
```

### Run Validation Inside Pod

```bash
# Inside the pod
cd /workspace/dllm-plugin

# Option 1: Run automated script
./tools/validate_softmax_fix.sh

# Option 2: Run individual tests
pytest tests/test_concatenated_virtual_batch.py -v
pytest tests/test_llada2_numerical_validation.py::TestAttentionValidation -v
pytest tests/test_llada2_numerical_validation.py::TestE2EValidation -v
```

### Collect Results

```bash
# Copy results from pod to local machine
kubectl cp dllm-plugin-validation-a100:/tmp/validation_c8_results.txt ./validation_results/c8_results.txt
kubectl cp dllm-plugin-validation-a100:/tmp/validation_token_results.txt ./validation_results/token_results.txt
kubectl cp dllm-plugin-validation-a100:/tmp/validation_full_results.txt ./validation_results/full_results.txt
```

---

## Expected Results Summary

### Before Fix (Dual-Chunk - BUGGY)

| Test | Metric | Value | Status |
|------|--------|-------|--------|
| C8 Attention | max_diff | 1.188 | ❌ FAIL |
| Token Generation | Predicted Token | 198 ("\n") | ❌ WRONG |
| Attention Weights | Sum | ~2.0 | ❌ WRONG |
| Mathematical Property | Softmax Count | 2 (separate) | ❌ WRONG |

**Root Cause:** TWO separate softmax normalizations cause weights to sum to 2.0

### After Fix (Concatenated - CORRECT)

| Test | Metric | Value | Status |
|------|--------|-------|--------|
| C8 Attention | max_diff | < 0.01 | ✅ PASS |
| Token Generation | Predicted Token | 30 ("?") | ✅ CORRECT |
| Attention Weights | Sum | 1.0 | ✅ CORRECT |
| Mathematical Property | Softmax Count | 1 (unified) | ✅ CORRECT |

**Fix:** ONE softmax normalization over ALL keys (prefix + block)

---

## Validation Checklist

- [ ] Unit tests pass (metadata construction)
- [ ] C8 attention divergence < 0.01
- [ ] Token generation matches dInfer (token 30)
- [ ] Full validation suite passes
- [ ] No regressions in other checkpoints
- [ ] Debug logging confirms single softmax
- [ ] Performance benchmarks (no slowdown)

---

## Debugging Failed Validation

### If C8 Divergence Still High (max_diff > 0.01)

Check debug logs for:

```python
# Should see this in logs:
[CONCATENATED ATTN] Using concatenated virtual batch (FIXED)
[CONCATENATED ATTN] Prefix lengths: [64, 32]
[CONCATENATED ATTN] Query shape: torch.Size([64, 4096])
[CONCATENATED ATTN] Single softmax normalization ✅
```

If you see instead:

```python
[DUAL CHUNK BUGGY] Combined prefix + block outputs (weights sum to 2.0)
```

Then the old code path is still being used. Check that:
1. `_forward_concatenated` is being called (not `_forward_dual_chunk_BUGGY`)
2. Forward context is correctly overridden with concatenated metadata

### If Token Still Wrong (token 198 instead of 30)

This indicates attention is still incorrect. Debug steps:

1. Check that `num_prefix_tokens_list` is not None
2. Verify concatenated metadata has correct seq_lens
3. Verify block_table is correctly concatenated
4. Check FlashAttention is reading from correct cache pages

### If Import Errors

```bash
# Ensure vLLM is installed in environment
pip install -e .

# Check module imports
python -c "from dllm_plugin.attention.concatenated_virtual_batch import create_concatenated_virtual_batch; print('OK')"
python -c "from dllm_plugin.models.llada2_attention import LLaDA2BlockAttention; print('OK')"
```

---

## Performance Validation

The fix should NOT introduce performance regressions:

```bash
# Benchmark before and after
pytest tests/benchmarks/ -v

# Expected: Similar or better performance
# (Single attention call may be faster than two separate calls)
```

---

## Files Modified by Fix

1. **New:** `dllm_plugin/attention/concatenated_virtual_batch.py`
   - Creates single virtual batch combining prefix + block KV

2. **Modified:** `dllm_plugin/models/llada2_attention.py`
   - Added `_forward_concatenated()` method
   - Updated forward call to use concatenated approach
   - Deprecated old `_forward_dual_chunk_BUGGY()` method

3. **New:** `tests/test_concatenated_virtual_batch.py`
   - Unit tests for metadata construction

4. **New:** `tools/validate_softmax_fix.sh`
   - Automated validation script

---

## Documentation Updates Needed After Validation

Once validation passes:

1. Update `docs/ATTENTION_DESIGN.md`
   - Correct the mathematical claim about dual-chunk equivalence
   - Document concatenated virtual batch approach

2. Update `CHANGELOG.md`
   - Add entry for softmax normalization bug fix

3. Remove deprecated code
   - Delete `_forward_dual_chunk_BUGGY()` method after confirmation

4. Update validation results
   - Document final max_diff values
   - Include before/after comparison

---

## Contact & Support

**Issue:** Phase 9.1 - Numerical Validation  
**Fix:** Softmax Normalization Bug (Concatenated Virtual Batch)  
**Plan:** `/Users/akellner/.claude/plans/let-s-plan-phase-7-agile-mochi.md`  
**Analysis:** `/tmp/ROOT_CAUSE_SOFTMAX_NORMALIZATION.md`

For questions or issues, review:
- `SOFTMAX_NORMALIZATION_FIX.md` - Implementation summary
- `ROOT_CAUSE_SOFTMAX_NORMALIZATION.md` - Root cause analysis
- Plan document - Full investigation and implementation plan
