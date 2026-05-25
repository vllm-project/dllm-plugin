# dInfer/LLaDA2 transformers Version Requirement

**Date:** 2026-05-12  
**Required Version:** transformers 4.57.6

---

## Critical Dependency

The LLaDA2.0-mini model from `inclusionAI/LLaDA2.0-mini` requires **transformers 4.57.6** specifically.

### Installation

```bash
pip install transformers==4.57.6
```

### Why This Specific Version?

1. **RoPE Implementation Compatibility**
   - LLaDA2 model code was written for transformers 4.57.6 RoPE API
   - transformers 5.x introduced breaking changes to RoPE implementation
   - Using transformers 5.x results in: `AttributeError: 'LLaDA2MoeRotaryEmbedding' object has no attribute 'rope_type'`

2. **Model Code Assumptions**
   - The model's `modeling_llada2_moe.py` expects the 4.57.6 API
   - RoPE initialization in transformers 4.57.6 doesn't require `rope_type` attribute
   - Attention mask handling matches the expected dtype (bfloat16)

3. **Tested Configuration**
   - ✅ transformers 4.57.6 + torch 2.x + LLaDA2.0-mini: **WORKS**
   - ❌ transformers 5.x + torch 2.x + LLaDA2.0-mini: **FAILS**

---

## Troubleshooting

### Error: `AttributeError: 'LLaDA2MoeRotaryEmbedding' object has no attribute 'rope_type'`

**Cause:** Using transformers 5.x instead of 4.57.6

**Solution:**
```bash
# Uninstall current version
pip uninstall transformers -y

# Install correct version
pip install transformers==4.57.6

# Clear transformers cache
rm -rf ~/.cache/huggingface/modules/transformers_modules/llada2_hyphen_mini
```

### Error: `RuntimeError: invalid dtype for bias - should match query's dtype`

**Cause:** Attention mask dtype mismatch

**Solution:**
```python
# Convert attention_mask to match model dtype (bfloat16)
attention_mask = attention_mask.to(torch.bfloat16)
```

---

## Validation Pod Setup

All validation setup scripts have been updated to install transformers 4.57.6:

- `tools/validation/setup_dinfer_pod.sh`
- `tools/validation/install-dinfer-deps.sh`
- `tools/validation/comprehensive_validation_setup.sh`

The scripts will:
1. Install transformers 4.57.6 specifically
2. Verify the version is correct
3. Warn if a different version is detected

---

## References

- **Model:** [inclusionAI/LLaDA2.0-mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini)
- **Issue Documentation:** [docs/PHASE9.1_BLOCKERS.md](PHASE9.1_BLOCKERS.md)
- **Validation Status:** [DLLM_DINFER_VALIDATION_STATUS.md](../DLLM_DINFER_VALIDATION_STATUS.md)

---

## Version Compatibility Matrix

| transformers | LLaDA2 Model | Status | Notes |
|--------------|--------------|--------|-------|
| 4.57.6 | ✅ Works | **RECOMMENDED** | Exact match for model code |
| 4.45.0-4.57.5 | ⚠️ Untested | Unknown | May work but not verified |
| 5.0.0+ | ❌ Fails | NOT COMPATIBLE | RoPE API breaking changes |

---

**Last Updated:** 2026-05-12  
**Tested By:** dllm-plugin validation suite  
**Validation Pod:** validation-dinfer
