# Upstream vLLM Issues to File

This document tracks vLLM integration issues discovered during Phase 7+8 implementation that require upstream fixes or enhancements.

**Status:** To be filed by maintainer  
**Date:** 2026-05-07

---

## Issue 1: ModelRegistry Should Be Checked During ModelConfig Validation

**Priority:** P1  
**Component:** Plugin Architecture / Model Registry

**Problem:**

vLLM's `ModelConfig` validation does not check the `ModelRegistry` during initialization, requiring workarounds like the `model_impl` parameter in tests to bypass validation for plugin-registered architectures.

**Current Workaround:**

```python
# test_llada2_gpu_integration.py line 53-54
# TODO: File vLLM issue requesting ModelRegistry lookup during validation
# TODO: Remove model_impl workaround once vLLM properly supports plugin architectures
vllm_config = vllm.config.VllmConfig(
    model_config=model_config,
    model_impl=self._get_model_impl(llada2_config.architectures[0]),  # Workaround
    ...
)
```

**Desired Behavior:**

`ModelConfig` validation should check `ModelRegistry` for plugin-registered architectures before failing with "unsupported architecture" errors.

**Impact:**

- Fragile test setup requiring manual `model_impl` parameter
- Poor developer experience for plugin authors
- Risk of misconfiguration in production

**Upstream PR Needed:**

Modify `vllm/config.py` `ModelConfig` validation to:
1. Check if architecture is registered in `ModelRegistry`
2. Only fail if architecture is both unknown AND not in registry
3. Document plugin registration workflow

---

## Issue 2: Plugin Architectures Should Support Custom Attention Patterns

**Priority:** P2  
**Component:** Attention Layer / Plugin API

**Problem:**

vLLM's attention backend does not provide a clean API for plugins to implement custom attention patterns (e.g., block-style attention with non-causal chunks). Current implementation requires:
- Manual `CommonAttentionMetadata` transformation
- Hardcoded KV cache block size assumptions
- No validation that slot_mapping behavior is correct for custom patterns

**Current Workaround:**

```python
# virtual_batches.py - Manual metadata transformation
def make_block_attention_virtual_batches(...):
    # Manually slice block_table_tensor
    # Assume kv_block_size = 16
    # Reuse slot_mapping without verification
```

**Desired Behavior:**

vLLM should provide:
1. Public API for custom attention pattern registration
2. Helper methods for KV cache slicing with correct block size querying
3. Validation hooks for custom attention metadata

**Impact:**

- Brittle implementation relying on internal vLLM assumptions
- Risk of breaking when vLLM changes KV cache layout
- No way to verify correctness of custom patterns

**Upstream Enhancement Needed:**

Add to vLLM:
```python
class CustomAttentionPattern:
    def transform_metadata(self, metadata: CommonAttentionMetadata) -> ...:
        pass
    
    def validate_metadata(self, metadata: CommonAttentionMetadata) -> None:
        pass

# Helper methods
def get_kv_cache_block_size(cache_config) -> int:
    pass
```

---

## Issue 3: Document KV Cache Block Size Configuration

**Priority:** P3  
**Component:** Documentation

**Problem:**

vLLM's KV cache block size is hardcoded as 16 in most examples, but there's no public documentation on:
- How to query the actual block size from `cache_config`
- Whether block size is configurable
- What happens if block size changes in future versions

**Current Workaround:**

```python
# virtual_batches.py line 80
kv_block_size = 16  # Standard vLLM block size
# Note: This assumes uniform block size; may need adjustment
```

**Desired Behavior:**

vLLM documentation should include:
1. How to query KV cache block size from configuration
2. Whether users can configure block size
3. Compatibility guarantees for block size changes

**Impact:**

- Plugins making assumptions about block size may break silently
- No programmatic way to query the correct value

**Upstream Documentation Needed:**

Add to vLLM docs:
```markdown
## KV Cache Configuration

### Block Size

The KV cache uses a block size of 16 tokens by default. To query the configured block size:

```python
from vllm.config import CacheConfig

block_size = cache_config.block_size  # or equivalent API
```

### Future Compatibility

vLLM maintains backward compatibility for KV cache block size. If the default changes,
plugins can query the configured value programmatically.
```

---

## Issue 4: MultiProcessing V1 vs V2 Model Runner Confusion

**Priority:** P3  
**Component:** Documentation / Environment Variables

**Problem:**

The relationship between `VLLM_USE_V2_MODEL_RUNNER`, `VLLM_ENABLE_V1_MULTIPROCESSING`, and plugin compatibility is not well documented, leading to configuration errors.

**Current State:**

```bash
# Required for dllm plugin, but not obvious why
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

**Desired Behavior:**

vLLM documentation should clearly explain:
1. Difference between V1 and V2 model runners
2. Which plugins require which runner
3. Compatibility matrix for environment variables

**Impact:**

- Confusion during setup
- Hard-to-debug errors when wrong runner is used

**Upstream Documentation Needed:**

Add to vLLM plugin development guide explaining runner architecture and configuration.

---

## Summary

| Issue | Priority | Component | Workaround Complexity | Upstream PR/Doc Needed |
|-------|----------|-----------|----------------------|----------------------|
| ModelRegistry validation | P1 | Core | Medium | PR + Tests |
| Custom attention API | P2 | Attention | High | Design Doc + PR |
| KV cache block size docs | P3 | Documentation | Low | Docs only |
| Runner configuration docs | P3 | Documentation | Low | Docs only |

**Total estimated upstream effort:** 2-3 weeks (for vLLM maintainers)

---

## Notes for Maintainer

When filing these issues:
1. Reference this document and PR #38
2. Link to specific workaround code locations
3. Provide concrete API proposals (not just problems)
4. Offer to contribute PRs if vLLM team approves design

**Tracking:** Update this document with issue URLs once filed
