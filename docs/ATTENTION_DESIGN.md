# LLaDA2.0 Block-Style Attention Design

**Issue:** [#11](https://github.com/vllm-project/dllm-plugin/issues/11)  
**Phase:** 7 (Real Model Integration)  
**Author:** Phase 7 Implementation  
**Status:** Draft

---

## 1. Problem Statement

LLaDA2.0 uses a **block-style attention pattern** that differs fundamentally from standard causal (autoregressive) attention. Each position in the current generation block must attend to:

1. **All committed prefix tokens** (non-causal, full visibility)
2. **All tokens in the current block** (non-causal, bidirectional within block)

This enables parallel refinement within blocks while maintaining access to the full committed context, supporting LLaDA2.0's iterative diffusion-based generation approach.

**Challenge:** vLLM's attention layers are optimized for standard causal masking. We need to support block-style masks without custom CUDA kernels in the MVP.

---

## 2. Mask Geometry

### 2.1 Standard Causal Attention (Baseline)

Traditional autoregressive LLMs use strictly causal attention:
```
Position:  0  1  2  3  4  5
      0 [  1  0  0  0  0  0 ]
      1 [  1  1  0  0  0  0 ]
      2 [  1  1  1  0  0  0 ]
      3 [  1  1  1  1  0  0 ]
      4 [  1  1  1  1  1  0 ]
      5 [  1  1  1  1  1  1 ]
```
Each position attends only to itself and prior positions (lower triangular).

### 2.2 LLaDA2.0 Block-Style Attention

With block size `DRAFT_SIZE = 32`, generation proceeds in blocks. At step 2 (64 tokens committed):

```
Committed prefix: [0-63]   (64 tokens)
Current block:    [64-95]  (32 tokens)

Attention pattern for positions 64-95:
- Each position attends to: [0-95] (full prefix + full block)
```

**Example mask for block size 6, step 2:**
```
         KV tokens →
         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
         [----prefix----] [----block 1----] [----block 2----]
Query ↓
  0-5   [  1  1  1  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0 ]  (Block 0)
  6-11  [  1  1  1  1  1  1  1  1  1  1  1  1  0  0  0  0  0  0 ]  (Block 1)
 12-17  [  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1 ]  (Block 2)
```

Each block has:
- **Full visibility to all prior committed tokens** (prefix grows each step)
- **Bidirectional attention within the current block** (non-causal)
- **No visibility to future blocks** (maintains causality across blocks)

Reference: `DESIGN_MVP.md` §9.2 describes this as "growing-prefix block-style mask."

---

## 3. Virtual Chunk Decomposition Strategy

### 3.1 MVP Approach: Two-Chunk Decomposition

Following vLLM's `chunked_local_attention` pattern, decompose each block's attention into two virtual chunks:

**Chunk 1: Prefix Attention (Non-Causal)**
- Query: Current block tokens [B_start, B_end]
- Key/Value: All committed prefix tokens [0, B_start)
- Pattern: Full non-causal attention (all queries attend to all prefix keys)

**Chunk 2: Block Self-Attention (Non-Causal)**
- Query: Current block tokens [B_start, B_end]
- Key/Value: Current block tokens [B_start, B_end]
- Pattern: Full non-causal attention (bidirectional within block)

**Output Combination:**
```
output = prefix_attn_output + block_self_attn_output
```

Both chunks use **non-causal attention** (`is_causal=False` or `causal=False`), avoiding the need for custom CUDA kernels.

### 3.2 Why This Works

Standard causal attention computes:
```
O = softmax(Q @ K^T / sqrt(d)) @ V    (with causal mask)
```

Our decomposition computes:
```
O_prefix = softmax(Q_block @ K_prefix^T / sqrt(d)) @ V_prefix
O_block  = softmax(Q_block @ K_block^T / sqrt(d)) @ V_block
O        = O_prefix + O_block
```

This is **mathematically equivalent** to single-pass attention with a block-style mask, but leverages existing non-causal attention kernels.

---

## 4. Attention Backend Support

### 4.1 Supported Backends

**FlashAttention** (`flash-attn`):
- Supports `is_causal=False` parameter
- Efficient kernel for non-causal attention patterns
- Production-ready, widely used in vLLM

**FlashInfer** (`flashinfer`):
- Supports `causal=False` parameter
- Alternative high-performance backend
- Also production-ready in vLLM

Both backends natively support the non-causal attention required for LLaDA2.0 block-style masks without custom CUDA.

### 4.2 Backend Selection

vLLM's `Attention` layer auto-selects backend based on:
- Environment variable: `VLLM_ATTENTION_BACKEND` (FLASH_ATTN or FLASHINFER)
- Hardware capabilities
- Model configuration

Our implementation works transparently with both backends.

---

## 5. Implementation Approach

### 5.1 Strategy 1: Attention Metadata Modification (Deferred to Post-MVP)

**Concept:** Modify `AttentionMetadata` to represent block-style mask visibility using vLLM's existing slot mapping mechanism.

**Implementation:**
```python
class LLaDA2BlockAttention(nn.Module):
    def __init__(self, ...):
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_size,
            num_kv_heads=num_kv_heads,
            ...
        )
    
    def forward(self, query, key, value, kv_cache, attn_metadata):
        # Modify attn_metadata to include:
        # - slot_mapping: full prefix + current block
        # - is_causal: False (non-causal within block)
        
        modified_metadata = self._prepare_block_mask_metadata(attn_metadata)
        return self.attn(query, key, value, kv_cache, modified_metadata)
```

**Advantages:**
- Single attention call (efficient)
- Leverages vLLM's existing infrastructure
- Minimal code changes

**Challenges:**
- Requires understanding vLLM's `AttentionMetadata` internals
- May need adjustments for slot mapping semantics

### 5.2 Strategy 2: Dual-Chunk Attention (Fallback)

**Concept:** Explicitly split into two attention calls and combine outputs.

**Implementation:**
```python
class LLaDA2BlockAttention(nn.Module):
    def forward(self, query, key, value, kv_cache, attn_metadata):
        batch_size, seq_len, hidden_dim = query.shape
        
        # Chunk 1: Prefix attention
        prefix_metadata = self._create_prefix_chunk_metadata(attn_metadata)
        prefix_output = self.attn(
            query, 
            key[:, :prefix_len, :], 
            value[:, :prefix_len, :],
            kv_cache, 
            prefix_metadata
        )
        
        # Chunk 2: Block self-attention
        block_metadata = self._create_block_chunk_metadata(attn_metadata)
        block_output = self.attn(
            query,
            key[:, prefix_len:, :],
            value[:, prefix_len:, :],
            kv_cache,
            block_metadata
        )
        
        # Combine
        return prefix_output + block_output
```

**Advantages:**
- More explicit control over attention chunks
- Easier to debug and validate
- Clear separation of prefix vs block attention

**Challenges:**
- Two attention calls per forward (2x overhead)
- Need to carefully manage KV cache slicing

**Phase 7 Decision:** Use Strategy 2 (dual-chunk) for MVP. Strategy 1 was deferred because:

1. **vLLM v1 CommonAttentionMetadata is read-only** - No clean APIs to modify slot_mapping or attention masks dynamically
2. **Dual-chunk approach proved simpler to validate** - Two separate attention calls with clear semantics
3. **Performance overhead is acceptable** - 2x attention calls, but each call is smaller (prefix vs block)
4. **Heterogeneous prefix support** - Dual-chunk naturally handles per-request prefix lengths

Strategy 1 may be revisited in Phase 8.4+ if profiling shows dual-chunk overhead is significant (e.g., >10% of forward pass time). For now, correctness and validation speed take priority over micro-optimization.

---

## 6. Backend Compatibility

### 6.1 Validation Requirements

Both FlashAttention and FlashInfer must produce **identical outputs** for the same block-style attention pattern.

**Test cases:**
1. Empty prefix (first block): Only block self-attention
2. Growing prefix: Prefix length = 0, 32, 64, 96, ...
3. Edge case: Single-token blocks
4. Large context: Prefix length = 1024+

**Acceptance:** Outputs must match within floating-point tolerance (< 1e-5 difference).

### 6.2 Environment Variable Testing

```bash
# Test with FlashAttention
VLLM_ATTENTION_BACKEND=FLASH_ATTN pytest tests/test_llada2_attention.py

# Test with FlashInfer
VLLM_ATTENTION_BACKEND=FLASHINFER pytest tests/test_llada2_attention.py
```

Both must pass all tests.

---

## 7. Performance Considerations

### 7.1 MVP Baseline

**Strategy 1 (metadata modification):**
- **Compute:** ~1x standard attention (ideal)
- **Memory:** Standard KV cache (no duplication)
- **Overhead:** Minimal metadata preparation

**Strategy 2 (dual-chunk):**
- **Compute:** ~2x standard attention (prefix + block chunks)
- **Memory:** Standard KV cache (sliced, not duplicated)
- **Overhead:** Additional KV cache slicing and output combination

**Expected:** Strategy 1 is preferred for performance if vLLM's metadata system supports it.

### 7.2 Comparison to Standard Attention

Block-style attention has:
- **More attention ops per token**: Each block token attends to growing prefix (not just prior tokens)
- **Non-causal compute**: Can't use causal masking optimizations
- **Larger attention matrices**: Full prefix visibility increases computation

**Rough estimate:** 1.5-2x compute vs. standard causal attention, depending on prefix length.

### 7.3 vLLM Integration Efficiency

Leveraging vLLM's existing infrastructure:
- ✅ **KV cache reuse**: Standard PagedAttention KV cache
- ✅ **Fused kernels**: FlashAttention/FlashInfer optimized kernels
- ✅ **Batching**: Works with vLLM's continuous batching
- ❌ **Causal optimizations**: Can't use, but not critical for MVP

---

## 8. Future Optimizations

### 8.1 Post-MVP Enhancements

**Custom CUDA kernel:**
- Fused prefix + block attention in single kernel
- Reduce memory bandwidth and compute overhead
- Target: ~1.2x standard attention performance

**Prefix KV cache sharing:**
- Share prefix KV cache across all blocks
- Reduce memory footprint for long contexts
- Complexity: Coordinate with PagedAttention

**Sparse block attention:**
- For very long prefixes, use sparse/windowed patterns
- Trade-off: Slight quality degradation for speed
- Deferred to post-MVP based on user needs

### 8.2 Out of MVP Scope

Explicitly deferred to future phases:
- ❌ Bespoke CUDA kernels
- ❌ Prefix caching under block-style masks
- ❌ Advanced sparse attention patterns
- ❌ Kernel fusion with MoE layers

---

## 9. Implementation Checklist

- [ ] `LLaDA2BlockAttention` class in `dllm_plugin/models/llada2_attention.py`
- [ ] Strategy 1 implementation (metadata modification)
- [ ] Strategy 2 implementation (dual-chunk fallback)
- [ ] Unit tests validating mask geometry
- [ ] Backend compatibility tests (FlashAttention vs FlashInfer)
- [ ] Integration tests with full model forward pass
- [ ] Performance benchmarks (compare to standard causal attention)
- [ ] Documentation in docstrings

---

## 10. References

- **Design:** `DESIGN_MVP.md` §9 (Attention and Execution MVP)
- **vLLM Attention:** `vllm.model_executor.layers.attention.Attention`
- **FlashAttention:** [Dao et al., 2022](https://arxiv.org/abs/2205.14135)
- **FlashInfer:** vLLM backend documentation
- **LLaDA2.0 Paper:** Block-wise iterative refinement attention patterns
- **SGlang Reference:** [llada2.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llada2.py)

---

## Appendix: Mask Visualization

### Example: Block Size 4, Step 3

```
Prefix (committed): [0-11]  (3 blocks * 4 tokens)
Current block:      [12-15] (block 3)

Attention Matrix (1=attend, 0=masked):
         KV → 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Query ↓     [--prefix--] [--blk1--] [--blk2--] [--blk3--]
     12     1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
     13     1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
     14     1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
     15     1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
```

All positions in block 3 have full visibility to prefix + block.

Compare to **standard causal**:
```
         KV → 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Query ↓
     12     1  1  1  1  1  1  1  1  1  1  1  1  1  0  0  0
     13     1  1  1  1  1  1  1  1  1  1  1  1  1  1  0  0
     14     1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  0
     15     1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
```

LLaDA2.0's block-style mask allows positions 12-14 to "look ahead" within their block.
