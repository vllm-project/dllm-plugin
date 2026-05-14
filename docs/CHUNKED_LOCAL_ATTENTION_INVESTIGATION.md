# ChunkedLocalAttention Investigation

**Date:** 2026-05-10  
**Context:** Phase 9.1 numerical validation - investigating how to implement block-bidirectional attention in vLLM

## Summary

Investigated vLLM's `ChunkedLocalAttention` pattern (used by Llama4) to understand local attention implementation. Found that it uses **local + causal** attention, but LLaDA2.0 needs **local + bidirectional**.

**Solution:** Modify `attn_metadata.causal = False` in LLaDA2BlockAttention to enable bidirectional attention.

---

## ChunkedLocalAttention Pattern (Llama4)

### Usage in Llama4

**File:** `/tmp/vllm-repo/vllm/model_executor/models/llama4.py`

```python
# Line 255-256: Decision logic
use_chunked_local_attn = not self.nope and config.attention_chunk_size
attn_cls = ChunkedLocalAttention if use_chunked_local_attn else Attention

# Line 257-270: Instantiation
self.attn = attn_cls(
    self.num_heads,
    self.head_size,
    self.scaling,
    num_kv_heads=self.num_kv_heads,
    cache_config=cache_config,
    quant_config=quant_config,
    prefix=f"{prefix}.attn",
    **(
        {"attention_chunk_size": config.attention_chunk_size}
        if use_chunked_local_attn
        else {}
    ),
)

# Line 309: Forward call (identical to standard Attention)
attn_output = self.attn(q, k, v)
```

**Key insight:** ChunkedLocalAttention is a drop-in replacement for Attention, controlled by `config.attention_chunk_size`.

---

## ChunkedLocalAttention Implementation

### Architecture

**File:** `/tmp/vllm-repo/vllm/model_executor/layers/attention/chunked_local_attention.py`

ChunkedLocalAttention is NOT a new attention algorithm—it's a **wrapper** that:

1. Decomposes sequences into local attention blocks using virtual batches
2. Calls the underlying attention backend (e.g., FlashAttention) with modified metadata
3. Returns `ChunkedLocalAttentionSpec` for KV cache (not standard decoder spec)

### Key Components

**Backend creation (lines 31-78):**
```python
def create_chunked_local_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
    attention_chunk_size: int,
) -> type[AttentionBackend]:
    class ChunkedLocalAttentionBuilder(underlying_builder):
        def build(self, ...):
            # Decompose into virtual batches
            cm, make_virtual_batches_block_table = make_local_attention_virtual_batches(
                attention_chunk_size,
                common_attn_metadata,
                self.kv_cache_spec.block_size,
            )
            metadata = super().build(common_prefix_len, cm, fast_build)
            return metadata
```

**Virtual batch decomposition:** `/tmp/vllm-repo/vllm/v1/attention/backends/utils.py:167-359`

Example: `q_seqlens=[4,10,5]` with `attn_chunk_size=4` becomes:
```
Original: [4 tokens, 10 tokens, 5 tokens]
Virtual:  [2+2 tokens, 4+4+1+1 tokens, 4+1 tokens]
          = [2, 2, 1, 4, 4, 1, 4, 1]
```

Each virtual batch item represents one local attention block.

---

## Critical Finding: Still Uses Causal Attention

**File:** `/tmp/vllm-repo/vllm/v1/attention/backends/utils.py`

```python
# Line 358
return CommonAttentionMetadata(
    # ... other fields ...
    causal=True,  # ← STILL CAUSAL
)
```

**ChunkedLocalAttention is:**
- ✓ Local (limited to chunk_size tokens)
- ✓ Causal (uses causal mask within each chunk)
- ✗ NOT bidirectional

---

## LLaDA2.0 Requirements vs ChunkedLocalAttention

| Aspect | ChunkedLocalAttention | LLaDA2.0 Needs |
|--------|----------------------|----------------|
| Locality | ✓ Local (chunk_size) | ✓ Local (32-token blocks) |
| Causality | ✓ Causal within chunks | ✗ **Bidirectional** within blocks |
| KV Cache | ✓ Supports caching | ✓ Supports caching |
| Cross-block | N/A (local only) | ✓ Causal across blocks |

**Conclusion:** ChunkedLocalAttention is close but not suitable because it's still causal.

---

## How Causal Flag is Set

### V2 Attention Backend

**File:** `/tmp/vllm-repo/vllm/v1/attention/backends/flash_attn.py`

```python
# FlashAttention call (lines ~510-550)
torch.ops.vllm.flash_attn_varlen_func(
    q=query,
    k=key,
    v=value,
    # ... other params ...
    causal=attn_metadata.causal,  # ← Comes from metadata
    # ...
)
```

**Key insight:** The `causal` flag comes from `attn_metadata.causal`.

### Encoder Attention (Bidirectional Example)

**File:** `/tmp/vllm-repo/vllm/v1/attention/backends/flash_attn.py:1037`

```python
# Encoder attention forward
torch.ops.vllm.flash_attn_varlen_func(
    # ... params ...
    causal=False,  # Encoder attention is bidirectional
    # ...
)
```

**Limitation:** Encoder attention (attn_type="encoder_only") doesn't use KV cache.  
LLaDA2.0 needs **decoder** (for KV cache) + **bidirectional** (causal=False).

---

## Solution: Modify attn_metadata.causal

### Implementation in LLaDA2BlockAttention

**File:** `/Users/akellner/MyDir/Code/Open/dllm-plugin/dllm_plugin/models/llada2_attention.py`

```python
# Access V2 Model Runner context
from vllm.forward_context import get_forward_context

forward_context = get_forward_context()
attn_metadata = forward_context.attn_metadata

# Temporarily disable causal masking for bidirectional attention
original_causal = getattr(attn_metadata, 'causal', True)
if hasattr(attn_metadata, 'causal'):
    attn_metadata.causal = False

try:
    # Apply attention with bidirectional mask
    attn_output = self.attn(query=q, key=k, value=v)
finally:
    # Restore original causal setting to not affect other layers
    if hasattr(attn_metadata, 'causal'):
        attn_metadata.causal = original_causal
```

**Why this works:**
1. V2 Model Runner sets `attn_metadata` in forward context
2. FlashAttention reads `attn_metadata.causal` to decide mask type
3. We temporarily set `causal=False` before attention, restore after
4. Other layers unaffected (causal restored after each attention call)

**Limitations:**
- Currently applies bidirectional attention to ENTIRE sequence (not just within blocks)
- Phase 7 will implement proper dual-chunk block boundaries:
  - Prefix chunk: Q=current_block, KV=previous_blocks (causal)
  - Block chunk: Q=current_block, KV=current_block (bidirectional)

---

## V2 Attention Signature

**File:** `/tmp/vllm-repo/vllm/model_executor/layers/attention/attention.py:457-475`

```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
) -> torch.Tensor:
    """
    The KV cache is stored inside this class and is accessed via
    `self.kv_cache`.

    Attention metadata (`attn_metadata`) is set using a context manager in
    the model runner's `execute_model` method. It is accessed via forward
    context using `vllm.forward_context.get_forward_context().attn_metadata`.
    """
```

**Key differences from V0:**
- ✗ NO `positions` parameter
- ✗ NO `kv_cache` parameter (accessed via `self.kv_cache`)
- ✗ NO `attn_metadata` parameter (accessed via `get_forward_context().attn_metadata`)
- ✓ Only `query`, `key`, `value` (+ optional `output_shape`)

---

## Next Steps

### Immediate (Phase 9.1)

1. **Run numerical validation in GPU pod:**
   ```bash
   kubectl exec llada2-numerical-validation -- bash -c '
     cd /workspace/dllm-plugin
     python3 /tmp/test_bidirectional_generation.py
   '
   ```

2. **Verify coherent text generation:**
   - Expected: "Hello, World!..." (programming tutorial from dInfer)
   - Previous (causal): "!\n\n\n..." (20 newlines)
   - New (bidirectional): Should match dInfer output

3. **Run prefill validation:**
   ```bash
   kubectl exec llada2-numerical-validation -- bash -c '
     cd /workspace
     python3 validate_prefill_only.py
   '
   ```

4. **Compare layer-by-layer outputs:**
   - Previous: layer0_attention divergence 0.72 max diff
   - Expected: layer0_attention < 1e-3 max diff (within BF16 tolerance)

### Phase 7: Proper Block Attention

Implement dual-chunk virtual batch decomposition:

1. **Calculate block boundaries from positions:**
   ```python
   # positions: [0, 1, 2, ..., 95] for 96-token prefill
   # block_size = 32
   # Block 0: positions 0-31
   # Block 1: positions 32-63
   # Block 2: positions 64-95
   ```

2. **Create virtual batches for prefix + block chunks:**
   - Use `make_local_attention_virtual_batches` pattern from ChunkedLocalAttention
   - Modify to support heterogeneous prefix lengths (Phase 7.1)

3. **Two attention calls per forward:**
   - Prefix chunk: `attn_metadata.causal = True` (attend to previous blocks)
   - Block chunk: `attn_metadata.causal = False` (attend within current block)
   - Combine: `output = prefix_output + block_output`

4. **Handle decode (single token) separately:**
   - No virtual batch decomposition needed
   - Single token attends to: prefix (causal) + current block (bidirectional)

---

## References

- **ChunkedLocalAttention:** `/tmp/vllm-repo/vllm/model_executor/layers/attention/chunked_local_attention.py`
- **Virtual batches:** `/tmp/vllm-repo/vllm/v1/attention/backends/utils.py:167-359`
- **Llama4 usage:** `/tmp/vllm-repo/vllm/model_executor/models/llama4.py:255-270`
- **FlashAttention backend:** `/tmp/vllm-repo/vllm/v1/attention/backends/flash_attn.py`
- **LLaDA2BlockAttention:** `/Users/akellner/MyDir/Code/Open/dllm-plugin/dllm_plugin/models/llada2_attention.py`

---

## Key Takeaways

1. **ChunkedLocalAttention** is local + causal (not suitable for LLaDA2.0)
2. **Causal flag** comes from `attn_metadata.causal` (can be modified)
3. **V2 signature** uses context for KV cache and metadata (not parameters)
4. **Simple fix** for Phase 9.1: set `attn_metadata.causal = False`
5. **Full fix** for Phase 7: dual-chunk virtual batch decomposition with block boundaries
6. **Validation needed** in GPU pod to verify numerical correctness

---

**Status:** Implementation complete, awaiting GPU pod validation.
