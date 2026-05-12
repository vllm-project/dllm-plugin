# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-style attention for LLaDA2.0 with non-causal mask support.

This module implements LLaDA2.0's unique attention pattern where each position
in the current generation block attends to:
1. All committed prefix tokens (non-causal, full visibility)
2. All tokens in the current block (non-causal, bidirectional)

See docs/ATTENTION_DESIGN.md for detailed design rationale.

**Supported backends:** FlashAttention (`flash-attn`) and FlashInfer (`flashinfer`),
both with `is_causal=False` or `causal=False` for non-causal attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm import ir
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)

# vLLM imports (centralized in vllm_compat for version handling)
from dllm_plugin.vllm_compat import Attention, CommonAttentionMetadata

# Use CommonAttentionMetadata for both type checking and runtime
# (vLLM 0.20+ uses v1.attention.backend.CommonAttentionMetadata)
AttentionMetadata = CommonAttentionMetadata


@CustomOp.register("llada2_rms_norm")
class LLaDA2RMSNorm(CustomOp):
    """RMSNorm with dInfer-compatible behavior for LLaDA2.

    This custom implementation forces the Python reference implementation
    instead of vLLM's C++ kernel to match dInfer's computation behavior.

    **Issue:** vLLM 0.20.1's RMSNorm C++ kernel internally uses FP32 for
    weight multiplication even when weights are BF16, causing a precision
    mismatch with dInfer (which uses vLLM 0.10.2 that multiplies in input dtype).

    **Solution:** Implement forward_cuda to call forward_native, bypassing the
    buggy CUDA kernel. This matches the pattern used by GemmaRMSNorm.

    **Performance:** Slightly slower than C++ kernel but ensures correctness.
    Can be removed once vLLM's kernel is fixed to respect weight dtype.

    See RMSNORM_BUG_FINAL_ANALYSIS.md for detailed root cause analysis.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pure PyTorch implementation that multiplies weight in BF16 (dInfer behavior)."""
        orig_dtype = x.dtype

        # Upcast to FP32 for variance calculation
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual
            residual = x.to(orig_dtype)

        # Compute variance and normalize in FP32
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        # Downcast back to original dtype (BF16)
        x = x.to(orig_dtype)

        # *** CRITICAL: Multiply weight in BF16, not FP32 ***
        # This matches dInfer (vLLM 0.10.2) behavior
        x = x * self.weight

        if residual is None:
            return x
        else:
            return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Force native implementation to avoid buggy CUDA kernel."""
        return self.forward_native(x, residual)


class LLaDA2BlockAttention(nn.Module):
    """Block-style attention for LLaDA2.0 using virtual chunk decomposition.

    Implements non-causal attention within blocks while maintaining causality
    across blocks. Each block token attends to:
    - Full committed prefix (all prior blocks)
    - Full current block (bidirectional within block)

    **Strategy:** Virtual two-chunk decomposition:
    1. Prefix chunk: Q=current_block, KV=committed_prefix (non-causal)
    2. Block chunk: Q=current_block, KV=current_block (non-causal)

    Both chunks use existing FlashAttention/FlashInfer kernels with
    `is_causal=False` or `causal=False`, avoiding custom CUDA.

    **Backends:** Works transparently with FlashAttention and FlashInfer.
    Set `VLLM_ATTENTION_BACKEND` env var to choose backend.

    Args:
        num_heads: Number of attention heads.
        head_size: Dimension of each attention head.
        scale: Attention scaling factor (default: 1/sqrt(head_size)).
        num_kv_heads: Number of key/value heads (for GQA/MQA).
        alibi_slopes: Optional ALiBi slopes (not used in LLaDA2.0).
        sliding_window: Optional sliding window size (not used in LLaDA2.0).
        kv_cache_dtype: Data type for KV cache.
        blocksparse_params: Optional block-sparse parameters (not used).
        logits_soft_cap: Optional logits soft capping (not used).
        prefix: Parameter name prefix for weight loading.

    Example:
        >>> attn = LLaDA2BlockAttention(
        ...     num_heads=32,
        ...     head_size=128,
        ...     num_kv_heads=32,
        ... )
        >>> # In model forward:
        >>> output = attn(hidden_states, positions, kv_cache, attn_metadata)
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        alibi_slopes: torch.Tensor | None = None,
        cache_config=None,
        quant_config=None,
        logits_soft_cap: float | None = None,
        prefix: str = "",
        attn_type: str = "decoder",
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        max_position_embeddings: int = 8192,
        partial_rotary_factor: float | None = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.hidden_size = num_heads * head_size

        # Store cache_config to query KV cache block size
        self.cache_config = cache_config

        # QKV projection (fused)
        # HF checkpoint: attention.query_key_value.weight
        # Fuses Q, K, V projections into single tensor for efficiency
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_size,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Q and K normalization (LLaDA2.0 specific)
        # Applied after QKV projection, before attention
        # Uses custom LLaDA2RMSNorm to match dInfer behavior (see RMSNORM_BUG_FINAL_ANALYSIS.md)
        self.q_norm = LLaDA2RMSNorm(self.head_size, eps=1e-6)
        self.k_norm = LLaDA2RMSNorm(self.head_size, eps=1e-6)

        # Output projection
        # HF checkpoint: attention.dense.weight
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Initialize RoPE (Rotary Position Embedding)
        # LLaDA2.0 uses RoPE with custom theta=600000 and partial_rotary_factor=0.5
        from vllm.model_executor.layers.rotary_embedding import get_rope

        # Build rope_parameters dict from config
        rope_parameters = {}
        if rope_theta != 10000:  # Non-default theta
            rope_parameters["rope_theta"] = rope_theta
        if partial_rotary_factor is not None:
            # LLaDA2 uses partial RoPE (typically 0.5 = rotate 50% of head dims)
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        if rope_scaling is not None:
            # Merge rope_scaling dict into rope_parameters
            rope_parameters.update(rope_scaling)

        self.rotary_emb = get_rope(
            head_size,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters if rope_parameters else None,
            is_neox_style=True,  # LLaDA2 uses GPT-NeoX style RoPE
        )

        # Use custom bidirectional attention backend for LLaDA2.0
        # Following ChunkedLocalAttention pattern: get underlying backend first, then wrap
        import torch
        from vllm.v1.attention.selector import get_attn_backend

        from dllm_plugin.models.llada2_attention_backend import (
            create_llada2_bidirectional_attention_backend,
        )

        dtype = torch.get_default_dtype()
        kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"

        print("[LLaDA2BlockAttention] Initializing with bidirectional attention")
        print(f"[LLaDA2BlockAttention] dtype={dtype}, kv_cache_dtype={kv_cache_dtype}, head_size={head_size}")
        print(f"[LLaDA2BlockAttention] RoPE: theta={rope_theta}, scaling={rope_scaling}")

        # Get the underlying attention backend (FlashAttention or FlashInfer)
        underlying_attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype)
        print(f"[LLaDA2BlockAttention] Underlying backend: {underlying_attn_backend}")

        # Wrap it with bidirectional attention (causal=False)
        attn_backend = create_llada2_bidirectional_attention_backend(underlying_attn_backend)
        print(f"[LLaDA2BlockAttention] Custom bidirectional backend: {attn_backend}")

        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale if scale is not None else (1.0 / (head_size**0.5)),
            num_kv_heads=num_kv_heads if num_kv_heads is not None else num_heads,
            alibi_slopes=None,  # Not used in LLaDA2
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=logits_soft_cap,
            per_layer_sliding_window=None,  # Not used in LLaDA2
            prefix=prefix,
            attn_type=attn_type,
            attn_backend=attn_backend,
        )
        print(f"[LLaDA2BlockAttention] Attention layer created with backend: {self.attn.attn_backend}")

        # Attention strategy: Dual-chunk decomposition (Strategy 2)
        # Decomposes each forward pass into prefix chunk (causal over
        # committed prefix) and block chunk (non-causal within current block).
        # Delegates to vLLM's PagedAttention with per-chunk metadata.
        # Strategy 1 (metadata modification) deferred - see ATTENTION_DESIGN.md

    def _get_kv_cache_block_size(self) -> int:
        """Query KV cache block size from cache_config.

        Returns:
            KV cache block size in tokens.

        Raises:
            ValueError: If cache_config is None (required for block attention).
        """
        if self.cache_config is None:
            raise ValueError(
                "cache_config required for LLaDA2 block attention. "
                "KV cache block size must match actual vLLM configuration "
                "for correct virtual batch slicing."
            )

        # Try common attribute names for block size
        for attr in ("block_size", "num_tokens_per_block", "token_block_size"):
            if hasattr(self.cache_config, attr):
                block_size = getattr(self.cache_config, attr)
                if block_size is not None and block_size > 0:
                    return int(block_size)

        # Default to 16 (standard vLLM block size in v1 architecture)
        return 16

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply block-style attention with QKV projection (V2 Model Runner signature).

        Args:
            positions: Position indices, shape (num_tokens,).
            hidden_states: Input tensor, shape (num_tokens, hidden_size).

        Returns:
            Attention output, shape (num_tokens, hidden_size).

        Note:
            V2 Model Runner pattern:
            - KV cache accessed by self.attn layer via self.kv_cache
            - Attention metadata accessed via get_forward_context().attn_metadata
            - No explicit kv_cache or attn_metadata parameters needed
        """
        # Project to Q, K, V
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_size,
                self.num_kv_heads * self.head_size,
                self.num_kv_heads * self.head_size,
            ],
            dim=-1,
        )

        # Apply normalization (LLaDA2.0 specific)
        # Reshape for per-head normalization
        num_tokens = q.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_size)
        k = k.view(num_tokens, self.num_kv_heads, self.head_size)

        # Normalize Q and K per-head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape back to (num_tokens, hidden_size)
        q = q.reshape(num_tokens, self.num_heads * self.head_size)
        k = k.reshape(num_tokens, self.num_kv_heads * self.head_size)

        # Apply RoPE (Rotary Position Embedding)
        # CRITICAL: This was missing and causing numerical divergence!
        # LLaDA2 uses partial RoPE (50% of head dims) with theta=600000

        # ALWAYS log positions during generation (not just warmup)
        # This is critical for debugging chunking issues
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        is_warmup = (ctx is None) or getattr(ctx, 'is_warmup', False)

        if not is_warmup and q.shape[0] > 10:
            # During actual generation with >10 tokens
            print(f"[ROPE POSITIONS] num_tokens={q.shape[0]}, positions={positions.tolist()}")
            print(f"[ROPE POSITIONS] Layer {self.layer_idx if hasattr(self, 'layer_idx') else '?'}")

        q_before_rope = q.clone() if q.shape[0] < 20 else None  # Debug: save for comparison
        k_before_rope = k.clone() if k.shape[0] < 20 else None  # Debug: save for comparison
        q, k = self.rotary_emb(positions, q, k)

        # Debug: verify RoPE actually changed the values
        if q_before_rope is not None:
            q_diff = torch.abs(q - q_before_rope).max().item()
            k_diff = torch.abs(k - k_before_rope).max().item() if k_before_rope is not None else 0.0
            print(f"[RoPE DEBUG] num_tokens={q.shape[0]}, positions={positions[:min(5, len(positions))]}")
            print(f"[RoPE DEBUG] Q max change: {q_diff:.6e}, K max change: {k_diff:.6e}")
            if q_diff < 1e-6 or k_diff < 1e-6:
                print("[RoPE WARNING] ❌ RoPE did NOT modify values! This is a BUG!")
                print(f"[RoPE WARNING] positions dtype={positions.dtype}, device={positions.device}")
                print(f"[RoPE WARNING] rotary_emb type={type(self.rotary_emb).__name__}")
            else:
                print("[RoPE DEBUG] ✅ RoPE working correctly")

        # Check if we should use chunked block attention (Strategy 2)
        from dllm_plugin.forward_context import get_num_prefix_tokens_list
        from vllm.forward_context import get_forward_context

        num_prefix_tokens_list = get_num_prefix_tokens_list()
        print(f"[ATTN DEBUG] num_prefix_tokens_list: {num_prefix_tokens_list}")

        if num_prefix_tokens_list is not None:
            # Use dual-chunk block-style attention (Strategy 2 from ATTENTION_DESIGN.md)
            # Access KV cache and attention metadata from V2 forward context
            context = get_forward_context()

            # In V2, attn_metadata is a dict mapping layer names to metadata objects
            # Extract metadata for this specific layer
            attn_metadata_dict = context.attn_metadata if context else None
            if isinstance(attn_metadata_dict, dict):
                # Get metadata for current layer using the layer's prefix/name
                layer_name = getattr(self, 'layer_name', None) or self.attn.layer_name
                attn_metadata = attn_metadata_dict.get(layer_name)
                print(f"[ATTN DEBUG] Layer: {layer_name}, got metadata: {attn_metadata is not None}")
            else:
                attn_metadata = attn_metadata_dict
                print(f"[ATTN DEBUG] attn_metadata is not dict, type: {type(attn_metadata_dict)}")

            kv_cache = self.attn.kv_cache  # KV cache stored on Attention layer

            # Delegate to _forward_concatenated for block-style attention
            # This uses a SINGLE virtual batch combining prefix + block KV
            # to fix the softmax normalization bug
            attn_output = self._forward_concatenated(
                query=q,
                key=k,
                value=v,
                attn_metadata=attn_metadata,
                num_prefix_tokens_list=num_prefix_tokens_list,
            )
        else:
            # Fallback to simple bidirectional attention (for testing or edge cases)
            # The LLaDA2BidirectionalFlashAttentionBackend overrides causal=False
            # in the metadata builder, so we just call attention normally
            attn_output = self.attn(
                query=q,
                key=k,
                value=v,
            )

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_concatenated(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        num_prefix_tokens_list: list[int],
    ) -> torch.Tensor:
        """Single-batch attention via concatenated virtual batch.

        **FIX for softmax normalization bug:**
        Instead of TWO separate attention calls with TWO softmax normalizations
        (which causes weights to sum to 2.0), we create ONE virtual batch combining
        prefix + block KV, resulting in ONE softmax normalization (weights sum to 1.0).

        **How it works:**
        1. Current block KV is written to cache BEFORE attention (via unified_kv_cache_update)
        2. FlashAttention reads BOTH prefix KV (already cached) + block KV (just cached)
           using the concatenated block_table
        3. Single attention call → single softmax → mathematically correct!

        See: /tmp/ROOT_CAUSE_SOFTMAX_NORMALIZATION.md for full analysis.

        Args:
            query: Query tensor [num_tokens, num_heads * head_size]
            key: Key tensor (will be written to cache before attention)
            value: Value tensor (will be written to cache before attention)
            attn_metadata: Original attention metadata from forward context
            num_prefix_tokens_list: List of committed prefix lengths per request

        Returns:
            Attention output with correct normalization
        """
        # Debug logging
        print(f"[CONCATENATED ATTN] Using concatenated virtual batch (FIXED)")
        print(f"[CONCATENATED ATTN] Prefix lengths: {num_prefix_tokens_list}")
        print(f"[CONCATENATED ATTN] Query shape: {query.shape}")

        from dllm_plugin.attention.concatenated_virtual_batch import (
            create_concatenated_virtual_batch,
        )
        from vllm.forward_context import get_forward_context, override_forward_context
        from dataclasses import replace

        # Get num_tokens from query shape
        num_tokens = query.shape[0]

        # Query KV cache block size from cache_config
        kv_cache_block_size = self._get_kv_cache_block_size()

        # Create unified virtual batch combining prefix + block KV
        concatenated_metadata = create_concatenated_virtual_batch(
            attn_metadata=attn_metadata,
            num_prefix_tokens_per_request=num_prefix_tokens_list,
            block_size=num_tokens,
            kv_cache_block_size=kv_cache_block_size,
        )

        # Override forward context with concatenated metadata
        context = get_forward_context()
        layer_name = getattr(self, 'layer_name', None) or self.attn.layer_name

        # Get the original attn_metadata dict
        original_metadata_dict = context.attn_metadata

        # Create modified metadata dict with concatenated_metadata for this layer
        concatenated_metadata_dict = original_metadata_dict.copy()
        concatenated_metadata_dict[layer_name] = concatenated_metadata

        # Create new context with concatenated metadata
        concatenated_context = replace(context, attn_metadata=concatenated_metadata_dict)

        # Single attention call with combined metadata
        # CRITICAL: key and value are passed, will be written to cache FIRST,
        # then FlashAttention reads from cache using block_table
        with override_forward_context(concatenated_context):
            output = self.attn(
                query=query,
                key=key,     # Will be written to cache first
                value=value,  # Will be written to cache first
            )

        print("[CONCATENATED ATTN] Single softmax normalization ✅")
        return output

    def _forward_dual_chunk_BUGGY(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        num_prefix_tokens_list: list[int],
    ) -> torch.Tensor:
        """[DEPRECATED - BUGGY] Dual-chunk attention via virtual batch decomposition.

        **BUG:** This implementation uses TWO separate softmax normalizations,
        causing attention weights to sum to 2.0 instead of 1.0.

        Mathematical problem:
        - prefix_output uses softmax over prefix keys only → weights sum to 1.0
        - block_output uses softmax over block keys only → weights sum to 1.0
        - Combined: prefix_output + block_output → total weight = 2.0 ❌

        This breaks mathematical equivalence with dInfer reference and causes
        numerical divergence (C8 max_diff=1.188).

        **Fix:** Use _forward_concatenated instead, which creates ONE virtual batch
        combining prefix + block KV, resulting in ONE softmax (weights sum to 1.0).

        See: /tmp/ROOT_CAUSE_SOFTMAX_NORMALIZATION.md for full analysis.

        Args:
            query: Query tensor
            key: Key tensor (for block chunk)
            value: Value tensor (for block chunk)
            attn_metadata: Original attention metadata from forward context
            num_prefix_tokens_list: List of committed prefix lengths per request

        Returns:
            Combined output: prefix_output + block_output (BUGGY!)
        """
        # Debug logging to verify chunked attention is being used
        print(f"[CHUNKED ATTN] Using dual-chunk attention")
        print(f"[CHUNKED ATTN] Prefix lengths: {num_prefix_tokens_list}")
        print(f"[CHUNKED ATTN] Query shape: {query.shape}")

        from dllm_plugin.attention.virtual_batches import (
            make_block_attention_virtual_batches,
        )
        from vllm.forward_context import get_forward_context, override_forward_context

        # Get num_tokens from query shape
        # Query is [num_tokens, num_heads * head_size]
        num_tokens = query.shape[0]

        # Query KV cache block size from cache_config
        kv_cache_block_size = self._get_kv_cache_block_size()

        # Create unified virtual batches with heterogeneous support
        prefix_metadata, block_metadata = make_block_attention_virtual_batches(
            attn_metadata=attn_metadata,
            num_prefix_tokens_per_request=num_prefix_tokens_list,
            block_size=num_tokens,  # Current block size
            kv_cache_block_size=kv_cache_block_size,
        )

        # Edge case: No prefix (first block)
        if prefix_metadata is None:
            # Only block self-attention needed
            # No need to modify context - block_metadata is same as original for first block
            print("[DUAL CHUNK] First block - using original attn call")
            return self.attn(
                query=query,
                key=key,
                value=value,
            )

        # Two-chunk case: prefix + block
        from vllm.forward_context import get_forward_context, override_forward_context

        context = get_forward_context()
        layer_name = getattr(self, 'layer_name', None) or self.attn.layer_name

        # Get the original attn_metadata dict
        original_metadata_dict = context.attn_metadata

        # Chunk 1: Prefix attention (Q=current_block, KV=cached_prefix)
        # Create modified metadata dict with prefix_metadata for this layer
        prefix_metadata_dict = original_metadata_dict.copy()
        prefix_metadata_dict[layer_name] = prefix_metadata

        # Create new context with prefix metadata
        from dataclasses import replace
        prefix_context = replace(context, attn_metadata=prefix_metadata_dict)

        with override_forward_context(prefix_context):
            prefix_output = self.attn(
                query=query,
                key=None,  # Use cached KV for prefix
                value=None,  # Use cached KV for prefix
            )

        # Chunk 2: Block self-attention (Q=current_block, KV=current_block)
        # IMPORTANT: Uses current forward pass K/V (key=key, value=value),
        # NOT cached KV. This is bidirectional attention within the generation block.
        # vLLM writes these K/V to cache (via slot_mapping) for future prefixes.
        block_metadata_dict = original_metadata_dict.copy()
        block_metadata_dict[layer_name] = block_metadata

        block_context = replace(context, attn_metadata=block_metadata_dict)

        with override_forward_context(block_context):
            block_output = self.attn(
                query=query,
                key=key,  # Current forward pass, not cache
                value=value,  # Current forward pass, not cache
            )

        # Combine outputs (additive for overlapping queries, disjoint KV)
        # ❌ BUG: This adds two separately-normalized outputs!
        # prefix_output has weights summing to 1.0 (softmax over prefix only)
        # block_output has weights summing to 1.0 (softmax over block only)
        # Combined weight = 2.0, which is mathematically incorrect!
        print("[DUAL CHUNK BUGGY] Combined prefix + block outputs (weights sum to 2.0)")
        return prefix_output + block_output  # ❌ BUGGY!


# Alias for compatibility with model code expecting standard naming
BlockStyleAttention = LLaDA2BlockAttention
