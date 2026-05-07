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

try:
    from vllm.model_executor.layers.attention import Attention
except ImportError:  # pragma: no cover
    try:
        from vllm.model_executor.layers.attention.layer import Attention
    except ImportError:
        from vllm.attention.layer import Attention

try:
    from vllm.attention.backends.abstract import AttentionMetadata
except ImportError:  # pragma: no cover
    try:
        from vllm.attention import AttentionMetadata
    except ImportError:
        # Fallback for type checking
        AttentionMetadata = object

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)


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
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.hidden_size = num_heads * head_size

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
        self.q_norm = RMSNorm(self.head_size, eps=1e-6)
        self.k_norm = RMSNorm(self.head_size, eps=1e-6)

        # Output projection
        # HF checkpoint: attention.dense.weight
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Use vLLM's standard Attention layer as backend
        # It auto-selects FlashAttention or FlashInfer based on environment
        # NOTE: LLaDA2 doesn't use sliding window, alibi, or blocksparse
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
        )

        # Attention strategy: Dual-chunk approach (delegates to vLLM attention backend)
        # Uses vLLM's PagedAttention with block-style metadata from scheduler.
        # The scheduler/worker sets up attention metadata for block visibility;
        # this layer trusts the backend to handle non-causal patterns correctly.
        self._use_dual_chunk = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_scale: float = 1.0,
        num_prefix_tokens: int | None = None,
    ) -> torch.Tensor:
        """Apply block-style attention with QKV projection.

        Args:
            hidden_states: Input tensor, shape (batch_size, seq_len, hidden_size).
            positions: Position indices for RoPE (unused in LLaDA2.0).
            kv_cache: KV cache tensor (PagedAttention format).
            attn_metadata: Attention metadata from vLLM.
            kv_scale: Scaling factor for KV cache (default: 1.0).
            num_prefix_tokens: Number of committed tokens (prefix length)
                for virtual batch attention.

        Returns:
            Attention output, shape (batch_size, seq_len, hidden_size).
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
        batch_size, seq_len = q.shape[0], q.shape[1]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_size)

        # Normalize Q and K per-head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape back
        q = q.reshape(batch_size, seq_len, self.num_heads * self.head_size)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads * self.head_size)

        # Apply attention
        if self._use_dual_chunk:
            attn_output = self._forward_dual_chunk(
                q, k, v, positions, kv_cache, attn_metadata, kv_scale, num_prefix_tokens
            )
        else:
            attn_output = self._forward_metadata_modification(
                q, k, v, positions, kv_cache, attn_metadata, kv_scale
            )

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_dual_chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_scale: float = 1.0,
        num_prefix_tokens: int | None = None,
    ) -> torch.Tensor:
        """Strategy 2: Dual-chunk attention via virtual batch decomposition.

        **Design:** Transform AttentionMetadata to create two virtual batches:
        1. Prefix chunk: Q=current_block, KV=committed_prefix (non-causal)
        2. Block chunk: Q=current_block, KV=current_block (non-causal)

        Follows vLLM's chunked_local_attention pattern:
        https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/chunked_local_attention.py

        Returns:
            Combined output: prefix_output + block_output
        """
        # Fall back to single-pass if num_prefix_tokens not provided
        if num_prefix_tokens is None:
            return self.attn(
                positions=positions,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                kv_scale=kv_scale,
            )

        from dllm_plugin.attention.virtual_batches import (
            make_block_attention_virtual_batches,
        )

        block_size = query.shape[1]

        # Create virtual batches
        prefix_metadata, block_metadata = make_block_attention_virtual_batches(
            attn_metadata=attn_metadata,
            num_prefix_tokens=num_prefix_tokens,
            block_size=block_size,
        )

        # Edge case: No prefix (first block)
        if prefix_metadata is None:
            return self.attn(
                positions=positions,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=block_metadata,
                kv_scale=kv_scale,
            )

        # Chunk 1: Prefix attention
        prefix_output = self.attn(
            positions=positions,
            query=query,
            key=None,  # Use KV cache
            value=None,  # Use KV cache
            kv_cache=kv_cache,
            attn_metadata=prefix_metadata,
            kv_scale=kv_scale,
        )

        # Chunk 2: Block self-attention
        block_output = self.attn(
            positions=positions,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=block_metadata,
            kv_scale=kv_scale,
        )

        # Combine outputs (additive for overlapping queries, disjoint KV)
        return prefix_output + block_output

    def _forward_metadata_modification(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        """Strategy 1: Modify attention metadata for block-style mask.

        **Not implemented in MVP.** This is a placeholder for future optimization.

        The idea is to modify `attn_metadata` to represent block-style visibility
        using vLLM's existing slot_mapping and is_causal=False, avoiding dual-chunk
        overhead.

        Deferred to post-MVP for performance optimization.
        """
        raise NotImplementedError(
            "Strategy 1 (metadata modification) not implemented in Phase 7 MVP. "
            "Using Strategy 2 (dual-chunk) instead. "
            "See ATTENTION_DESIGN.md for details."
        )


# Alias for compatibility with model code expecting standard naming
BlockStyleAttention = LLaDA2BlockAttention
