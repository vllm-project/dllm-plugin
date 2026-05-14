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
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)

from dllm_plugin.vllm_compat import Attention


@CustomOp.register("llada2_rms_norm")
class LLaDA2RMSNorm(CustomOp):
    """RMSNorm matching dInfer's BF16 computation behavior.

    Stock vLLM RMSNorm dispatches to ir.ops.rms_norm which may produce
    subtly different BF16 rounding due to IrOp tracing/compilation.
    This custom implementation forces pure PyTorch computation to
    exactly match dInfer.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual
            residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        x = x * self.weight
        if residual is None:
            return x
        return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        # Uses custom LLaDA2RMSNorm to match dInfer
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
        # Following ChunkedLocalAttention pattern
        import torch
        from vllm.v1.attention.selector import get_attn_backend

        from dllm_plugin.models.llada2_attention_backend import (
            create_llada2_bidirectional_attention_backend,
        )

        dtype = torch.get_default_dtype()
        kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )

        # FlashInfer is required for non-causal paged attention.
        # FA2's paged kernel ignores causal=False on A100 (SM80).
        # TODO(#45): version-gate per GPU arch once FA2 is fixed.
        import logging as _logging
        import os

        _attn_log = _logging.getLogger(__name__)
        user_backend = os.environ.get("VLLM_ATTENTION_BACKEND")
        try:
            from vllm.v1.attention.backends.flashinfer import (
                FlashInferBackend,
            )

            underlying_attn_backend = FlashInferBackend
            if user_backend and user_backend.upper() != "FLASHINFER":
                _attn_log.warning(
                    "LLaDA2 requires FlashInfer for non-causal "
                    "attention. Ignoring VLLM_ATTENTION_BACKEND=%s",
                    user_backend,
                )
        except ImportError:
            _attn_log.warning(
                "FlashInfer unavailable, falling back to "
                "auto-selected backend. Non-causal attention "
                "may not work on A100 (SM80).",
            )
            underlying_attn_backend = get_attn_backend(head_size, dtype, kv_cache_dtype)

        # Wrap it with bidirectional attention (causal=False) + block concatenation
        kv_cache_block_size = self._get_kv_cache_block_size()
        attn_backend = create_llada2_bidirectional_attention_backend(
            underlying_attn_backend,
            kv_cache_block_size=kv_cache_block_size,
        )

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
        q, k = self.rotary_emb(positions, q, k)

        # Bidirectional attention with prefix+block concatenation.
        # The builder handles both causal=False and prefix page concatenation
        # (see llada2_attention_backend.py), so the forward path is always simple.
        attn_output = self.attn(
            query=q,
            key=k,
            value=v,
        )

        # Output projection
        output, _ = self.o_proj(attn_output)
        return output
