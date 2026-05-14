"""Custom attention backend for LLaDA2.0 bidirectional attention.

Wraps the underlying attention backend (FlashInfer or FlashAttention) to:
1. Set causal=False for bidirectional attention within blocks
2. Apply prefix+block page concatenation for multi-block generation (§9.3)

The concatenation runs inside the builder's build() method, operating on
CommonAttentionMetadata BEFORE the backend-specific transformation. This
makes it work with both FlashInfer and FlashAttention backends.
"""

from __future__ import annotations

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import subclass_attention_backend


def create_llada2_bidirectional_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
    kv_cache_block_size: int = 16,
) -> type[AttentionBackend]:
    """Create a bidirectional attention backend for LLaDA2.0.

    Args:
        underlying_attn_backend: The base attention backend to wrap
        kv_cache_block_size: KV cache block size for page table construction

    Returns:
        A new attention backend class with bidirectional (non-causal) attention
    """
    prefix = "LLaDA2Bidirectional_"

    underlying_builder = underlying_attn_backend.get_builder_cls()
    assert issubclass(underlying_builder, AttentionMetadataBuilder)

    class LLaDA2BidirectionalAttentionBuilder(underlying_builder):
        """Builder that forces causal=False and applies prefix+block concatenation.

        Unlike upstream ChunkedLocalAttentionBuilder, we do NOT override
        ``update_block_table()`` because our concatenation modifies
        ``CommonAttentionMetadata.block_table_tensor`` directly in ``build()``,
        which is consumed before any ``update_block_table`` call.
        """

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ):
            from dataclasses import replace

            from dllm_plugin.forward_context import get_num_prefix_tokens_list

            num_prefix_tokens_list = get_num_prefix_tokens_list()

            if num_prefix_tokens_list and any(n > 0 for n in num_prefix_tokens_list):
                from dllm_plugin.attention.concatenated_virtual_batch import (
                    create_concatenated_virtual_batch,
                )

                block_size = (
                    common_attn_metadata.num_actual_tokens
                    // common_attn_metadata.num_reqs
                )

                common_attn_metadata = create_concatenated_virtual_batch(
                    attn_metadata=common_attn_metadata,
                    num_prefix_tokens_per_request=num_prefix_tokens_list,
                    block_size=block_size,
                    kv_cache_block_size=kv_cache_block_size,
                )
            else:
                common_attn_metadata = replace(common_attn_metadata, causal=False)

            return super().build(common_prefix_len, common_attn_metadata, fast_build)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=LLaDA2BidirectionalAttentionBuilder,
    )

    # Disable CUDAGraphs: the concatenated virtual batch creates
    # different-shaped metadata per step (varying num_prefix_tokens),
    # so a fixed captured graph would be incorrect.
    # Matches upstream ChunkedLocalAttention pattern.
    try:
        from vllm.v1.attention.backend import AttentionCGSupport

        attn_backend._cudagraph_support = AttentionCGSupport.NEVER
    except ImportError:
        pass

    return attn_backend
