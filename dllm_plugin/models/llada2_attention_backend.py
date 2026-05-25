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
        """Builder that forces causal=False and applies prefix+block concatenation."""

        @classmethod
        def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
            """UNIFORM_BATCH: only model.forward() is graph-captured.

            ModelState hooks (before_step, prepare_inputs, prepare_attn)
            and DiffusionSampler all run in eager mode before/after graph
            replay.  Python dicts and torch.tensor() calls in those eager
            hooks are acceptable.  Block diffusion decode steps have
            constant batch shape (1 request, DRAFT_SIZE tokens) compatible
            with graph replay.
            """
            from vllm.v1.attention.backend import AttentionCGSupport

            return AttentionCGSupport.UNIFORM_BATCH

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ):
            from dataclasses import replace

            num_prefix_tokens_list = getattr(
                common_attn_metadata, "dllm_prefix_lengths", None
            )

            if num_prefix_tokens_list and any(n > 0 for n in num_prefix_tokens_list):
                from dllm_plugin.attention.concatenated_virtual_batch import (
                    create_concatenated_virtual_batch,
                )
                from dllm_plugin.config import DRAFT_SIZE

                common_attn_metadata = create_concatenated_virtual_batch(
                    attn_metadata=common_attn_metadata,
                    num_prefix_tokens_per_request=num_prefix_tokens_list,
                    block_size=DRAFT_SIZE,
                    kv_cache_block_size=kv_cache_block_size,
                )
                # Store for CUDAGraph replay via update_block_table()
                self._last_prefix_tokens_list = num_prefix_tokens_list
            else:
                common_attn_metadata = replace(common_attn_metadata, causal=False)
                self._last_prefix_tokens_list = None

            return super().build(common_prefix_len, common_attn_metadata, fast_build)

        def update_block_table(self, metadata, block_table, slot_mapping):
            if self._last_prefix_tokens_list and any(
                n > 0 for n in self._last_prefix_tokens_list
            ):
                import torch

                from dllm_plugin.config import DRAFT_SIZE

                num_reqs = len(self._last_prefix_tokens_list)
                prefix_t = torch.tensor(
                    self._last_prefix_tokens_list,
                    dtype=torch.int32,
                    device=block_table.device,
                )
                n_prefix_blocks = (
                    prefix_t + kv_cache_block_size - 1
                ) // kv_cache_block_size
                n_block_blocks = (
                    DRAFT_SIZE + kv_cache_block_size - 1
                ) // kv_cache_block_size
                n_total = n_prefix_blocks + n_block_blocks
                max_total = int(n_total.max().item())

                out = torch.full(
                    (num_reqs, max_total),
                    -1,
                    dtype=block_table.dtype,
                    device=block_table.device,
                )
                col_idx = torch.arange(max_total, device=block_table.device).unsqueeze(
                    0
                )
                valid = col_idx < n_total.unsqueeze(1)
                src_idx = col_idx.clamp(max=block_table.shape[1] - 1)
                out[valid] = block_table.gather(1, src_idx.expand(num_reqs, -1))[valid]
                block_table = out

            return super().update_block_table(metadata, block_table, slot_mapping)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=LLaDA2BidirectionalAttentionBuilder,
    )

    return attn_backend
