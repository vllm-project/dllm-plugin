"""Custom attention backend for LLaDA2.0 bidirectional attention.

Based on vLLM's ChunkedLocalAttention pattern but for bidirectional attention.
"""

from typing import Type

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import subclass_attention_backend
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)


def create_llada2_bidirectional_attention_backend(
    underlying_attn_backend: Type[AttentionBackend],
) -> Type[AttentionBackend]:
    """Create a bidirectional attention backend for LLaDA2.0.

    Wraps the underlying attention backend (e.g., FlashAttention) and
    overrides the causal flag to False for bidirectional attention.

    Args:
        underlying_attn_backend: The base attention backend to wrap

    Returns:
        A new attention backend class with bidirectional (non-causal) attention
    """
    prefix = "LLaDA2Bidirectional_"

    underlying_builder = underlying_attn_backend.get_builder_cls()
    assert issubclass(underlying_builder, AttentionMetadataBuilder)

    class LLaDA2BidirectionalAttentionBuilder(underlying_builder):  # type: ignore
        """Custom builder that forces causal=False for bidirectional attention."""

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ):
            """Build attention metadata with causal=False for bidirectional attention.

            Overrides the causal flag in CommonAttentionMetadata before calling
            the parent builder.
            """
            # CRITICAL: Override causal flag to False for bidirectional attention
            # This must be done on the CommonAttentionMetadata BEFORE the parent
            # builder creates the backend-specific metadata (e.g., FlashAttentionMetadata)
            original_causal = common_attn_metadata.causal
            common_attn_metadata.causal = False

            # DEBUG: Verify causal flag is being set
            print(f"[LLaDA2 Bidirectional Backend] Setting causal={common_attn_metadata.causal} (was {original_causal})")

            try:
                # Call parent builder with modified causal=False
                metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

                # DEBUG: Verify the metadata has causal=False
                if hasattr(metadata, 'causal'):
                    print(f"[LLaDA2 Bidirectional Backend] Built metadata with causal={metadata.causal}")

                return metadata
            finally:
                # Restore original causal flag (though this metadata object
                # is likely discarded after this call)
                common_attn_metadata.causal = original_causal

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=LLaDA2BidirectionalAttentionBuilder,
    )

    return attn_backend


# Create bidirectional FlashAttention backend for LLaDA2.0
LLaDA2BidirectionalFlashAttentionBackend = create_llada2_bidirectional_attention_backend(
    FlashAttentionBackend
)
