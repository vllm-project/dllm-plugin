"""Forward context for passing dLLM-specific state through the model hierarchy.

Uses contextvars for thread-safe, request-safe state propagation without
modifying function signatures.
"""

import contextvars

# Context variable for num_prefix_tokens per request in current batch
_num_prefix_tokens_list_ctx: contextvars.ContextVar[list[int] | None] = (
    contextvars.ContextVar("_dllm_num_prefix_tokens_list", default=None)
)


def set_num_prefix_tokens_list(tokens_list: list[int] | None):
    """Set num_prefix_tokens_list for current forward pass.

    Args:
        tokens_list: List of committed prefix lengths per request,
                     or None if not using chunked attention.

    Returns:
        Token for resetting context later (use in try/finally).
    """
    return _num_prefix_tokens_list_ctx.set(tokens_list)


def get_num_prefix_tokens_list() -> list[int] | None:
    """Get num_prefix_tokens_list for current forward pass.

    Returns:
        List of prefix lengths per request, or None if not set.
    """
    return _num_prefix_tokens_list_ctx.get(None)
