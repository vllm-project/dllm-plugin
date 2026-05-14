# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor extraction utilities for numerical validation.

This module provides utilities to extract intermediate activations from
both HuggingFace and vLLM models for numerical comparison.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


class TensorExtractor:
    """Extract intermediate tensors from model forward pass.

    This class uses PyTorch forward hooks to capture intermediate activations
    at specific layers during the forward pass.

    Usage:
        extractor = TensorExtractor()
        extractor.register_hooks(model, layer_names=["layer.0", "layer.1"])
        outputs = model(inputs)
        embeddings = extractor.get_tensor("embeddings")
        layer0_output = extractor.get_tensor("layer.0")
    """

    def __init__(self):
        """Initialize tensor extractor."""
        self._tensors: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []

    def clear(self):
        """Clear all extracted tensors."""
        self._tensors.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_tensor(self, name: str) -> torch.Tensor | None:
        """Get extracted tensor by name.

        Args:
            name: Tensor name (same as used in register_hook)

        Returns:
            Extracted tensor or None if not found
        """
        return self._tensors.get(name)

    def register_hook(self, module: torch.nn.Module, name: str):
        """Register a forward hook on a module.

        Args:
            module: PyTorch module to hook
            name: Name to store the output tensor
        """

        def hook_fn(module, input, output):
            # Store output tensor (detached to avoid gradient tracking)
            if isinstance(output, torch.Tensor):
                self._tensors[name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0:
                # Some modules return tuples (e.g., attention with past_key_values)
                self._tensors[name] = output[0].detach()

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up hooks."""
        self.remove_hooks()


def extract_hf_embeddings(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract embeddings from HuggingFace model.

    Args:
        model: HuggingFace model (e.g., AutoModelForCausalLM)
        input_ids: (batch, seq_len) token IDs

    Returns:
        Embeddings: (batch, seq_len, hidden_size)
    """
    # Access embedding layer
    # For LLaDA2: model.model.embed_tokens
    embed_tokens = model.model.embed_tokens
    with torch.no_grad():
        embeddings = embed_tokens(input_ids)
    return embeddings


def extract_hf_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract final logits from HuggingFace model.

    Args:
        model: HuggingFace model
        input_ids: (batch, seq_len) token IDs

    Returns:
        Logits: (batch, seq_len, vocab_size)
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        return outputs.logits


def extract_hf_intermediate_activations(
    model,
    input_ids: torch.Tensor,
    layer_names: list[str],
) -> dict[str, torch.Tensor]:
    """Extract intermediate activations from HuggingFace model.

    Args:
        model: HuggingFace model
        input_ids: (batch, seq_len) token IDs
        layer_names: List of layer names to extract (e.g., ["model.layers.0"])

    Returns:
        Dictionary mapping layer names to output tensors
    """
    extractor = TensorExtractor()

    # Register hooks for requested layers
    for layer_name in layer_names:
        # Navigate to the module
        module = model
        for part in layer_name.split("."):
            module = getattr(module, part)
        extractor.register_hook(module, layer_name)

    # Run forward pass
    with torch.no_grad():
        _ = model(input_ids=input_ids)

    # Collect tensors
    tensors = {name: extractor.get_tensor(name) for name in layer_names}

    # Cleanup
    extractor.remove_hooks()

    return tensors


def extract_vllm_embeddings(llm, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract embeddings from vLLM model.

    Args:
        llm: vLLM LLM instance
        input_ids: (batch, seq_len) token IDs (on GPU)

    Returns:
        Embeddings: (batch, seq_len, hidden_size)

    Note:
        Accesses vLLM's internal model structure:
        llm.llm_engine.model_executor.driver_worker.model_runner.model
    """
    # Access vLLM's internal model
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Access embedding layer
    # vLLM's dllm-plugin LLaDA2ForCausalLM has embed_tokens at top level
    embed_tokens = model.embed_tokens

    with torch.no_grad():
        embeddings = embed_tokens(input_ids)

    return embeddings


def extract_vllm_logits(
    llm,
    input_ids: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract final logits from vLLM model.

    Args:
        llm: vLLM LLM instance
        input_ids: (batch, seq_len) token IDs (on GPU)
        positions: (batch, seq_len) position IDs (optional, auto-generated if None)

    Returns:
        Logits: (batch, seq_len, vocab_size)

    Note:
        This function requires GPU testing to validate the approach.
        Two possible approaches:

        1. Direct model forward (used below):
           - Access model via llm.llm_engine.model_executor.
             driver_worker.model_runner.model
           - Construct simplified AttentionMetadata for single-sequence prefill
           - May fail if vLLM's internal API has changed

        2. Fallback via sampling API (not yet implemented):
           - Use llm.generate() with logprobs=True
           - Extract logits from output
           - More robust but slower and requires decoding

        TODO: Test and validate on GPU pod, implement fallback if needed
    """
    # Access vLLM's internal model
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Auto-generate positions if not provided
    if positions is None:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        # Approach 1: Direct model forward pass
        # This is the most direct approach but relies on vLLM internals

        try:
            # Import vLLM attention metadata
            # Note: This import may fail or have different structure in
            # newer vLLM versions
            from vllm.attention import AttentionMetadata

            batch_size, seq_len = input_ids.shape

            # Construct simplified attention metadata for prefill
            # WARNING: This is a simplified version and may not work with
            # all vLLM versions. AttentionMetadata is a complex dataclass
            # with many fields.
            #
            # For numerical validation, we only need prefill mode:
            # - No decode tokens
            # - No KV cache (enforce_eager=True in LLM config)
            # - Single-sequence batch
            attn_metadata = AttentionMetadata(
                num_prefills=batch_size,
                num_decode_tokens=0,
                slot_mapping=torch.arange(
                    seq_len * batch_size, dtype=torch.long, device=input_ids.device
                ),
                num_prefill_tokens=seq_len * batch_size,
                max_prefill_seq_len=seq_len,
                # Additional fields may be required by vLLM
                # TODO: Validate these on GPU pod and add missing fields
            )

            # Run forward pass through model
            hidden_states = model.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=None,  # No KV cache for numerical validation
                attn_metadata=attn_metadata,
            )

            # Get logits from LM head
            logits = model.compute_logits(hidden_states, sampling_metadata=None)

            return logits

        except (ImportError, AttributeError, TypeError) as e:
            # Fallback: use sampling API
            # This is more robust but slower
            raise NotImplementedError(
                f"Direct vLLM logits extraction failed: {e}\n"
                "TODO: Implement fallback via sampling API (llm.generate with logprobs)"
            ) from e


def extract_vllm_intermediate_activations(
    llm,
    input_ids: torch.Tensor,
    layer_names: list[str],
    positions: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Extract intermediate activations from vLLM model.

    Args:
        llm: vLLM LLM instance
        input_ids: (batch, seq_len) token IDs (on GPU)
        layer_names: List of layer names to extract (e.g., ["model.layers.0"])
        positions: (batch, seq_len) position IDs (optional)

    Returns:
        Dictionary mapping layer names to output tensors

    Note:
        Uses PyTorch forward hooks, same approach as HuggingFace extraction.
        This requires the extract_vllm_logits() function to work correctly.

        Example layer names:
        - "model.layers.0" - First decoder layer output
        - "model.layers.0.self_attn" - Attention output
        - "model.layers.0.mlp" - MoE output
        - "model.norm" - Final RMSNorm output

        TODO: Validate layer names on GPU pod and document correct paths
    """
    # Access vLLM's internal model
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    extractor = TensorExtractor()

    # Register hooks for requested layers
    for layer_name in layer_names:
        # Navigate to the module
        try:
            module = model
            for part in layer_name.split("."):
                module = getattr(module, part)
            extractor.register_hook(module, layer_name)
        except AttributeError as e:
            raise ValueError(
                f"Layer '{layer_name}' not found in vLLM model. "
                f"Use dir(model.model) to explore available layers. Error: {e}"
            ) from e

    # Auto-generate positions if not provided
    if positions is None:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

    # Run forward pass (uses extract_vllm_logits internally)
    with torch.no_grad():
        try:
            # Run full forward pass - hooks will capture intermediate activations
            extract_vllm_logits(llm, input_ids, positions)
        except Exception as e:
            # Clean up hooks before re-raising
            extractor.remove_hooks()
            raise NotImplementedError(
                f"vLLM intermediate extraction failed: {e}\n"
                "This likely means extract_vllm_logits() needs to be fixed first."
            ) from e

    # Collect tensors
    tensors = {}
    for name in layer_names:
        tensor = extractor.get_tensor(name)
        if tensor is None:
            raise ValueError(
                f"Layer '{name}' hook did not capture any tensor. "
                f"This may indicate the layer was not executed during forward pass."
            )
        tensors[name] = tensor

    # Cleanup
    extractor.remove_hooks()

    return tensors
