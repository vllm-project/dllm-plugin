"""Hook infrastructure for chunked block attention validation.

This module provides reusable infrastructure for capturing intermediate states
during chunked block attention computation in both vLLM and dInfer implementations.

The 10 critical checkpoints:
1. C1: Input QKV - After QKV projection split
2. C2: Q/K Normalized - After Q/K RMSNorm
3. C3: Q/K After RoPE - After rotary embeddings
4. C4: Prefix Metadata - Virtual batch metadata for prefix chunk
5. C5: Block Metadata - Virtual batch metadata for block chunk
6. C6: Prefix Output - Output from prefix attention chunk
7. C7: Block Output - Output from block attention chunk
8. C8: Combined Output - After prefix + block addition
9. C9: After O Proj - Final output projection
10. C10: Attention Stats - Metadata inspection (seq_lens, block_table)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dllm_plugin.validation_utils import (
    TOLERANCE_BF16,
    TOLERANCE_BF16_LOOSE,
    TOLERANCE_FP32,
    ToleranceBounds,
)


@dataclass(frozen=True)
class ChunkedAttentionCheckpoint:
    """Definition of a single checkpoint in chunked attention flow."""

    name: str
    description: str
    expected_shape: str  # Human-readable shape description
    tolerance: ToleranceBounds | None  # None for metadata checkpoints
    capture_location: str  # Where in the code this is captured


# Define the 10 critical checkpoints with their tolerances
CHECKPOINT_DEFINITIONS: dict[str, ChunkedAttentionCheckpoint] = {
    "c1_qkv_projection": ChunkedAttentionCheckpoint(
        name="c1_qkv_projection",
        description="QKV tensors after projection split",
        expected_shape="[num_tokens, num_heads, head_dim]",
        tolerance=TOLERANCE_BF16,  # Linear layer
        capture_location="After qkv_proj forward hook",
    ),
    "c2_q_normalized": ChunkedAttentionCheckpoint(
        name="c2_q_normalized",
        description="Query tensor after RMSNorm",
        expected_shape="[num_tokens, num_heads, head_dim]",
        tolerance=TOLERANCE_FP32,  # RMSNorm (critical!)
        capture_location="After q_norm forward hook",
    ),
    "c2_k_normalized": ChunkedAttentionCheckpoint(
        name="c2_k_normalized",
        description="Key tensor after RMSNorm",
        expected_shape="[num_tokens, num_heads, head_dim]",
        tolerance=TOLERANCE_FP32,  # RMSNorm (critical!)
        capture_location="After k_norm forward hook",
    ),
    "c3_q_rope": ChunkedAttentionCheckpoint(
        name="c3_q_rope",
        description="Query tensor after RoPE application",
        expected_shape="[num_tokens, num_heads, head_dim]",
        tolerance=TOLERANCE_BF16,  # RoPE rotation
        capture_location="After rotary_emb in attention forward",
    ),
    "c3_k_rope": ChunkedAttentionCheckpoint(
        name="c3_k_rope",
        description="Key tensor after RoPE application",
        expected_shape="[num_tokens, num_heads, head_dim]",
        tolerance=TOLERANCE_BF16,  # RoPE rotation
        capture_location="After rotary_emb in attention forward",
    ),
    "c4_prefix_metadata": ChunkedAttentionCheckpoint(
        name="c4_prefix_metadata",
        description="Attention metadata for prefix chunk",
        expected_shape="CommonAttentionMetadata",
        tolerance=None,  # Structural comparison
        capture_location="Before prefix attention call in _forward_dual_chunk",
    ),
    "c5_block_metadata": ChunkedAttentionCheckpoint(
        name="c5_block_metadata",
        description="Attention metadata for block chunk",
        expected_shape="CommonAttentionMetadata",
        tolerance=None,  # Structural comparison
        capture_location="Before block attention call in _forward_dual_chunk",
    ),
    "c6_prefix_output": ChunkedAttentionCheckpoint(
        name="c6_prefix_output",
        description="Output from prefix attention chunk",
        expected_shape="[num_tokens, hidden_size]",
        tolerance=TOLERANCE_BF16_LOOSE,  # Attention output
        capture_location="After prefix attention call in _forward_dual_chunk",
    ),
    "c7_block_output": ChunkedAttentionCheckpoint(
        name="c7_block_output",
        description="Output from block attention chunk",
        expected_shape="[num_tokens, hidden_size]",
        tolerance=TOLERANCE_BF16_LOOSE,  # Attention output
        capture_location="After block attention call in _forward_dual_chunk",
    ),
    "c8_combined_output": ChunkedAttentionCheckpoint(
        name="c8_combined_output",
        description="Combined output after prefix + block",
        expected_shape="[num_tokens, hidden_size]",
        tolerance=TOLERANCE_BF16_LOOSE,  # Addition
        capture_location="After prefix_output + block_output",
    ),
    "c9_after_o_proj": ChunkedAttentionCheckpoint(
        name="c9_after_o_proj",
        description="Final output after projection",
        expected_shape="[num_tokens, hidden_size]",
        tolerance=TOLERANCE_BF16_LOOSE,  # Final projection
        capture_location="After o_proj forward hook",
    ),
}

# Convenience mapping for tolerance lookup
CHECKPOINT_TOLERANCES: dict[str, ToleranceBounds | None] = {
    name: checkpoint.tolerance for name, checkpoint in CHECKPOINT_DEFINITIONS.items()
}


class ChunkedAttentionCaptureHarness:
    """Base class for capturing chunked attention checkpoints.

    This harness provides common infrastructure for both vLLM and dInfer
    capture implementations. Subclasses implement framework-specific hook
    registration.

    Usage:
        harness = VLLMCaptureHarness(model, layer_idx=0)
        # Run forward pass
        with torch.no_grad():
            output = model(input_ids)
        # Save captures
        harness.save_captures(output_dir, metadata)
    """

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.captures: dict[str, torch.Tensor | dict[str, Any]] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, checkpoint_name: str) -> Callable:
        """Create a forward hook that captures output to self.captures.

        Args:
            checkpoint_name: Name of checkpoint (must be in CHECKPOINT_DEFINITIONS)

        Returns:
            Forward hook function
        """

        def hook(module, input, output):
            try:
                tensor = output[0] if isinstance(output, tuple) else output

                # Detach and move to CPU for storage
                self.captures[checkpoint_name] = tensor.detach().cpu().clone()

            except Exception:
                import logging as _log

                _log.getLogger(__name__).debug(
                    "Capture hook failed for %s", checkpoint_name, exc_info=True
                )

        return hook

    def register_hook(self, module: torch.nn.Module, checkpoint_name: str) -> None:
        """Register a forward hook on a module.

        Args:
            module: PyTorch module to hook
            checkpoint_name: Name of checkpoint
        """
        hook_fn = self._make_hook(checkpoint_name)
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def capture_metadata(
        self, checkpoint_name: str, metadata_dict: dict[str, Any]
    ) -> None:
        """Manually capture metadata (for C4, C5 checkpoints).

        Args:
            checkpoint_name: Name of metadata checkpoint
            metadata_dict: Dictionary containing metadata fields
        """
        self.captures[checkpoint_name] = metadata_dict.copy()

    def save_captures(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        """Save all captured checkpoints to disk.

        Args:
            output_dir: Directory to save captures
            metadata: Metadata about this capture run (prompt, num_prefix_tokens, etc.)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each checkpoint
        for checkpoint_name, capture_data in self.captures.items():
            checkpoint_path = output_dir / f"layer{self.layer_idx}_{checkpoint_name}.pt"

            # Prepare save data
            save_data = {
                "checkpoint_name": checkpoint_name,
                "layer_idx": self.layer_idx,
                "metadata": metadata,
            }

            if isinstance(capture_data, dict):
                # Metadata checkpoint (C4, C5)
                save_data["checkpoint_metadata"] = capture_data
            else:
                # Tensor checkpoint
                save_data["tensor"] = capture_data

            # Save to disk
            torch.save(save_data, checkpoint_path)

        # Save summary metadata
        summary_path = output_dir / f"layer{self.layer_idx}_summary.json"
        summary = {
            "layer_idx": self.layer_idx,
            "num_checkpoints": len(self.captures),
            "checkpoint_names": list(self.captures.keys()),
            "metadata": metadata,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def load_capture(capture_path: str | Path) -> dict[str, Any]:
    """Load a single capture file.

    Args:
        capture_path: Path to .pt file

    Returns:
        Dict with checkpoint_name, layer_idx, metadata, tensor
    """
    data = torch.load(capture_path, weights_only=True)
    return data


def load_all_captures(
    capture_dir: str | Path, layer_idx: int
) -> dict[str, dict[str, Any]]:
    """Load all captures for a specific layer.

    Args:
        capture_dir: Directory containing capture files
        layer_idx: Layer index to load

    Returns:
        Dictionary mapping checkpoint_name -> capture data
    """
    capture_dir = Path(capture_dir)
    captures = {}

    for checkpoint_name in CHECKPOINT_DEFINITIONS:
        capture_path = capture_dir / f"layer{layer_idx}_{checkpoint_name}.pt"
        if capture_path.exists():
            captures[checkpoint_name] = load_capture(capture_path)

    return captures


def compare_checkpoint_metadata(
    vllm_meta: dict[str, Any], dinfer_meta: dict[str, Any], checkpoint_name: str
) -> dict[str, Any]:
    """Compare metadata from C4 or C5 checkpoints.

    Args:
        vllm_meta: vLLM metadata dict
        dinfer_meta: dInfer metadata dict
        checkpoint_name: Name of checkpoint (for reporting)

    Returns:
        Comparison result dict with status and detailed checks
    """
    checks = {}

    # Compare seq_lens
    if "seq_lens" in vllm_meta and "seq_lens" in dinfer_meta:
        vllm_seq = (
            torch.tensor(vllm_meta["seq_lens"])
            if not isinstance(vllm_meta["seq_lens"], torch.Tensor)
            else vllm_meta["seq_lens"]
        )
        dinfer_seq = (
            torch.tensor(dinfer_meta["seq_lens"])
            if not isinstance(dinfer_meta["seq_lens"], torch.Tensor)
            else dinfer_meta["seq_lens"]
        )
        checks["seq_lens_match"] = torch.equal(vllm_seq, dinfer_seq)

    # Compare max_seq_len
    if "max_seq_len" in vllm_meta and "max_seq_len" in dinfer_meta:
        checks["max_seq_len_match"] = (
            vllm_meta["max_seq_len"] == dinfer_meta["max_seq_len"]
        )

    # Compare causal flag
    if "causal" in vllm_meta and "causal" in dinfer_meta:
        checks["causal_match"] = vllm_meta["causal"] == dinfer_meta["causal"]

    # Compare block_table if present
    if "block_table" in vllm_meta and "block_table" in dinfer_meta:
        vllm_bt = (
            torch.tensor(vllm_meta["block_table"])
            if not isinstance(vllm_meta["block_table"], torch.Tensor)
            else vllm_meta["block_table"]
        )
        dinfer_bt = (
            torch.tensor(dinfer_meta["block_table"])
            if not isinstance(dinfer_meta["block_table"], torch.Tensor)
            else dinfer_meta["block_table"]
        )
        checks["block_table_match"] = torch.equal(vllm_bt, dinfer_bt)

    return {
        "checkpoint_name": checkpoint_name,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "vllm_metadata": vllm_meta,
        "dinfer_metadata": dinfer_meta,
    }
