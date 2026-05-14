#!/usr/bin/env python3
"""Shared capture-replay library for dInfer vs dllm-plugin numerical investigation.

Provides a unified framework for capturing intermediate tensors at 7 granularity
levels from both dInfer and dllm-plugin implementations of LLaDA2.0-mini.

Usage:
    from capture_lib import CaptureRegistry, make_capture_hook, load_config

    config = load_config()
    registry = CaptureRegistry(framework="dinfer", scenario="first_block")
    hook = make_capture_hook(registry, "L4_layer.layer00.hidden_out")
    model.layers[0].register_forward_hook(hook)
    # ... run forward pass ...
    registry.save(config["capture_root"])
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

LEVELS = [
    "L1_output",
    "L2_probs",
    "L3_logits",
    "L4_layer",
    "L5_sublayer",
    "L6_subattn",
    "L7_atomic",
]

L5_OPS = [
    "input_norm_out",
    "attn_out",
    "residual1",
    "post_attn_norm_out",
    "moe_out",
    "residual2",
]

L6_OPS = [
    "qkv_proj_out",
    "q_split",
    "k_split",
    "v_split",
    "q_norm",
    "k_norm",
    "q_rope",
    "k_rope",
    "attn_output",
    "o_proj_out",
]

L7_OPS = [
    "rmsnorm_variance",
    "rmsnorm_rsqrt",
    "rmsnorm_normalized",
    "rmsnorm_scaled",
    "gate_logits",
    "gate_sigmoid",
    "group_scores",
    "top_groups",
    "expert_weights",
    "expert_indices",
    "routed_output",
    "shared_gate",
    "shared_up",
    "shared_silu",
    "shared_hidden",
    "shared_output",
    "moe_combined",
    "rope_cos",
    "rope_sin",
]


def load_config(config_path: str | Path | None = None) -> dict:
    """Load shared investigation config."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


def checkpoint_name(level: str, layer: str, operation: str) -> str:
    """Build a checkpoint name from components."""
    return f"{level}.{layer}.{operation}"


def filename_for_checkpoint(
    framework: str, scenario: str, level: str, layer: str, operation: str
) -> str:
    """Build the .pt filename for a checkpoint."""
    return f"{framework}.{scenario}.{level}.{layer}.{operation}.pt"


@dataclass
class CaptureMetadata:
    """Metadata for a single captured tensor."""

    name: str
    shape: list[int]
    dtype: str
    numel: int
    description: str = ""

    @classmethod
    def from_tensor(
        cls, name: str, tensor: torch.Tensor, description: str = ""
    ) -> CaptureMetadata:
        return cls(
            name=name,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            numel=tensor.numel(),
            description=description,
        )


class CaptureRegistry:
    """Collects and saves captured tensors with systematic naming.

    Each capture is stored as:
        {capture_root}/{scenario}/{framework}.{scenario}.{level}.{layer}.{op}.pt
    """

    def __init__(self, framework: str, scenario: str):
        self.framework = framework
        self.scenario = scenario
        self.captures: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, CaptureMetadata] = {}
        self.model_config: dict[str, Any] = {}
        self.input_ids: list[int] = []
        self.positions: list[int] = []
        self.extra: dict[str, Any] = {}

    def register(self, name: str, tensor: torch.Tensor, description: str = "") -> None:
        """Register a captured tensor."""
        t = tensor.detach().cpu().clone()
        self.captures[name] = t
        self.metadata[name] = CaptureMetadata.from_tensor(name, t, description)

    def register_non_tensor(self, name: str, data: Any, description: str = "") -> None:
        """Register non-tensor data (indices, metadata)."""
        if isinstance(data, torch.Tensor):
            self.register(name, data, description)
            return
        self.extra[name] = data

    def save(self, capture_root: str | Path) -> Path:
        """Save all captures to {capture_root}/{scenario}/."""
        out_dir = Path(capture_root) / self.scenario
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, tensor in self.captures.items():
            fname = f"{self.framework}.{self.scenario}.{name}.pt"
            torch.save(tensor, out_dir / fname)

        # Save metadata
        meta = {
            "framework": self.framework,
            "scenario": self.scenario,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_config": self.model_config,
            "input_ids": self.input_ids,
            "positions": self.positions,
            "num_captures": len(self.captures),
            "captures": {k: asdict(v) for k, v in self.metadata.items()},
            "extra": {k: _serialize(v) for k, v in self.extra.items()},
        }
        meta_path = out_dir / f"{self.framework}.{self.scenario}.metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        total_mb = (
            sum(t.numel() * t.element_size() for t in self.captures.values()) / 1e6
        )
        print(f"[SAVE] {len(self.captures)} captures ({total_mb:.1f} MB) -> {out_dir}/")
        return out_dir

    @classmethod
    def load(
        cls, capture_root: str | Path, framework: str, scenario: str
    ) -> CaptureRegistry:
        """Load captures from disk."""
        in_dir = Path(capture_root) / scenario
        meta_path = in_dir / f"{framework}.{scenario}.metadata.json"

        registry = cls(framework=framework, scenario=scenario)

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            registry.model_config = meta.get("model_config", {})
            registry.input_ids = meta.get("input_ids", [])
            registry.positions = meta.get("positions", [])
            registry.extra = meta.get("extra", {})

        prefix = f"{framework}.{scenario}."
        for pt_file in sorted(in_dir.glob(f"{prefix}*.pt")):
            name = pt_file.stem[len(prefix) :]
            tensor = torch.load(pt_file, map_location="cpu", weights_only=True)
            registry.captures[name] = tensor
            registry.metadata[name] = CaptureMetadata.from_tensor(name, tensor)

        print(f"[LOAD] {len(registry.captures)} captures from {in_dir}/")
        return registry


def make_capture_hook(
    registry: CaptureRegistry, name: str, output_index: int | None = None
) -> Callable:
    """Create a PyTorch forward hook that captures module output."""

    def hook(module, input, output):
        try:
            if isinstance(output, tuple):
                tensor = output[output_index if output_index is not None else 0]
            else:
                tensor = output
            if isinstance(tensor, torch.Tensor):
                registry.register(name, tensor)
        except Exception as e:
            print(f"[HOOK ERROR] {name}: {e}")

    return hook


def make_capture_pre_hook(
    registry: CaptureRegistry, name: str, input_index: int = 0
) -> Callable:
    """Create a PyTorch forward pre-hook that captures module input."""

    def hook(module, input):
        try:
            tensor = input[input_index] if isinstance(input, tuple) else input
            if isinstance(tensor, torch.Tensor):
                registry.register(name, tensor)
        except Exception as e:
            print(f"[PRE-HOOK ERROR] {name}: {e}")

    return hook


def normalize_for_comparison(tensor: torch.Tensor, framework: str) -> torch.Tensor:
    """Normalize tensor shapes for cross-framework comparison.

    dInfer uses [batch=1, seq_len, ...], vLLM uses [num_tokens, ...].
    """
    if framework == "dinfer" and tensor.dim() >= 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def assert_config_match(meta_a: dict, meta_b: dict) -> None:
    """Validate that two captures used identical model configurations."""
    keys = [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "rope_theta",
        "partial_rotary_factor",
        "num_experts",
        "num_experts_per_tok",
    ]
    mismatches = []
    for key in keys:
        va = meta_a.get(key)
        vb = meta_b.get(key)
        if va != vb:
            mismatches.append(f"  {key}: {va} vs {vb}")
    if mismatches:
        raise ValueError("Model config mismatch:\n" + "\n".join(mismatches))


def extract_model_config(config) -> dict:
    """Extract relevant config values from a HuggingFace PretrainedConfig."""
    keys = [
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "rope_theta",
        "partial_rotary_factor",
        "num_experts",
        "num_experts_per_tok",
        "num_shared_experts",
        "n_group",
        "topk_group",
        "routed_scaling_factor",
        "rms_norm_eps",
        "vocab_size",
        "max_position_embeddings",
    ]
    result = {}
    for key in keys:
        val = getattr(config, key, None)
        if val is not None:
            result[key] = val
    return result


def _serialize(v: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(v, torch.Tensor):
        return {"type": "tensor", "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (list, tuple)):
        return [_serialize(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _serialize(val) for k, val in v.items()}
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)
