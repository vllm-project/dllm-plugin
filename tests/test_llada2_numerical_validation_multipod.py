#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 9.1 Multi-Pod Numerical Validation Tests for LLaDA2.0

This test suite compares vLLM/dllm-plugin implementation against pre-extracted
HuggingFace reference tensors from Pod 1.

Architecture:
- Pod 1 (HuggingFace): Extracted reference tensors with transformers 5.1.0
- Pod 3 (vLLM): This pod, runs vLLM 0.20.1 + dllm-plugin
- File-based transfer: HF tensors uploaded to /workspace/validation_outputs_hf

Test Hierarchy:
- TestEmbeddingValidation: Validation Point 1
- TestE2EValidation: Validation Point 8 (end-to-end)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

# Mark entire module for GPU integration
pytestmark = [
    pytest.mark.dllm_gpu_integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU"),
]

# Reference tensor location (uploaded from Pod 1)
REFERENCE_DIR = Path("/workspace/validation_outputs_hf")


@pytest.fixture(scope="module")
def fixed_seed():
    """Set random seeds for reproducible tests."""
    import random

    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture(scope="module")
def hf_reference_tensors():
    """Load pre-extracted HuggingFace reference tensors."""
    if not REFERENCE_DIR.exists():
        pytest.skip(f"Reference tensors not found at {REFERENCE_DIR}")

    scenarios = {}
    for scenario_dir in REFERENCE_DIR.iterdir():
        if scenario_dir.is_dir():
            scenario_name = scenario_dir.name
            scenarios[scenario_name] = {
                "dir": scenario_dir,
                "input_ids": torch.load(scenario_dir / "input_ids.pt"),
            }

            # Load metadata
            with open(scenario_dir / "metadata.json") as f:
                scenarios[scenario_name]["metadata"] = json.load(f)

            # Load all tensor files
            for tensor_file in scenario_dir.glob("hf_*.pt"):
                tensor_name = tensor_file.stem.replace("hf_", "")
                scenarios[scenario_name][tensor_name] = torch.load(tensor_file)

    return scenarios


@pytest.fixture(scope="module")
def vllm_model(llada2_real_model_dir: Path):
    """Load vLLM/dllm-plugin model."""
    # Critical setup
    os.environ["VLLM_PLUGINS"] = "dllm"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DLLM_USE_MOCK_MODEL"] = "0"

    from vllm import LLM

    from dllm_plugin import register_dllm

    register_dllm()

    llm = LLM(
        model=str(llada2_real_model_dir),
        trust_remote_code=True,
        model_impl="dllm_plugin.models.llada2:LLaDA2ForCausalLM",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=256,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="dllm_plugin.Worker",
    )

    return llm


class TestEmbeddingValidation:
    """Validation Point 1: Embedding Layer (Exact Match Expected)"""

    @pytest.mark.parametrize(
        "scenario",
        ["single_token", "short_seq", "full_block", "multi_block"],
    )
    def test_embedding_exact_match(
        self,
        scenario: str,
        hf_reference_tensors,
        vllm_model,
        fixed_seed,
    ):
        """Test that embeddings produce exact match (deterministic lookup)."""
        # Load HF reference
        hf_data = hf_reference_tensors[scenario]
        hf_embeddings = hf_data["embeddings"].cuda()
        input_ids = hf_data["input_ids"].cuda()

        print(f"\n=== {scenario} ===")
        print(f"Input shape: {input_ids.shape}")
        print(f"HF embeddings shape: {hf_embeddings.shape}")

        # Extract vLLM embeddings
        from tests.test_helpers.tensor_extraction import extract_vllm_embeddings

        vllm_embeddings = extract_vllm_embeddings(vllm_model, input_ids)
        print(f"vLLM embeddings shape: {vllm_embeddings.shape}")

        # Compare: EXACT match expected (deterministic lookup)
        # Allow tiny floating point error from GPU computations
        from dllm_plugin.validation_utils import ToleranceBounds, assert_tensors_close

        tolerance = ToleranceBounds(
            atol=1e-6,
            rtol=1e-6,
            description="Embedding lookup (near-exact match with tiny FP error)",
        )

        metrics = assert_tensors_close(
            vllm_embeddings,
            hf_embeddings,
            tolerance=tolerance,
            name=f"{scenario} embeddings",
        )

        print(f"Max abs diff: {metrics['max_abs_diff']:.2e}")
        print(f"Mean abs diff: {metrics['mean_abs_diff']:.2e}")
        print(f"Max rel diff: {metrics['max_rel_diff']:.2e}")
        print("✓ Embeddings match within tolerance")


class TestE2EValidation:
    """Validation Point 8: End-to-End Logits Validation"""

    @pytest.mark.parametrize(
        "scenario",
        ["single_token", "short_seq", "full_block", "multi_block"],
    )
    def test_e2e_logits(
        self,
        scenario: str,
        hf_reference_tensors,
        vllm_model,
        fixed_seed,
    ):
        """Test end-to-end logits comparison."""
        # Load HF reference
        hf_data = hf_reference_tensors[scenario]
        hf_logits = hf_data["logits_e2e"].cuda()
        input_ids = hf_data["input_ids"].cuda()

        print(f"\n=== {scenario} ===")
        print(f"Input shape: {input_ids.shape}")
        print(f"HF logits shape: {hf_logits.shape}")

        # Extract vLLM logits
        from tests.test_helpers.tensor_extraction import extract_vllm_logits

        try:
            vllm_logits = extract_vllm_logits(vllm_model, input_ids)
            print(f"vLLM logits shape: {vllm_logits.shape}")
        except NotImplementedError as e:
            pytest.skip(f"vLLM logits extraction not yet implemented: {e}")

        # Compare with accumulated error tolerance
        from dllm_plugin.validation_utils import (
            TOLERANCE_ACCUMULATED,
            assert_tensors_close,
        )

        metrics = assert_tensors_close(
            vllm_logits,
            hf_logits,
            tolerance=TOLERANCE_ACCUMULATED,
            name=f"{scenario} E2E logits",
        )

        print(f"Max abs diff: {metrics['max_abs_diff']:.2e}")
        print(f"Mean abs diff: {metrics['mean_abs_diff']:.2e}")
        print(f"Max rel diff: {metrics['max_rel_diff']:.2e}")

        # Check top-k prediction agreement
        top_k = 10
        _, hf_top_indices = torch.topk(hf_logits, k=top_k, dim=-1)
        _, vllm_top_indices = torch.topk(vllm_logits, k=top_k, dim=-1)

        # Top-1 agreement
        top1_agreement = (
            (hf_top_indices[..., 0] == vllm_top_indices[..., 0]).float().mean().item()
        )
        print(f"Top-1 prediction agreement: {top1_agreement * 100:.1f}%")

        # Top-k agreement (how many of top-k predictions match)
        top_k_agreement = 0.0
        for i in range(top_k):
            matches = (
                (hf_top_indices[..., i : i + 1] == vllm_top_indices)
                .any(dim=-1)
                .float()
                .mean()
                .item()
            )
            top_k_agreement += matches
        top_k_agreement /= top_k
        print(f"Top-{top_k} prediction agreement: {top_k_agreement * 100:.1f}%")

        # Assert reasonable agreement
        assert top1_agreement > 0.95, (
            f"Top-1 agreement too low: {top1_agreement * 100:.1f}%"
        )
        print("✓ E2E validation passed")


if __name__ == "__main__":
    # Allow running this file directly for debugging
    pytest.main([__file__, "-v", "-s"])
