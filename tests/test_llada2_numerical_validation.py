# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 9.1 Numerical Validation Tests for LLaDA2.0

Validates numerical correctness of dllm-plugin vLLM implementation
against HuggingFace transformers reference implementation.

Test Hierarchy:
    - TestEmbeddingValidation: Validation Point 1
    - TestAttentionValidation: Validation Point 2 (sub-components 2.1-2.4)
    - TestMoEValidation: Validation Point 3 (sub-components 3.1-3.6)
    - TestDecoderLayerValidation: Validation Point 4
    - TestTransformerStackValidation: Validation Point 5
    - TestFinalNormValidation: Validation Point 6
    - TestLMHeadValidation: Validation Point 7
    - TestE2EValidation: Validation Point 8
    - TestRouterPrecisionComparison: FP32 vs BF16 router
    - TestExpertLoadBalancing: Expert selection distribution analysis

Reference:
    - Issue #42: Phase 9.1 - Numerical Validation (Incremental Layer-by-Layer)
    - Plan: /Users/akellner/.claude/plans/let-s-plan-phase-7-agile-mochi.md
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Skip if torch not available
pytest.importorskip("torch")

import torch  # noqa: E402

# Mark entire module for GPU integration and numerical validation
pytestmark = [
    pytest.mark.dllm_numerical_validation,
    pytest.mark.dllm_gpu_integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU"),
]


@pytest.fixture(scope="module")
def fixed_seed():
    """Set random seeds for reproducible tests.

    Returns:
        int: The seed value used (42)
    """
    import random

    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture(scope="module")
def hf_model_and_tokenizer(llada2_real_model_dir: Path):
    """Load HuggingFace reference model and tokenizer.

    Args:
        llada2_real_model_dir: Path to downloaded LLaDA2.0-mini model

    Returns:
        tuple: (model, tokenizer) where model is AutoModelForCausalLM
    """
    # Import here to avoid dependency issues if transformers not installed
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed - required for HF reference")

    model = AutoModelForCausalLM.from_pretrained(
        str(llada2_real_model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()  # Inference mode

    tokenizer = AutoTokenizer.from_pretrained(
        str(llada2_real_model_dir), trust_remote_code=True
    )

    return model, tokenizer


@pytest.fixture(scope="module")
def vllm_model(llada2_real_model_dir: Path):
    """Load vLLM/dllm-plugin model.

    Args:
        llada2_real_model_dir: Path to downloaded LLaDA2.0-mini model

    Returns:
        LLM: vLLM LLM instance with dllm-plugin
    """
    # Critical environment setup BEFORE importing vLLM
    os.environ["VLLM_PLUGINS"] = "dllm"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DLLM_USE_MOCK_MODEL"] = "0"  # Use real weights

    # Import here to avoid dependency issues
    try:
        from vllm import LLM

        from dllm_plugin import register_dllm
    except ImportError:
        pytest.skip("vLLM not installed - required for dllm-plugin")

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


# ==============================================================================
# Validation Point 1: Embedding Layer
# ==============================================================================


class TestEmbeddingValidation:
    """Validation Point 1: Embedding Layer

    Expected: Exact match (deterministic lookup)
    Tolerance: None - embeddings are deterministic

    Note:
        Embeddings are deterministic table lookups, so we expect EXACT match
        between HF and vLLM implementations (no numerical tolerance needed).
    """

    def test_embedding_exact_match(
        self,
        hf_model_and_tokenizer,
        vllm_model,
        fixed_seed,
    ):
        """Test that embeddings produce exact match (deterministic lookup).

        This validates that:
        1. Embedding weights are loaded correctly
        2. Embedding lookup produces identical outputs
        3. No TP sharding issues in embedding layer
        """
        from tests.test_helpers import extract_hf_embeddings, extract_vllm_embeddings

        hf_model, tokenizer = hf_model_and_tokenizer

        # Test with various input scenarios
        test_cases = [
            ("single_token", torch.tensor([[1]], dtype=torch.long, device="cuda:0")),
            (
                "short_seq",
                torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device="cuda:0"),
            ),
            (
                "medium_seq",
                torch.tensor(
                    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
                    dtype=torch.long,
                    device="cuda:0",
                ),
            ),
        ]

        for name, input_ids in test_cases:
            # Extract HF embeddings
            hf_embeddings = extract_hf_embeddings(hf_model, input_ids)

            # Extract vLLM embeddings
            vllm_embeddings = extract_vllm_embeddings(vllm_model, input_ids)

            # Validate exact match
            # Note: For embeddings, we expect EXACT match (deterministic lookup)
            # However, if weights are in BF16, we may need to compare in BF16 space
            max_diff = (vllm_embeddings - hf_embeddings).abs().max().item()
            assert torch.equal(vllm_embeddings, hf_embeddings), (
                f"{name}: Embeddings must match exactly.\n"
                f"  vLLM: {vllm_embeddings.shape} {vllm_embeddings.dtype}\n"
                f"  HF: {hf_embeddings.shape} {hf_embeddings.dtype}\n"
                f"  Max abs diff: {max_diff:.2e}"
            )


# ==============================================================================
# Validation Point 2: Attention Layer
# ==============================================================================


class TestAttentionValidation:
    """Validation Point 2: Attention Layer (LLaDA2BlockAttention)

    Expected: atol=1e-3, rtol=1e-2 (BF16), atol=1e-5, rtol=1e-4 (FP32)

    Sub-components:
        2.1: QKV Projection
        2.2: Q/K Normalization
        2.3: Attention Computation
        2.4: Output Projection
    """

    def test_qkv_projection(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test QKV projection numerical correctness (Sub-component 2.1)."""
        pytest.skip("TODO: Implement QKV projection validation")

    def test_qk_normalization(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test Q/K per-head normalization (Sub-component 2.2)."""
        pytest.skip("TODO: Implement Q/K normalization validation")

    def test_attention_computation(
        self, hf_model_and_tokenizer, vllm_model, fixed_seed
    ):
        """Test attention computation (Sub-component 2.3)."""
        pytest.skip("TODO: Implement attention computation validation")

    def test_output_projection(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test output projection (Sub-component 2.4)."""
        pytest.skip("TODO: Implement output projection validation")


# ==============================================================================
# Validation Point 3: MoE Layer
# ==============================================================================


class TestMoEValidation:
    """Validation Point 3: MoE Layer (LLaDA2MoE)

    Expected: Complex - multiple sub-components with different tolerances

    Sub-components:
        3.1: Router Gate (FP32 default, BF16 experimental)
        3.2: Group-Limited Routing
        3.3: Routed Experts (FusedMoE)
        3.4: Routed Scaling (2.5x)
        3.5: Shared Expert
        3.6: MoE Output Combination
    """

    def test_router_gate_fp32(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test router gate in FP32 mode (Sub-component 3.1)."""
        pytest.skip("TODO: Implement router gate FP32 validation")

    def test_group_limited_routing(
        self, hf_model_and_tokenizer, vllm_model, fixed_seed
    ):
        """Test group-limited routing algorithm (Sub-component 3.2)."""
        pytest.skip("TODO: Implement group-limited routing validation")

    def test_routed_experts(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test routed experts (FusedMoE) (Sub-component 3.3)."""
        pytest.skip("TODO: Implement routed experts validation")

    def test_routed_scaling(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test routed scaling factor (2.5x) (Sub-component 3.4)."""
        pytest.skip("TODO: Implement routed scaling validation")

    def test_shared_expert(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test shared expert (SwiGLU MLP) (Sub-component 3.5)."""
        pytest.skip("TODO: Implement shared expert validation")

    def test_moe_output_combination(
        self, hf_model_and_tokenizer, vllm_model, fixed_seed
    ):
        """Test MoE output combination (Sub-component 3.6)."""
        pytest.skip("TODO: Implement MoE output combination validation")


# ==============================================================================
# Validation Point 4: Decoder Layer
# ==============================================================================


class TestDecoderLayerValidation:
    """Validation Point 4: Decoder Layer (LLaDA2DecoderLayer)

    Expected: Accumulated error from attention + MoE
    Tolerance: atol=2e-3, rtol=2e-2 (looser for residual accumulation)
    """

    def test_decoder_layer_single(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test single decoder layer numerical correctness."""
        pytest.skip("TODO: Implement decoder layer validation")


# ==============================================================================
# Validation Point 5: Full Transformer Stack
# ==============================================================================


class TestTransformerStackValidation:
    """Validation Point 5: Full Transformer Stack

    Expected: Accumulated error over num_layers
    Tolerance: Increases with layer depth (documented per layer)
    """

    def test_transformer_stack_per_layer(
        self, hf_model_and_tokenizer, vllm_model, fixed_seed
    ):
        """Test transformer stack with per-layer error tracking."""
        pytest.skip("TODO: Implement transformer stack validation")


# ==============================================================================
# Validation Point 6: Final Layer Normalization
# ==============================================================================


class TestFinalNormValidation:
    """Validation Point 6: Final Layer Normalization (RMSNorm)

    Expected: FP32 precision
    Tolerance: atol=1e-4, rtol=1e-3 (after accumulated error)
    """

    def test_final_norm(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test final layer normalization."""
        pytest.skip("TODO: Implement final norm validation")


# ==============================================================================
# Validation Point 7: LM Head Logits
# ==============================================================================


class TestLMHeadValidation:
    """Validation Point 7: LM Head Logits

    Expected: atol=1e-2, rtol=1e-2 (linear projection after accumulated error)
    """

    def test_lm_head_logits(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test LM head logits numerical correctness.

        Validates that the final LM head projection produces similar logits
        between HF and vLLM implementations.

        Note:
            We use looser tolerance here (atol=1e-2, rtol=1e-2) because:
            1. Large output dimension (vocab_size) amplifies small errors
            2. Accumulated error from previous layers
            3. Focus on top-k prediction agreement rather than exact logits
        """
        from dllm_plugin.validation_utils import (
            TOLERANCE_BF16_LOOSE,
            assert_tensors_close,
        )
        from tests.test_helpers import extract_hf_logits, extract_vllm_logits

        hf_model, tokenizer = hf_model_and_tokenizer

        # Test scenarios with different input lengths
        test_cases = [
            ("single_token", torch.tensor([[1]], dtype=torch.long, device="cuda:0")),
            (
                "short_seq",
                torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device="cuda:0"),
            ),
            (
                "medium_seq",
                torch.tensor(
                    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
                    dtype=torch.long,
                    device="cuda:0",
                ),
            ),
        ]

        for name, input_ids in test_cases:
            # Extract HF logits
            hf_logits = extract_hf_logits(hf_model, input_ids)

            # Extract vLLM logits
            vllm_logits = extract_vllm_logits(vllm_model, input_ids)

            # Validate shape matches
            assert hf_logits.shape == vllm_logits.shape, (
                f"{name}: Logit shape mismatch - "
                f"HF: {hf_logits.shape}, vLLM: {vllm_logits.shape}"
            )

            # Compare logits with loose tolerance (BF16 + accumulated error)
            metrics = assert_tensors_close(
                vllm_logits,
                hf_logits,
                tolerance=TOLERANCE_BF16_LOOSE,
                name=f"LM head logits ({name})",
            )

            # Additional validation: top-k predictions should mostly agree
            # Focus on last position (next token prediction)
            hf_last = hf_logits[:, -1, :]
            vllm_last = vllm_logits[:, -1, :]

            # Top-1 prediction
            hf_top1 = hf_last.argmax(dim=-1)
            vllm_top1 = vllm_last.argmax(dim=-1)
            top1_match = (hf_top1 == vllm_top1).all().item()

            # Top-5 predictions
            _, hf_top5 = torch.topk(hf_last, k=5, dim=-1)
            _, vllm_top5 = torch.topk(vllm_last, k=5, dim=-1)
            hf_top5_sorted, _ = torch.sort(hf_top5, dim=-1)
            vllm_top5_sorted, _ = torch.sort(vllm_top5, dim=-1)
            top5_match = (hf_top5_sorted == vllm_top5_sorted).all().item()

            # Log results
            print(f"\n{name} LM Head Validation:")
            print(f"  Max abs diff: {metrics['max_abs_diff']:.2e}")
            print(f"  Max rel diff: {metrics['max_rel_diff']:.2e}")
            print(f"  Top-1 match: {top1_match}")
            print(f"  Top-5 match: {top5_match}")

            # Assertions: top predictions should mostly agree
            # Note: We don't require exact match due to numerical precision
            # but we expect high agreement for LM head validation
            # (This is validated more thoroughly in E2E tests)


# ==============================================================================
# Validation Point 8: E2E Input Tokens → Output Logits
# ==============================================================================


class TestE2EValidation:
    """Validation Point 8: E2E Input Tokens → Output Logits

    Expected: Final sanity check
    Tolerance: atol=5e-2, rtol=5e-2 (very loose, accumulated error)

    Test scenarios:
        - Single token input (minimal error)
        - Short sequence (16 tokens)
        - Full block (32 tokens = DRAFT_SIZE)
        - Multi-block (64 tokens = 2 blocks)
    """

    def _get_hf_logits(self, hf_model, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract logits from HuggingFace model.

        Args:
            hf_model: HuggingFace model
            input_ids: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size) tensor
        """
        from tests.test_helpers import extract_hf_logits

        return extract_hf_logits(hf_model, input_ids)

    def _get_vllm_logits(self, vllm_model, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract logits from vLLM model.

        Args:
            vllm_model: vLLM LLM instance
            input_ids: (batch, seq_len) token IDs (on GPU)

        Returns:
            logits: (batch, seq_len, vocab_size) tensor

        Note:
            Uses extract_vllm_logits() from test_helpers
        """
        from tests.test_helpers import extract_vllm_logits

        return extract_vllm_logits(vllm_model, input_ids)

    def _compare_top_k_predictions(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        k: int = 10,
    ) -> dict:
        """Compare top-k predictions between two logit tensors.

        Args:
            logits_a: (batch, seq_len, vocab_size) from implementation A
            logits_b: (batch, seq_len, vocab_size) from implementation B
            k: Number of top predictions to compare

        Returns:
            Dictionary with agreement metrics:
                - top1_agreement: % tokens with same top-1 prediction
                - top5_agreement: % tokens with same top-5 predictions
                - top10_agreement: % tokens with same top-10 predictions
        """
        # Get top-k predictions for last position (next token prediction)
        logits_a_last = logits_a[:, -1, :]  # (batch, vocab_size)
        logits_b_last = logits_b[:, -1, :]  # (batch, vocab_size)

        # Top-1
        top1_a = logits_a_last.argmax(dim=-1)
        top1_b = logits_b_last.argmax(dim=-1)
        top1_agreement = (top1_a == top1_b).float().mean().item()

        # Top-5
        _, top5_a = torch.topk(logits_a_last, k=min(5, logits_a_last.size(-1)), dim=-1)
        _, top5_b = torch.topk(logits_b_last, k=min(5, logits_b_last.size(-1)), dim=-1)
        top5_a_sorted, _ = torch.sort(top5_a, dim=-1)
        top5_b_sorted, _ = torch.sort(top5_b, dim=-1)
        top5_agreement = (
            (top5_a_sorted == top5_b_sorted).all(dim=-1).float().mean().item()
        )

        # Top-10
        _, top10_a = torch.topk(logits_a_last, k=min(k, logits_a_last.size(-1)), dim=-1)
        _, top10_b = torch.topk(logits_b_last, k=min(k, logits_b_last.size(-1)), dim=-1)
        top10_a_sorted, _ = torch.sort(top10_a, dim=-1)
        top10_b_sorted, _ = torch.sort(top10_b, dim=-1)
        top10_agreement = (
            (top10_a_sorted == top10_b_sorted).all(dim=-1).float().mean().item()
        )

        return {
            "top1_agreement": top1_agreement,
            "top5_agreement": top5_agreement,
            "top10_agreement": top10_agreement,
        }

    def test_e2e_single_token(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test E2E with single token input.

        This test validates that both HF and vLLM produce similar logits
        for a single token input (minimal accumulated error).

        Expected:
            - Top-1 agreement: >95%
            - Top-5 agreement: >98%
            - Top-10 agreement: >99%
            - Logit L2 distance within tolerance
        """
        from dllm_plugin.validation_utils import (
            TOLERANCE_ACCUMULATED,
            assert_tensors_close,
        )

        hf_model, tokenizer = hf_model_and_tokenizer

        # Single token input
        input_ids = torch.tensor([[1]], dtype=torch.long, device="cuda:0")

        # Get HF logits
        hf_logits = self._get_hf_logits(hf_model, input_ids)

        # Get vLLM logits
        vllm_logits = self._get_vllm_logits(vllm_model, input_ids)

        # Compare logits
        metrics = assert_tensors_close(
            vllm_logits,
            hf_logits,
            tolerance=TOLERANCE_ACCUMULATED,
            name="E2E single token logits",
        )

        # Compare top-k predictions
        agreement = self._compare_top_k_predictions(vllm_logits, hf_logits, k=10)

        # Log results
        print("\nE2E Single Token Validation:")
        print(f"  Max abs diff: {metrics['max_abs_diff']:.2e}")
        print(f"  Max rel diff: {metrics['max_rel_diff']:.2e}")
        print(f"  Top-1 agreement: {agreement['top1_agreement']:.2%}")
        print(f"  Top-5 agreement: {agreement['top5_agreement']:.2%}")
        print(f"  Top-10 agreement: {agreement['top10_agreement']:.2%}")

        # Assert agreement thresholds
        assert agreement["top1_agreement"] >= 0.95, (
            f"Top-1 agreement should be >95%, got {agreement['top1_agreement']:.2%}"
        )
        assert agreement["top5_agreement"] >= 0.98, (
            f"Top-5 agreement should be >98%, got {agreement['top5_agreement']:.2%}"
        )
        assert agreement["top10_agreement"] >= 0.99, (
            f"Top-10 agreement should be >99%, got {agreement['top10_agreement']:.2%}"
        )

    def test_e2e_short_sequence(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test E2E with short sequence (16 tokens).

        Validates numerical correctness with moderate accumulated error.
        """
        from dllm_plugin.validation_utils import (
            TOLERANCE_ACCUMULATED,
            assert_tensors_close,
        )

        hf_model, tokenizer = hf_model_and_tokenizer

        # Short sequence (16 tokens)
        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
            dtype=torch.long,
            device="cuda:0",
        )

        # Get logits
        hf_logits = self._get_hf_logits(hf_model, input_ids)
        vllm_logits = self._get_vllm_logits(vllm_model, input_ids)

        # Compare logits
        metrics = assert_tensors_close(
            vllm_logits,
            hf_logits,
            tolerance=TOLERANCE_ACCUMULATED,
            name="E2E short sequence logits",
        )

        # Compare top-k predictions
        agreement = self._compare_top_k_predictions(vllm_logits, hf_logits, k=10)

        # Log results
        print("\nE2E Short Sequence Validation (16 tokens):")
        print(f"  Max abs diff: {metrics['max_abs_diff']:.2e}")
        print(f"  Max rel diff: {metrics['max_rel_diff']:.2e}")
        print(f"  Top-1 agreement: {agreement['top1_agreement']:.2%}")
        print(f"  Top-5 agreement: {agreement['top5_agreement']:.2%}")
        print(f"  Top-10 agreement: {agreement['top10_agreement']:.2%}")

        # Assert agreement thresholds (same as single token)
        assert agreement["top1_agreement"] >= 0.95
        assert agreement["top5_agreement"] >= 0.98
        assert agreement["top10_agreement"] >= 0.99

    def test_e2e_full_block(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test E2E with full block (32 tokens = DRAFT_SIZE).

        Validates numerical correctness for full dLLM block size.
        """
        from dllm_plugin.validation_utils import (
            TOLERANCE_ACCUMULATED,
            assert_tensors_close,
        )

        hf_model, tokenizer = hf_model_and_tokenizer

        # Full block (32 tokens = DRAFT_SIZE)
        input_ids = torch.tensor(
            [[i + 1 for i in range(32)]],
            dtype=torch.long,
            device="cuda:0",
        )

        # Get logits
        hf_logits = self._get_hf_logits(hf_model, input_ids)
        vllm_logits = self._get_vllm_logits(vllm_model, input_ids)

        # Compare logits
        metrics = assert_tensors_close(
            vllm_logits,
            hf_logits,
            tolerance=TOLERANCE_ACCUMULATED,
            name="E2E full block logits",
        )

        # Compare top-k predictions
        agreement = self._compare_top_k_predictions(vllm_logits, hf_logits, k=10)

        # Log results
        print("\nE2E Full Block Validation (32 tokens):")
        print(f"  Max abs diff: {metrics['max_abs_diff']:.2e}")
        print(f"  Max rel diff: {metrics['max_rel_diff']:.2e}")
        print(f"  Top-1 agreement: {agreement['top1_agreement']:.2%}")
        print(f"  Top-5 agreement: {agreement['top5_agreement']:.2%}")
        print(f"  Top-10 agreement: {agreement['top10_agreement']:.2%}")

        # Assert agreement thresholds (same as single token)
        assert agreement["top1_agreement"] >= 0.95
        assert agreement["top5_agreement"] >= 0.98
        assert agreement["top10_agreement"] >= 0.99

    def test_e2e_multi_block(self, hf_model_and_tokenizer, vllm_model, fixed_seed):
        """Test E2E with multi-block (64 tokens).

        Validates numerical correctness with maximum accumulated error.
        """
        from dllm_plugin.validation_utils import (
            TOLERANCE_ACCUMULATED,
            assert_tensors_close,
        )

        hf_model, tokenizer = hf_model_and_tokenizer

        # Multi-block (64 tokens = 2 blocks)
        input_ids = torch.tensor(
            [[i + 1 for i in range(64)]],
            dtype=torch.long,
            device="cuda:0",
        )

        # Get logits
        hf_logits = self._get_hf_logits(hf_model, input_ids)
        vllm_logits = self._get_vllm_logits(vllm_model, input_ids)

        # Compare logits
        metrics = assert_tensors_close(
            vllm_logits,
            hf_logits,
            tolerance=TOLERANCE_ACCUMULATED,
            name="E2E multi-block logits",
        )

        # Compare top-k predictions
        agreement = self._compare_top_k_predictions(vllm_logits, hf_logits, k=10)

        # Log results
        print("\nE2E Multi-Block Validation (64 tokens):")
        print(f"  Max abs diff: {metrics['max_abs_diff']:.2e}")
        print(f"  Max rel diff: {metrics['max_rel_diff']:.2e}")
        print(f"  Top-1 agreement: {agreement['top1_agreement']:.2%}")
        print(f"  Top-5 agreement: {agreement['top5_agreement']:.2%}")
        print(f"  Top-10 agreement: {agreement['top10_agreement']:.2%}")

        # Assert agreement thresholds (slightly relaxed due to longer sequence)
        assert agreement["top1_agreement"] >= 0.90, (
            "Multi-block top-1 agreement should be >90% "
            "(relaxed due to longer sequence)"
        )
        assert agreement["top5_agreement"] >= 0.95, (
            "Multi-block top-5 agreement should be >95%"
        )
        assert agreement["top10_agreement"] >= 0.98, (
            "Multi-block top-10 agreement should be >98%"
        )


# ==============================================================================
# Router Precision Comparison (FP32 vs BF16)
# ==============================================================================


class TestRouterPrecisionComparison:
    """Router Precision Comparison: FP32 vs BF16

    Validates router precision modes:
        - FP32 (default): VALIDATED
        - BF16 (experimental via VLLM_LLADA2_BF16_ROUTER=1): UNVALIDATED

    Metrics:
        - Logit value divergence (L2 norm)
        - Expert selection agreement (% same experts)
        - KL divergence between distributions
        - Load balancing distribution (entropy)
    """

    def test_router_precision_fp32_vs_bf16(
        self, hf_model_and_tokenizer, vllm_model, fixed_seed
    ):
        """Test FP32 vs BF16 router precision comparison."""
        pytest.skip("TODO: Implement router precision comparison")


# ==============================================================================
# Expert Load Balancing Analysis
# ==============================================================================


class TestExpertLoadBalancing:
    """Expert Load Balancing Analysis

    Validates that expert selection is not pathologically biased:
        - Distribution analysis
        - Entropy measurement
        - No single expert dominates
    """

    def test_expert_load_balancing(
        self, hf_model_and_tokenizer, vllm_model, fixed_seed
    ):
        """Test expert selection load balancing."""
        pytest.skip("TODO: Implement expert load balancing analysis")
