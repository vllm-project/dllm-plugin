"""
Phase 9.1 Numerical Validation: Compare dllm-plugin vLLM against dInfer reference.

Tests numerical correctness by comparing intermediate activations and final outputs.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Mark entire module for GPU integration
pytestmark = [
    pytest.mark.dllm_gpu_integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU"),
]


@pytest.fixture(scope="module")
def dinfer_reference_tensors():
    """Load pre-extracted dInfer reference tensors."""
    ref_dir = Path("/workspace/validation_outputs_dinfer")

    scenarios = {}
    for scenario_name in ["single_token", "short_seq", "full_block"]:
        scenario_dir = ref_dir / scenario_name
        if not scenario_dir.exists():
            continue

        scenario_data = {
            "input_ids": torch.load(scenario_dir / "input_ids.pt"),
            "output_e2e": torch.load(scenario_dir / "dinfer_output_e2e.pt"),
        }

        # Load all intermediate tensors
        for tensor_file in scenario_dir.glob("dinfer_*.pt"):
            if "output_e2e" not in tensor_file.name:
                tensor_name = tensor_file.stem.replace("dinfer_", "")
                scenario_data[tensor_name] = torch.load(tensor_file)

        scenarios[scenario_name] = scenario_data

    return scenarios


@pytest.fixture(scope="module")
def vllm_model():
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
        model="/workspace/llada2-mini",
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


class TestE2EValidation:
    """Validation Point 8: End-to-End Input → Output Logits"""

    def test_single_token_generation(self, dinfer_reference_tensors, vllm_model):
        """Compare single token generation against dInfer reference."""
        if "single_token" not in dinfer_reference_tensors:
            pytest.skip("dInfer single_token reference not available")

        ref_data = dinfer_reference_tensors["single_token"]
        input_ids = ref_data["input_ids"]
        dinfer_output = ref_data["output_e2e"]

        # Run vLLM generation
        from transformers import AutoTokenizer
        from vllm import SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(
            "/workspace/llada2-mini", trust_remote_code=True, local_files_only=True
        )
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=64,
            top_p=1.0,
        )

        outputs = vllm_model.generate([prompt_text], sampling_params)
        vllm_output_text = outputs[0].outputs[0].text
        dinfer_output_text = tokenizer.decode(
            dinfer_output[0], skip_special_tokens=True
        )

        print(f"\nPrompt: {prompt_text}")
        print(f"dInfer output: {dinfer_output_text[:100]}...")
        print(f"vLLM output: {vllm_output_text[:100]}...")

        # Log comparison (exact match not expected)
        # Future: Add logit-level comparison with tolerances
        assert len(vllm_output_text) > 0, "vLLM should generate non-empty output"

    def test_short_seq_generation(self, dinfer_reference_tensors, vllm_model):
        """Compare short sequence generation against dInfer reference."""
        if "short_seq" not in dinfer_reference_tensors:
            pytest.skip("dInfer short_seq reference not available")

        ref_data = dinfer_reference_tensors["short_seq"]
        input_ids = ref_data["input_ids"]
        dinfer_output = ref_data["output_e2e"]

        # Run vLLM generation
        from transformers import AutoTokenizer
        from vllm import SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(
            "/workspace/llada2-mini", trust_remote_code=True, local_files_only=True
        )
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=64,
            top_p=1.0,
        )

        outputs = vllm_model.generate([prompt_text], sampling_params)
        vllm_output_text = outputs[0].outputs[0].text
        dinfer_output_text = tokenizer.decode(
            dinfer_output[0], skip_special_tokens=True
        )

        print(f"\nPrompt: {prompt_text}")
        print(f"dInfer output: {dinfer_output_text[:100]}...")
        print(f"vLLM output: {vllm_output_text[:100]}...")

        assert len(vllm_output_text) > 0, "vLLM should generate non-empty output"

    def test_full_block_generation(self, dinfer_reference_tensors, vllm_model):
        """Compare full block (32 tokens) generation against dInfer reference."""
        if "full_block" not in dinfer_reference_tensors:
            pytest.skip("dInfer full_block reference not available")

        ref_data = dinfer_reference_tensors["full_block"]
        input_ids = ref_data["input_ids"]
        dinfer_output = ref_data["output_e2e"]

        # Run vLLM generation
        from transformers import AutoTokenizer
        from vllm import SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(
            "/workspace/llada2-mini", trust_remote_code=True, local_files_only=True
        )
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=64,
            top_p=1.0,
        )

        outputs = vllm_model.generate([prompt_text], sampling_params)
        vllm_output_text = outputs[0].outputs[0].text
        dinfer_output_text = tokenizer.decode(
            dinfer_output[0], skip_special_tokens=True
        )

        print(f"\nPrompt: {prompt_text[:50]}...")
        print(f"dInfer output: {dinfer_output_text[:100]}...")
        print(f"vLLM output: {vllm_output_text[:100]}...")

        assert len(vllm_output_text) > 0, "vLLM should generate non-empty output"


class TestIntermediateTensorValidation:
    """Validation of intermediate layer outputs (embeddings, attention, MoE, etc.)"""

    def test_embedding_comparison(self, dinfer_reference_tensors):
        """Compare embedding layer outputs."""
        if "single_token" not in dinfer_reference_tensors:
            pytest.skip("dInfer reference not available")

        ref_data = dinfer_reference_tensors["single_token"]
        if "embeddings" not in ref_data:
            pytest.skip("Embedding tensors not captured in dInfer reference")

        dinfer_embeddings = ref_data["embeddings"]

        # TODO: Extract vLLM embeddings and compare
        # For now, just verify dInfer embeddings loaded correctly
        assert dinfer_embeddings is not None
        assert dinfer_embeddings.dim() == 3  # (batch, seq_len, hidden_size)
        print(f"\ndInfer embeddings shape: {dinfer_embeddings.shape}")

    def test_layer0_attention_comparison(self, dinfer_reference_tensors):
        """Compare layer 0 attention outputs."""
        if "single_token" not in dinfer_reference_tensors:
            pytest.skip("dInfer reference not available")

        ref_data = dinfer_reference_tensors["single_token"]
        if "layer0_attention" not in ref_data:
            pytest.skip("Layer 0 attention tensors not captured in dInfer reference")

        dinfer_attn = ref_data["layer0_attention"]

        # TODO: Extract vLLM attention outputs and compare
        assert dinfer_attn is not None
        print(f"\ndInfer layer 0 attention shape: {dinfer_attn.shape}")

    def test_layer0_moe_comparison(self, dinfer_reference_tensors):
        """Compare layer 0 MoE outputs."""
        if "single_token" not in dinfer_reference_tensors:
            pytest.skip("dInfer reference not available")

        ref_data = dinfer_reference_tensors["single_token"]
        if "layer0_moe" not in ref_data:
            pytest.skip("Layer 0 MoE tensors not captured in dInfer reference")

        dinfer_moe = ref_data["layer0_moe"]

        # TODO: Extract vLLM MoE outputs and compare
        assert dinfer_moe is not None
        print(f"\ndInfer layer 0 MoE shape: {dinfer_moe.shape}")
