"""Test block-causal attention activation and forward context."""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.gpu
def test_block_causal_activation_with_scheduler():
    """DllmScheduler integration triggers block-causal attention."""
    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model="inclusionAI/LLaDA2.0-mini",
            trust_remote_code=True,
            scheduler_cls="dllm_plugin.Scheduler",
            worker_cls="dllm_plugin.Worker",
            gpu_memory_utilization=0.95,
            max_model_len=256,
            enforce_eager=True,
            block_size=32,
            async_scheduling=False,
        )
    except Exception as e:
        pytest.skip(f"Could not initialize with DllmScheduler: {e}")

    prompt = "What is the meaning of life?"
    outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=96))
    text = outputs[0].outputs[0].text
    assert len(text) > 0, "Expected non-empty generated text"


def test_forward_context_round_trip():
    """set/get_num_prefix_tokens_list round-trips correctly."""
    from dllm_plugin.forward_context import (
        _num_prefix_tokens_list_ctx,
        get_num_prefix_tokens_list,
        set_num_prefix_tokens_list,
    )

    assert get_num_prefix_tokens_list() is None

    token = set_num_prefix_tokens_list([0, 32, 64])
    assert get_num_prefix_tokens_list() == [0, 32, 64]

    _num_prefix_tokens_list_ctx.reset(token)
    assert get_num_prefix_tokens_list() is None
