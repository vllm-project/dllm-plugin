# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 6 mock-stack integration test with concrete vLLM runtime objects."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_vllm_mock_stack_end_to_end_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model_dir = Path(__file__).parent / "fixtures" / "mock_llada2_hf_config"

    monkeypatch.setenv("VLLM_PLUGINS", "dllm")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        skip_tokenizer_init=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=128,
        max_num_seqs=1,
        load_format="dummy",
        scheduler_cls="dllm_plugin.Scheduler",
        worker_cls="dllm_plugin.Worker",
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2, 3, 4])],
        SamplingParams(max_tokens=1, temperature=0.0, detokenize=False),
    )
    assert outputs
    assert outputs[0].outputs
    assert outputs[0].outputs[0].token_ids
