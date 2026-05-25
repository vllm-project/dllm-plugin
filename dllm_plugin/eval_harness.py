# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""lm-eval harness for dllm-plugin.

Registers ``dllm_plugin_eval`` as an lm-eval model backend that uses
vLLM with the dLLM scheduler and worker plugins for offline inference.
Exercises the full production path: ModelState → DiffusionSampler →
Triton kernel → DllmRuntimeScheduler.

Usage::

    lm_eval --model dllm_plugin_eval \
        --model_args model_path=inclusionAI/LLaDA2.0-mini,gen_length=2048 \
        --tasks gsm8k_llada_mini \
        --include_path evaluations/tasks
"""

from __future__ import annotations

import logging
import os
from typing import Any

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

logger = logging.getLogger(__name__)


@register_model("dllm_plugin_eval")
class DllmPluginEvalHarness(LM):
    """lm-eval backend that runs LLaDA2 via vLLM + dllm-plugin."""

    def __init__(
        self,
        model_path: str = "",
        gen_length: int = 2048,
        max_model_len: int = 2048,
        max_num_seqs: int = 4,
        batch_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        os.environ.setdefault("VLLM_PLUGINS", "dllm")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        from vllm import LLM

        self._gen_length = int(gen_length)
        self._batch_size = int(batch_size)
        self.model_path = model_path

        logger.info("Loading vLLM with dllm-plugin: %s", model_path)
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            gpu_memory_utilization=float(gpu_memory_utilization),
            enforce_eager=True,
            scheduler_cls="dllm_plugin.runtime_scheduler.DllmRuntimeScheduler",
            worker_cls="dllm_plugin.runtime_worker.DllmRuntimeWorker",
        )
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM loaded successfully")

    def generate_until(self, requests: list) -> list[str]:
        from vllm import SamplingParams

        prompts = []
        gen_kwargs_list = []
        for req in requests:
            prompt = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            prompts.append(prompt)
            gen_kwargs_list.append(gen_kwargs)

        answers: list[str] = []
        for i in range(0, len(prompts), self._batch_size):
            batch_prompts = prompts[i : i + self._batch_size]
            batch_kwargs = gen_kwargs_list[i : i + self._batch_size]

            stop_seqs = batch_kwargs[0].get("until", [])
            if isinstance(stop_seqs, str):
                stop_seqs = [stop_seqs]

            params = SamplingParams(
                max_tokens=self._gen_length,
                temperature=0.0,
                stop=stop_seqs if stop_seqs else None,
            )

            outputs = self.llm.generate(batch_prompts, params)
            for output in outputs:
                text = output.outputs[0].text
                answers.append(text)

            if (i // self._batch_size) % 10 == 0:
                logger.info(
                    "Generated %d/%d (batch %d)",
                    min(i + self._batch_size, len(prompts)),
                    len(prompts),
                    i // self._batch_size,
                )

        return answers

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "loglikelihood requires Monte Carlo estimation for diffusion "
            "models. Use generation tasks (GSM8K) instead."
        )

    def loglikelihood_rolling(self, requests: list) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "loglikelihood_rolling not supported for diffusion models."
        )
