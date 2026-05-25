#!/usr/bin/env python3
"""Minimal OpenAI-compatible server wrapping dInfer BlockDiffusionLLM.

Exposes /v1/completions with SSE streaming for GuideLLM benchmarking.
Single-request at a time (no batching) — matches dInfer's design.

Usage:
    source /workspace/dinfer-env/bin/activate
    python /workspace/scripts/dinfer_server.py \
        --model /workspace/llada2-mini --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dinfer_server")

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=1)

MODEL_STATE: dict[str, Any] = {}


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str | list[str] = ""
    max_tokens: int = 256
    temperature: float = 0.0
    stream: bool = False
    guided_regex: str | None = None


def init_model(model_path: str) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12399")
    os.environ.setdefault("TORCH_DYNAMO_DISABLE", "1")

    from vllm import distributed
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    parallel_config = ParallelConfig(enable_expert_parallel=True)
    vllm_config = VllmConfig(parallel_config=parallel_config)

    with set_current_vllm_config(vllm_config):
        from dinfer.model import LLaDA2MoeModelLM
        model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = LLaDA2MoeModelLM(config=model_config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    from dinfer import (
        BlockDiffusionLLM,
        BlockIteratorFactory,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )

    mask_id = 156895
    eos_id = tokenizer.eos_token_id or 156892

    decoder = ThresholdParallelDecoder(
        temperature=0, threshold=0.9, mask_id=mask_id, eos_id=eos_id,
    )
    dllm = BlockDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True,
    )

    MODEL_STATE.update({
        "dllm": dllm,
        "tokenizer": tokenizer,
        "device": device,
        "vllm_config": vllm_config,
        "model_path": model_path,
        "mask_id": mask_id,
        "block_length": 32,
    })
    logger.info("Model loaded: %s", model_path)


def generate_sync(prompt: str, max_tokens: int, temperature: float) -> tuple:
    dllm = MODEL_STATE["dllm"]
    tokenizer = MODEL_STATE["tokenizer"]
    device = MODEL_STATE["device"]
    vllm_config = MODEL_STATE["vllm_config"]
    block_length = MODEL_STATE["block_length"]

    input_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    gen_length = max_tokens
    # Round up to block boundary
    if gen_length % block_length != 0:
        gen_length = ((gen_length // block_length) + 1) * block_length

    from vllm.config import get_current_vllm_config, set_current_vllm_config
    from vllm.forward_context import set_forward_context

    with set_current_vllm_config(vllm_config):
        with set_forward_context(None, vllm_config):
            outputs = dllm.generate(
                input_ids,
                gen_length=gen_length,
                block_length=block_length,
            )

    generated_ids = outputs[0, input_ids.shape[1]:].cpu().tolist()
    # Truncate to max_tokens
    generated_ids = generated_ids[:max_tokens]

    # Remove mask tokens and EOS
    eos_id = tokenizer.eos_token_id or 156892
    mask_id = MODEL_STATE["mask_id"]
    clean_ids = []
    for tid in generated_ids:
        if tid == eos_id:
            break
        if tid != mask_id:
            clean_ids.append(tid)

    text = tokenizer.decode(clean_ids, skip_special_tokens=True)
    return text, len(clean_ids), input_ids.shape[1]


def generate_block_by_block(prompt: str, max_tokens: int, temperature: float):
    """Generate one block at a time, yielding (text, token_count) per block.

    This matches the plugin's behavior: each block is a separate forward
    pass through the denoising loop, and results are streamed as each
    block converges.
    """
    dllm = MODEL_STATE["dllm"]
    tokenizer = MODEL_STATE["tokenizer"]
    device = MODEL_STATE["device"]
    vllm_config = MODEL_STATE["vllm_config"]
    block_length = MODEL_STATE["block_length"]

    current_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    prompt_len = current_ids.shape[1]
    eos_id = tokenizer.eos_token_id or 156892
    mask_id = MODEL_STATE["mask_id"]
    total_generated = 0

    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    with set_current_vllm_config(vllm_config):
        with set_forward_context(None, vllm_config):
            while total_generated < max_tokens:
                outputs = dllm.generate(
                    current_ids,
                    gen_length=block_length,
                    block_length=block_length,
                )

                new_ids = outputs[0, current_ids.shape[1]:].cpu().tolist()
                clean = []
                hit_eos = False
                for tid in new_ids:
                    if tid == eos_id:
                        hit_eos = True
                        break
                    if tid != mask_id:
                        clean.append(tid)

                if clean:
                    text = tokenizer.decode(clean, skip_special_tokens=True)
                    total_generated += len(clean)
                    yield text, len(clean)

                if hit_eos or not clean:
                    break

                current_ids = outputs


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/v1/models")
async def list_models():
    model_path = MODEL_STATE.get("model_path", "unknown")
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": model_path,
            "object": "model",
            "owned_by": "dinfer",
        }],
    })


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    prompt = request.prompt
    if isinstance(prompt, list):
        prompt = prompt[0]

    req_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    model_name = MODEL_STATE.get("model_path", request.model)
    created = int(time.time())

    if request.stream:
        return await _stream_completion(
            prompt, request.max_tokens, request.temperature,
            req_id, model_name, created,
        )

    loop = asyncio.get_event_loop()
    text, completion_tokens, prompt_tokens = await loop.run_in_executor(
        executor,
        generate_sync,
        prompt,
        request.max_tokens,
        request.temperature,
    )

    return JSONResponse({
        "id": req_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "text": text,
            "logprobs": None,
            "finish_reason": "length",
            "stop_reason": None,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens": completion_tokens,
        },
    })


async def _stream_completion(
    prompt: str,
    max_tokens: int,
    temperature: float,
    req_id: str,
    model_name: str,
    created: int,
):
    loop = asyncio.get_event_loop()
    tokenizer = MODEL_STATE["tokenizer"]
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    async def event_generator():
        total_tokens = 0

        def _run_gen():
            return list(generate_block_by_block(prompt, max_tokens, temperature))

        blocks = await loop.run_in_executor(executor, _run_gen)

        for text, count in blocks:
            total_tokens += count
            data = {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"

        data = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "text": "",
                "logprobs": None,
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + total_tokens,
                "completion_tokens": total_tokens,
            },
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    init_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
