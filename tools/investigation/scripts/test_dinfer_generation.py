#!/usr/bin/env python3
"""Phase A: Verify dInfer produces coherent text with LLaDA2.0-mini.

Uses BlockDiffusionLLM with ThresholdParallelDecoder.
Key: torch.no_grad() + torch._dynamo.config.suppress_errors to avoid
torch.compile recompilation on every dynamic shape.
"""

import os

import torch

# Disable torch.compile/dynamo to avoid hours-long recompilation
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12388"

model_path = "/workspace/llada2-mini"

from transformers import AutoConfig, AutoTokenizer
from vllm import distributed

distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
distributed.initialize_model_parallel(1, backend="nccl")

from dinfer import (
    BlockDiffusionLLM,
    BlockIteratorFactory,
    KVCacheFactory,
    ThresholdParallelDecoder,
)
from dinfer.model.modeling_llada2_moe import LLaDA2MoeModelLM
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# dInfer hardcodes mask_id=156895 across all LLaDA2 code
mask_id = 156895
eos_id = 156892

print("=== dInfer Generation Test ===")
print(f"tokenizer.pad_token_id = {tokenizer.pad_token_id}")
print(f"mask_id = {mask_id}, eos_id = {eos_id}")

parallel_config = ParallelConfig(
    pipeline_parallel_size=1,
    tensor_parallel_size=1,
    expert_parallel_size=1,
)

with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
    model = LLaDA2MoeModelLM(config=config).eval()
    model.load_weights(model_path, torch_dtype=torch.bfloat16, device="cuda")
    model = model.to("cuda")

decoder = ThresholdParallelDecoder(
    temperature=0,
    threshold=0.9,
    mask_id=mask_id,
    eos_id=eos_id,
)

gen_length = 32
prompts = [
    "The quick brown fox",
    "Once upon a time",
    "The future of AI",
    "In the beginning",
]

print(f"\n=== Generating {gen_length} tokens per prompt ===\n")

vllm_config = VllmConfig(parallel_config=parallel_config)
from vllm.forward_context import set_forward_context

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    prompt_len = input_ids.shape[1]
    total_len = ((prompt_len + gen_length + 31) // 32) * 32

    dllm = BlockDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True, max_length=total_len),
        early_stop=True,
    )

    with torch.no_grad(), set_forward_context(None, vllm_config):
        output_ids = dllm.generate(
            input_ids,
            gen_length=gen_length,
            block_length=32,
        )

    generated_ids = output_ids[0, prompt_len:].tolist()
    generated_ids = [t for t in generated_ids if t != mask_id and t != eos_id]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Prompt: {prompt!r}")
    print(f"Generated ({len(generated_ids)} tokens): {generated_text!r}")
    print()
    del dllm
    torch.cuda.empty_cache()

distributed.destroy_model_parallel()
distributed.destroy_distributed_environment()
print("=== Done ===")
