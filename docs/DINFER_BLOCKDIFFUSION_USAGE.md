# dInfer BlockDiffusionLLM Usage Guide

## Critical Requirements

**IMPORTANT:** dInfer cannot be used with transformers' standard `model.generate()` API. It requires:

1. **vLLM Backend 0.10.2**: dInfer uses vLLM as its inference backend
2. **Distributed Initialization**: Must initialize vLLM distributed environment (even for single GPU)
3. **BlockDiffusionLLM Wrapper**: Custom generation wrapper that implements block diffusion semantics
4. **Parallel Config**: Must enable expert parallelism for MoE models (LLaDA2)
5. **transformers 4.57.6**: Required for LLaDA2 RoPE compatibility

---

## Complete Working Example

```python
#!/usr/bin/env python3
"""
dInfer E2E Generation using proper BlockDiffusionLLM wrapper.
Based on dInfer's test_bd.py pattern.
"""
import os
import torch
from transformers import AutoConfig, AutoTokenizer

# CRITICAL STEP 1: Initialize vLLM distributed environment
# This is REQUIRED even for single GPU usage
device = torch.device('cuda:0')
gpu_id = 0
torch.cuda.set_device(gpu_id)

from vllm import distributed
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12388'

# Initialize distributed (required by dInfer)
distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
distributed.initialize_model_parallel(1, backend='nccl')

# CRITICAL STEP 2: Load model with vLLM parallel config
from vllm.config import ParallelConfig, VllmConfig
from vllm.config import set_current_vllm_config, get_current_vllm_config
from vllm.forward_context import set_forward_context
from dinfer.model import LLaDA2MoeModelLM

# Setup parallel config (enable expert parallelism for MoE)
parallel_config = ParallelConfig(enable_expert_parallel=True)

with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
    # Load config and model
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = LLaDA2MoeModelLM(config=model_config).eval()
    model.load_weights(model_path, torch_dtype=torch.bfloat16)
    model = model.to(device)

# CRITICAL STEP 3: Create BlockDiffusionLLM wrapper
from dinfer import BlockDiffusionLLM, ThresholdParallelDecoder, BlockIteratorFactory, KVCacheFactory

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Create decoder (greedy sampling with temperature=0)
decoder = ThresholdParallelDecoder(
    temperature=0,
    threshold=0.9,
    mask_id=tokenizer.pad_token_id or 0,
    eos_id=tokenizer.eos_token_id
)

# Create BlockDiffusionLLM wrapper with KV cache
dllm = BlockDiffusionLLM(
    model,
    decoder,
    BlockIteratorFactory(use_block_diffusion=True),
    cache_factory=KVCacheFactory('prefix', is_bd_model=True),
    early_stop=True
)

# CRITICAL STEP 4: Generate with forward_context
input_ids = tokenizer.encode("Hello, how are you?", add_special_tokens=False, return_tensors="pt").to(device)

vllm_config = get_current_vllm_config()
with set_forward_context(None, vllm_config):
    outputs = dllm.generate(
        input_ids,
        gen_length=32,      # Number of tokens to generate
        block_length=32     # Block size for LLaDA2 (fixed at 32)
    )

# Extract generated tokens (outputs includes prompt)
generated_ids = outputs[0, input_ids.shape[1]:].cpu().tolist()
text = tokenizer.decode(generated_ids, skip_special_tokens=True)

# CRITICAL STEP 5: Cleanup distributed environment
distributed.destroy_model_parallel()
distributed.destroy_distributed_environment()
```

---

## Common Mistakes

### ❌ WRONG: Using transformers API directly

```python
# This will NOT work with dInfer!
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
outputs = model.generate(input_ids, max_new_tokens=32)
```

**Why it fails:** LLaDA2MoeModelLM is designed for dInfer's block diffusion framework, not transformers' standard generation API.

---

### ❌ WRONG: Skipping distributed initialization

```python
# This will fail with initialization errors!
model = LLaDA2MoeModelLM(config=model_config)
dllm = BlockDiffusionLLM(model, decoder, ...)
```

**Why it fails:** dInfer requires vLLM's distributed environment to be initialized, even for single GPU usage.

---

### ❌ WRONG: Missing forward_context

```python
# This will fail with context errors!
outputs = dllm.generate(input_ids, gen_length=32, block_length=32)
```

**Why it fails:** dInfer requires `set_forward_context()` wrapper around generation calls.

---

### ❌ WRONG: Using wrong transformers version

```python
# Using transformers 5.x will fail with RoPE errors!
# AttributeError: 'LLaDA2MoeRotaryEmbedding' object has no attribute 'rope_type'
```

**Why it fails:** LLaDA2 model code was written for transformers 4.57.6 RoPE API.

**Fix:** `pip install transformers==4.57.6`

---

## BlockDiffusionLLM Components

### 1. ThresholdParallelDecoder

Controls how tokens are decoded from logits during generation.

```python
decoder = ThresholdParallelDecoder(
    temperature=0,      # 0 = greedy (deterministic), >0 = sampling
    threshold=0.9,      # Confidence threshold for parallel decoding
    mask_id=156895,     # Token ID to use for masking
    eos_id=156892       # End-of-sequence token ID
)
```

**For deterministic generation:** Use `temperature=0`

**For sampling:** Use `temperature > 0` (e.g., 0.7, 1.0)

---

### 2. BlockIteratorFactory

Controls block diffusion iteration strategy.

```python
BlockIteratorFactory(
    use_block_diffusion=True,      # Enable block diffusion semantics
    start_block_align=False         # Align starting position to block boundary
)
```

**For LLaDA2:** Always use `use_block_diffusion=True`

**For batch inference with varying prompt lengths:** Use `start_block_align=True`

---

### 3. KVCacheFactory

Manages key-value cache for attention.

```python
KVCacheFactory(
    'prefix',              # Cache type: 'prefix' or 'full'
    is_bd_model=True       # Block diffusion model flag
)
```

**Recommended:** Use `'prefix'` cache with `is_bd_model=True` for LLaDA2

---

## Generation Parameters

### dllm.generate()

```python
outputs = dllm.generate(
    input_ids,              # [batch_size, seq_len] tensor
    gen_length=256,         # Number of NEW tokens to generate
    block_length=32         # Block size (32 for LLaDA2)
)
```

**Important:**
- `gen_length`: Total NEW tokens to generate (excluding prompt)
- `block_length`: Fixed at 32 for LLaDA2 models
- `outputs`: Includes both prompt and generated tokens (shape: [batch_size, prompt_len + gen_length])

---

## Memory Optimization

If you encounter OOM (Out-of-Memory) errors:

### 1. Reduce generation length

```python
# Instead of generating all at once:
outputs = dllm.generate(input_ids, gen_length=256, block_length=32)

# Generate in chunks:
for chunk in range(8):  # 8 chunks of 32 tokens = 256 total
    outputs = dllm.generate(current_ids, gen_length=32, block_length=32)
    current_ids = outputs  # Use output as next input
```

### 2. Use float16 instead of bfloat16

```python
# Load model with float16 (less memory)
model.load_weights(model_path, torch_dtype=torch.float16)
```

### 3. Disable KV cache for testing

```python
# Create BlockDiffusionLLM without cache
dllm = BlockDiffusionLLM(
    model,
    decoder,
    BlockIteratorFactory(use_block_diffusion=True),
    cache_factory=None,  # Disable cache
    early_stop=True
)
```

---

## Batch Inference

For batch inference with varying prompt lengths:

```python
# Pad input_ids to max length with mask_id
batch_input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
batch_input_ids.fill_(mask_id)

for i, input_ids in enumerate(batch_inputs):
    batch_input_ids[i, :len(input_ids)] = input_ids

# Use start_block_align for batch
dllm = BlockDiffusionLLM(
    model,
    decoder,
    BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
    cache_factory=KVCacheFactory('prefix', is_bd_model=True),
    early_stop=True
)

outputs = dllm.generate(batch_input_ids, gen_length=256, block_length=32)
```

---

## Reference Implementation

See dInfer's official test: `/tmp/dInfer/tests/test_bd.py` (lines 61-86)

Key pattern from dInfer repo:
```python
def run_bd(use_kvcache):
    dllm = BlockDiffusionLLM(
        model, 
        decoder, 
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory('prefix', is_bd_model=True) if use_kvcache else None,
        early_stop=True
    )
    
    vllm_config = get_current_vllm_config()
    with set_forward_context(None, vllm_config):
        out = dllm.generate(input_ids, gen_length=256, block_length=32)
    
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
```

---

## Troubleshooting

### Error: `KeyError: 'default'` in ROPE_INIT_FUNCTIONS

**Cause:** Using transformers 5.x instead of 4.57.6

**Fix:**
```bash
pip install transformers==4.57.6
```

### Error: `Model architectures ['LLaDA2MoeModelLM'] are not supported`

**Cause:** Trying to use LLaDA2MoeModelLM with transformers `AutoModelForCausalLM`

**Fix:** Use dInfer's `LLaDA2MoeModelLM` directly (not through transformers)

### Error: `RuntimeError: distributed environment not initialized`

**Cause:** Skipped distributed initialization step

**Fix:** Call `distributed.init_distributed_environment()` before loading model

### Error: Exit 137 (OOM)

**Cause:** Out of memory during generation

**Fix:** Reduce `gen_length`, use float16, or disable KV cache (see Memory Optimization section)

---

## Version Requirements

| Package | Version | Required For |
|---------|---------|--------------|
| transformers | 4.57.6 | LLaDA2 RoPE compatibility |
| vllm | 0.10.2 | dInfer backend |
| dinfer | latest | Block diffusion framework |
| torch | ≥2.0 | CUDA and distributed support |

**Install all at once:**
```bash
pip install transformers==4.57.6 vllm==0.10.2
git clone https://github.com/inclusionAI/dInfer.git
cd dInfer && pip install .
```

---

## Summary: 5-Step dInfer Pattern

1. **Initialize distributed environment** (vLLM backend)
2. **Load model** with `LLaDA2MoeModelLM` and parallel config
3. **Create BlockDiffusionLLM wrapper** with decoder and cache factory
4. **Generate** with `set_forward_context()` wrapper
5. **Cleanup** distributed environment

**Never** use transformers' `model.generate()` - always use `BlockDiffusionLLM.generate()`
