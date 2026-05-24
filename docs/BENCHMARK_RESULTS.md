# Benchmark Results: 6-Scenario GuideLLM Suite

**Date:** 2026-05-24
**Model:** inclusionAI/LLaDA2.0-mini (MoE, 256 experts, 32-token blocks)
**Hardware:** NVIDIA A100-SXM4-40GB
**Fork:** `dllm-fork-coherent` (6b08edd + Patch 1/2 + non-causal fix)
**Plugin:** commit e39563c + bootstrap num_sampled fix
**Config:** `--enforce-eager --no-async-scheduling --max-num-seqs 32 --max-model-len 2048`
**GuideLLM:** v0.6.0, `prompt_tokens=500, output_tokens=500, count=10000, max_seconds=300`

## Results

| # | Scenario | Requests | Errors | Completed RPS | Output TPS | Avg Latency | Quality |
|---|----------|----------|--------|---------------|------------|-------------|---------|
| 1 | Sync free-form | 19 | 0 | 0.06 | 31.7 | 16.0s | 3/3 coherent |
| 2 | Sync structured | 19 | 0 | 0.06 | 31.7 | 16.1s | 3/3 valid |
| 3 | 5 RPS free-form | 394 | 0 | 1.31 | 656.7 | 115.8s | 3/3 coherent |
| 4 | 5 RPS structured | 409 | 0 | 1.36 | 681.7 | 116.3s | 3/3 valid |
| 5 | 20 RPS free-form | 419 | 0 | 1.40 | 698.3 | 147.6s | 3/3 coherent |
| 6 | 20 RPS structured | 421 | 0 | 1.40 | 701.7 | 146.2s | 3/3 valid |

**Total: 1,681 requests, 0 errors, 18/18 quality checks passed.**

## Quality Verification

Each scenario verified 3 prompts at temperature=0 before benchmarking:

- **Free-form**: Checked for coherence (>3 unique words in output)
- **Structured**: Checked regex match (`[A-Z][a-z]+( [A-Z][a-z]+)*`)

Prompts tested: "The quick brown fox", "Explain what gravity is",
"Write a haiku about rain".

## Performance Notes

- **Sync (scenarios 1-2)**: 0.06 RPS = one request at a time, 31.7 TPS.
  Each request takes ~16s for 500 output tokens (~32 TPS single-stream).
- **Batched (scenarios 3-6)**: Saturates at ~1.4 completed RPS regardless
  of injection rate (5 or 20 RPS). Output TPS scales to ~700 with batching.
  The server queues requests and processes them concurrently up to
  `max-num-seqs=32`.
- **Structured output** has no measurable overhead vs free-form at the
  same load level — the diffusion denoising loop dominates latency.

## Optimizations Enabled

- Fused Triton remasking kernel (2 kernels vs 7 PyTorch ops)
- GPU-resident per-request state (denoise_step, kv_refresh, prompt_len)
- Persistent input buffers for CUDA graph compatibility
- Non-causal FlashInfer attention (fork fix)
- Block-causal virtual batch attention decomposition
- First-block full recomputation with slot mapping remap
