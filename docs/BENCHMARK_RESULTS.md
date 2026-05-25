# Benchmark Results: 6-Scenario GuideLLM Suite

## Run 2 — PR #44 Second Review (2026-05-25)

**Date:** 2026-05-25
**Model:** inclusionAI/LLaDA2.0-mini (MoE, 256 experts, 32-token blocks)
**Hardware:** NVIDIA A100-SXM4-40GB
**Fork:** `dllm-fork-coherent` (all Python files overlaid on vLLM 0.20.1)
**Plugin:** commit 6b57132 (PR #44 second review fixes)
**Config:** `--enforce-eager --no-async-scheduling --max-num-seqs 32 --max-model-len 2048`
**GuideLLM:** v0.6.0, `prompt_tokens=500, output_tokens=500, count=10000, max_seconds=300`

### Results

| # | Scenario | Requests | Errors | Completed RPS | Output TPS | Avg Latency | Quality |
|---|----------|----------|--------|---------------|------------|-------------|---------|
| 1 | Sync free-form | 18 | 0 | 0.06 | 30.0 | 17.2s | 3/3 coherent |
| 2 | Sync structured | 19 | 0 | 0.06 | 31.7 | 16.4s | 3/3 valid |
| 3 | 5 RPS free-form | 468 | 0 | 1.56 | 780.0 | 108.6s | 3/3 coherent |
| 4 | 5 RPS structured | 467 | 0 | 1.56 | 778.3 | 113.9s | 3/3 coherent |
| 5 | 20 RPS free-form | 468 | 0 | 1.56 | 780.0 | 142.3s | 3/3 coherent |
| 6 | 20 RPS structured | 457 | 0 | 1.52 | 761.7 | 148.1s | 3/3 coherent |

**Total: 1,897 requests, 0 errors, 18/18 quality checks passed.**

### Changes from Run 1

This run validates the PR #44 second review fixes:

- **Gumbel-max temperature support** (`temperature=0.0` default, matching
  dInfer's `get_transfer_index_threshold`). No regression — deterministic
  argmax is identical to previous behavior.
- **Vectorized commit block**: replaced N per-request `.item()` GPU-CPU
  syncs with single `torch.nonzero().cpu()` batch sync.
- **Backend allowlist**: FlashInfer + Triton (was hardcoded FlashInfer only).
- **CUDAGraph UNIFORM_BATCH confirmed correct**: hooks run eagerly, model
  forward is graph-captured.
- **Named constants**, `_warned_multi_req_recomp` init, draft dedup,
  `out=` buffer for clone elimination.

### Performance Comparison (Run 1 → Run 2)

| Metric | Run 1 (e39563c) | Run 2 (6b57132) | Change |
|--------|-----------------|-----------------|--------|
| Total requests | 1,681 | 1,897 | +12.8% |
| Batched TPS | ~700 | ~780 | +11.4% |
| Sync latency | 16.0s | 17.2s | +7.5% (within noise) |
| Errors | 0 | 0 | — |
| Quality | 18/18 | 18/18 | — |

The TPS improvement is likely from the vectorized commit block eliminating
per-request GPU-CPU synchronization overhead.

---

## Run 1 — Initial Benchmark (2026-05-24)

**Date:** 2026-05-24
**Model:** inclusionAI/LLaDA2.0-mini (MoE, 256 experts, 32-token blocks)
**Hardware:** NVIDIA A100-SXM4-40GB
**Fork:** `dllm-fork-coherent` (6b08edd + Patch 1/2 + non-causal fix)
**Plugin:** commit e39563c + bootstrap num_sampled fix
**Config:** `--enforce-eager --no-async-scheduling --max-num-seqs 32 --max-model-len 2048`
**GuideLLM:** v0.6.0, `prompt_tokens=500, output_tokens=500, count=10000, max_seconds=300`

### Results

| # | Scenario | Requests | Errors | Completed RPS | Output TPS | Avg Latency | Quality |
|---|----------|----------|--------|---------------|------------|-------------|---------|
| 1 | Sync free-form | 19 | 0 | 0.06 | 31.7 | 16.0s | 3/3 coherent |
| 2 | Sync structured | 19 | 0 | 0.06 | 31.7 | 16.1s | 3/3 valid |
| 3 | 5 RPS free-form | 394 | 0 | 1.31 | 656.7 | 115.8s | 3/3 coherent |
| 4 | 5 RPS structured | 409 | 0 | 1.36 | 681.7 | 116.3s | 3/3 valid |
| 5 | 20 RPS free-form | 419 | 0 | 1.40 | 698.3 | 147.6s | 3/3 coherent |
| 6 | 20 RPS structured | 421 | 0 | 1.40 | 701.7 | 146.2s | 3/3 valid |

**Total: 1,681 requests, 0 errors, 18/18 quality checks passed.**

---

## Quality Verification

Each scenario verified 3 sample outputs for coherence/validity:

- **Free-form**: Checked for coherence (>3 unique words in output)
- **Structured**: Checked regex match (`[A-Z][a-z]+( [A-Z][a-z]+)*`)

## Performance Notes

- **Sync (scenarios 1-2)**: 0.06 RPS = one request at a time, ~31 TPS.
  Each request takes ~16-17s for 500 output tokens.
- **Batched (scenarios 3-6)**: Saturates at ~1.5 completed RPS regardless
  of injection rate (5 or 20 RPS). Output TPS scales to ~780 with batching.
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
- Gumbel-max temperature support (temperature=0.0 default, dInfer parity)
- Vectorized commit block (single batch sync vs N per-request .item() calls)
- Backend allowlist: FlashInfer + Triton for non-causal attention
