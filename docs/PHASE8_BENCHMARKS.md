# Phase 8 Benchmark Results: vLLM-Native torch.compile

**Date:** 2026-05-06  
**Branch:** feat/phase7-llada2-real-model  
**Commit:** [6bdb2c1](https://github.com/vllm-project/dllm-plugin/commit/6bdb2c1)

## Overview

Phase 8 implements official vLLM torch.compile integration via the `@support_torch_compile` decorator. This replaces the manual `torch.compile()` approach with vLLM's native compilation system for better integration and automatic graph optimization.

## Test Configuration

- **Model:** inclusionAI/LLaDA2.0-mini
- **Hardware:** NVIDIA A100-SXM4-40GB (compute capability 8.0)
- **vLLM Version:** 0.20.1
- **Environment:**
  - `VLLM_PLUGINS=dllm`
  - `VLLM_USE_V2_MODEL_RUNNER=1`
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `VLLM_DLLM_STRICT_STACK_VALIDATION=false`

## Benchmark Methodology

**Tool:** GuideLLM 0.6.0  
**Profile:** Synchronous (batch size 1)  
**Duration:** 180 seconds  
**Workload:**
- Input tokens: 256
- Output tokens: 1000

## Performance Results

### Throughput Metrics

| Metric | Median | Mean | p95 |
|--------|--------|------|-----|
| **Output Tokens/sec** | **345.5** | **350.7** | - |
| **Total Tokens/sec** | - | **440.8** | - |
| **Requests/sec** | - | **0.3** | - |

### Latency Metrics

| Metric | Median | Mean | p95 |
|--------|--------|------|-----|
| **Request Latency** | 2.9 s | - | 3.0 s |
| **TTFT (Time To First Token)** | 522.1 ms | - | 528.4 ms |
| **ITL (Inter-Token Latency)** | 2.4 ms | - | 2.4 ms |
| **TPOT (Time Per Output Token)** | 2.9 ms | - | 3.0 ms |

### Test Summary

- **Completed Requests:** 63 (100% success rate)
- **Total Output Tokens:** 63,000
- **Total Input Tokens:** 17,262
- **Concurrency:** 1.0 (synchronous)
- **torch.compile overhead:** 3.25s (one-time compilation)

## Implementation Changes

### Key Changes from Manual torch.compile

1. **Decorator-Based Opt-In** ([llada2.py:463](../dllm_plugin/models/llada2.py#L463))
   ```python
   @support_torch_compile(
       dynamic_arg_dims={"input_ids": 0, "positions": 0},
   )
   class LLaDA2ForCausalLM(nn.Module):
       ...
   ```

2. **Removed Manual Compilation**
   - Deleted manual `torch.compile()` calls on routing methods
   - Simplified to GPU capability logging only
   - vLLM's compilation system handles optimization automatically

3. **API Compatibility** ([validation.py:124](../dllm_plugin/validation.py#L124))
   - Updated for vLLM 0.6.x/0.20+ API
   - Supports both `get_scheduler_cls()` method and `scheduler_cls` attribute

### Files Modified

- [`dllm_plugin/models/llada2.py`](../dllm_plugin/models/llada2.py): Added `@support_torch_compile` decorator
- [`dllm_plugin/validation.py`](../dllm_plugin/validation.py): vLLM 0.20+ API compatibility
- [`.gitignore`](../.gitignore): Added benchmark results to ignore list

## Absolute Performance (vLLM 0.20.1)

**Phase 8 with vLLM-native torch.compile:**
- **Throughput:** ~346 tokens/sec (median output tokens/sec)
- **Hardware:** A100-SXM4-40GB
- **vLLM Version:** 0.20.1

**Note:** This PR upgrades from vLLM 0.6.x to 0.20.1. Cross-version performance comparisons are not provided as vLLM 0.20.1 includes numerous optimizations unrelated to torch.compile. For proper performance comparison, future work should benchmark with and without torch.compile on the same vLLM version.

## Server Logs

Key log messages confirming torch.compile activation:

```
INFO [backends.py:1069] Using cache directory: /root/.cache/vllm/torch_compile_cache/...
INFO [decorators.py:668] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/...
INFO [monitor.py:53] torch.compile took 3.25 s in total
INFO     Application startup complete.
```

## Next Steps (Future Optimizations)

Phase 8 establishes the baseline with vLLM-native torch.compile. Future optimizations outlined in the [Phase 8 plan](../docs/PHASE8_PLAN.md) include:

1. **Single-Pass Attention** (Phase 3): Target +10-20% TTFT improvement
2. **CUTLASS FusedMoE** (Phase 4): Target +15-30% TPS improvement on A100
3. **FlashInfer Fused TopK** (Phase 5): Target +20-40% TPS on H100+ (requires compute capability 9.0)

## References

- **GuideLLM Documentation:** https://github.com/neuralmagic/guidellm
- **vLLM torch.compile Guide:** https://docs.vllm.ai/en/latest/design/torch_compile/
- **support_torch_compile API:** https://docs.vllm.ai/en/latest/api/vllm/compilation/decorators.html
- **Full benchmark results:** Not tracked in git (see `benchmarks/` directory locally)

## Reproducibility

To reproduce these benchmarks:

1. **Setup A100 Pod:**
   ```bash
   kubectl apply -f tools/k8s/debug-pod-a100.yaml
   ./tools/setup_a100_pod.sh
   ```

2. **Run Benchmark:**
   ```bash
   kubectl port-forward llada2-debug 8000:8000 &
   
   guidellm benchmark \
     --target http://localhost:8000 \
     --model inclusionAI/LLaDA2.0-mini \
     --profile synchronous \
     --max-seconds 180 \
     --data "prompt_tokens=256,output_tokens=1000" \
     --output-path results.json
   ```

3. **Check Server Logs:**
   ```bash
   kubectl exec llada2-debug -- tail -100 /tmp/vllm-server.log | grep -E 'torch.compile|Application startup'
   ```

See [`tools/A100_POD_SETUP.md`](../tools/A100_POD_SETUP.md) for detailed setup instructions.
