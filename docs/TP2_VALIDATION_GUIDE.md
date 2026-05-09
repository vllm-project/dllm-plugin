# TP=2 Validation Guide

**Status:** In Progress  
**Date:** 2026-05-09  
**Goal:** Validate Tensor Parallelism (TP=2) implementation and benchmark performance scaling

---

## Overview

This guide covers the complete TP=2 validation workflow including:
1. Setting up dual A100-40GB GPU environment
2. Starting vLLM server with TP=2
3. Running comprehensive benchmarks
4. Comparing TP=2 vs TP=1 performance

## Prerequisites

- Kubernetes cluster with A100-40GB GPU nodes
- `kubectl` configured and authenticated
- `guidellm` benchmark tool (installed via `uv pip install guidellm==0.6.0`)
- `dllm-plugin` repository cloned locally

## Quick Start (Automated Setup)

For automated setup and validation, use the provided scripts:

```bash
# Complete setup: pod deployment, code copy, server start
./tools/setup_tp2_validation.sh

# Run all 6 benchmark scenarios
./tools/run_tp2_benchmarks.sh

# Cleanup when done
./tools/cleanup_tp2_validation.sh
```

**Scripts included:**
- [`tools/setup_tp2_validation.sh`](../tools/setup_tp2_validation.sh) - Automated complete setup
- [`tools/run_tp2_benchmarks.sh`](../tools/run_tp2_benchmarks.sh) - Run all 6 benchmark configurations
- [`tools/cleanup_tp2_validation.sh`](../tools/cleanup_tp2_validation.sh) - Cleanup environment
- [`tools/k8s/debug-pod-a100-tp2.yaml`](../tools/k8s/debug-pod-a100-tp2.yaml) - Pod manifest

**For manual step-by-step setup, continue to the sections below.**

---

## Step 1: Deploy K8s Pod with 2x A100 GPUs

### Create and Deploy Pod

```bash
# Deploy the debug pod manifest
kubectl apply -f tools/k8s/debug-pod-a100-tp2.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/llada2-tp2-debug --timeout=300s

# Verify 2 GPUs are available
kubectl exec llada2-tp2-debug -- nvidia-smi --list-gpus
```

**Expected output:**
```
GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-...)
GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-...)
```

### Pod Configuration

The pod manifest ([tools/k8s/debug-pod-a100-tp2.yaml](../tools/k8s/debug-pod-a100-tp2.yaml)) includes:
- 2x A100-40GB GPUs
- 128Gi memory
- 80Gi ephemeral storage
- 16 CPU cores
- 2-hour TTL (7200s)
- Pre-configured environment variables for dLLM

---

## Step 2: Copy Code to Pod

```bash
# Create temporary archive of plugin code
tar -czf /tmp/dllm-plugin.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='benchmarks' \
  dllm_plugin/ tests/ pyproject.toml

# Copy to pod
kubectl cp /tmp/dllm-plugin.tar.gz llada2-tp2-debug:/tmp/dllm-plugin.tar.gz

# Extract in pod
kubectl exec llada2-tp2-debug -- bash -c "
  cd /workspace && \
  tar -xzf /tmp/dllm-plugin.tar.gz && \
  rm /tmp/dllm-plugin.tar.gz
"

# Verify code copied
kubectl exec llada2-tp2-debug -- ls -la /workspace/dllm_plugin
```

---

## Step 3: Start vLLM Server with TP=2

### Start Server

```bash
# Start vLLM server with TP=2 configuration
kubectl exec llada2-tp2-debug -- bash -c "
cd /workspace && \
nohup uv run python -m vllm.entrypoints.openai.api_server \
  --model inclusionAI/LLaDA2.0-mini \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 2 \
  --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
  --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker \
  > /tmp/vllm-tp2.log 2>&1 &
"
```

### Verify Server Started

```bash
# Wait for server to initialize (typically 30-60 seconds)
sleep 60

# Check server logs
kubectl exec llada2-tp2-debug -- tail -50 /tmp/vllm-tp2.log

# Look for these success indicators:
# - "world_size=2 rank=0 local_rank=0" (TP rank 0)
# - "world_size=2 rank=1 local_rank=1" (TP rank 1)
# - "Loading model from scratch..." (both ranks)
# - "Model loading took X GiB" (both ranks)
# - "Application startup complete."

# Test health endpoint
kubectl exec llada2-tp2-debug -- curl -s http://localhost:8000/health

# Verify model loaded
kubectl exec llada2-tp2-debug -- curl -s http://localhost:8000/v1/models | jq .
```

**Expected model response:**
```json
{
  "data": [
    {
      "id": "inclusionAI/LLaDA2.0-mini",
      "object": "model",
      "max_model_len": 2048
    }
  ]
}
```

### Port Forwarding (for Local Access)

```bash
# Forward port 8000 to localhost
kubectl port-forward llada2-tp2-debug 8000:8000 &

# Test from local machine
curl -s http://localhost:8000/health
```

---

## Step 4: Run Benchmark Suite

### Install GuideLLM (if not already installed)

```bash
uv pip install guidellm==0.6.0
```

### Run All 6 Benchmark Configurations

```bash
# Execute the comprehensive benchmark suite
./tools/run_tp2_benchmarks.sh
```

This runs 6 scenarios (total ~30 minutes):
1. **Synchronous + Free-form** - Sequential processing, no constraints
2. **Synchronous + Structured** - Sequential processing, regex-constrained
3. **Constant 5 RPS + Free-form** - Moderate load
4. **Constant 5 RPS + Structured** - Moderate load, regex-constrained
5. **Constant 10 RPS + Free-form** - High load (near saturation)
6. **Constant 10 RPS + Structured** - High load, regex-constrained

**Benchmark Parameters:**
- Input tokens: 500
- Output tokens: 500
- Max requests: 10,000 per scenario
- Max duration: 300 seconds per scenario
- Regex pattern: `[A-Z][a-z]+( [A-Z][a-z]+)*`

**Results Location:**
```
benchmarks/tp2_validation/
├── 1_sync_freeform.json
├── 2_sync_structured.json
├── 3_5rps_freeform.json
├── 4_5rps_structured.json
├── 5_10rps_freeform.json
└── 6_10rps_structured.json
```

### Manual Benchmark Execution (Individual Scenarios)

If you need to run scenarios individually:

```bash
# Scenario 1: Synchronous + Free-form
uv run guidellm benchmark run \
  --target http://localhost:8000 \
  --model inclusionAI/LLaDA2.0-mini \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile synchronous \
  --max-requests 10000 \
  --max-seconds 300 \
  --output-path benchmarks/tp2_validation/1_sync_freeform.json

# Scenario 2: Synchronous + Structured
uv run guidellm benchmark run \
  --target http://localhost:8000 \
  --model inclusionAI/LLaDA2.0-mini \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile synchronous \
  --max-requests 10000 \
  --max-seconds 300 \
  --backend-kwargs '{"extras": {"body": {"guided_regex": "[A-Z][a-z]+( [A-Z][a-z]+)*"}}}' \
  --output-path benchmarks/tp2_validation/2_sync_structured.json

# Scenario 3: Constant 5 RPS + Free-form
uv run guidellm benchmark run \
  --target http://localhost:8000 \
  --model inclusionAI/LLaDA2.0-mini \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile constant \
  --rate 5 \
  --max-requests 10000 \
  --max-seconds 300 \
  --output-path benchmarks/tp2_validation/3_5rps_freeform.json

# Scenario 4: Constant 5 RPS + Structured
uv run guidellm benchmark run \
  --target http://localhost:8000 \
  --model inclusionAI/LLaDA2.0-mini \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile constant \
  --rate 5 \
  --max-requests 10000 \
  --max-seconds 300 \
  --backend-kwargs '{"extras": {"body": {"guided_regex": "[A-Z][a-z]+( [A-Z][a-z]+)*"}}}' \
  --output-path benchmarks/tp2_validation/4_5rps_structured.json

# Scenario 5: Constant 10 RPS + Free-form
uv run guidellm benchmark run \
  --target http://localhost:8000 \
  --model inclusionAI/LLaDA2.0-mini \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile constant \
  --rate 10 \
  --max-requests 10000 \
  --max-seconds 300 \
  --output-path benchmarks/tp2_validation/5_10rps_freeform.json

# Scenario 6: Constant 10 RPS + Structured
uv run guidellm benchmark run \
  --target http://localhost:8000 \
  --model inclusionAI/LLaDA2.0-mini \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile constant \
  --rate 10 \
  --max-requests 10000 \
  --max-seconds 300 \
  --backend-kwargs '{"extras": {"body": {"guided_regex": "[A-Z][a-z]+( [A-Z][a-z]+)*"}}}' \
  --output-path benchmarks/tp2_validation/6_10rps_structured.json
```

---

## Step 5: Analyze Results

### Extract Key Metrics

```bash
# Extract throughput, latency, and TTFT from each result
for file in benchmarks/tp2_validation/*.json; do
  echo "=== $(basename $file) ==="
  jq '.results.summary | {
    requests: .completed_requests,
    throughput_rps: .request_throughput,
    throughput_tps: .token_throughput,
    latency_median: .latency_percentiles.p50,
    latency_p95: .latency_percentiles.p95,
    ttft_median: .ttft_percentiles.p50,
    itl_median: .itl_percentiles.p50
  }' "$file"
  echo ""
done
```

### Compare Against TP=1 Baseline

Compare against the TP=1 results from [BENCHMARK_RESULTS_MULTI_REQUEST.md](BENCHMARK_RESULTS_MULTI_REQUEST.md):

| Metric | TP=1 (Sync) | TP=2 (Sync) | Scaling |
|--------|-------------|-------------|---------|
| Throughput (tokens/s) | 501.3 | TBD | TBD |
| Latency Median (ms) | 2,000 | TBD | TBD |
| TTFT Median (ms) | 17.2 | TBD | TBD |
| ITL Median (ms) | 4.0 | TBD | TBD |

**Expected TP=2 scaling:**
- Throughput: ~1.8-2.0x improvement
- Latency: Similar or slightly lower
- TTFT: Similar (prefill phase scaling depends on sequence length)
- ITL: Similar or slightly lower (decode phase benefits from TP)

---

## Troubleshooting

### Server Fails to Start

**Check GPU availability:**
```bash
kubectl exec llada2-tp2-debug -- nvidia-smi
```

**Check server logs for errors:**
```bash
kubectl exec llada2-tp2-debug -- tail -100 /tmp/vllm-tp2.log | grep -i "error\|exception\|failed"
```

**Common issues:**
1. **OOM (Out of Memory):** Reduce `--gpu-memory-utilization` to 0.75
2. **TP rank mismatch:** Verify both GPUs detected with `nvidia-smi`
3. **Module import error:** Ensure `dllm_plugin/` copied to `/workspace/`

### Benchmark Fails

**Check server is responsive:**
```bash
curl -v http://localhost:8000/health
```

**Test a simple completion:**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-mini",
    "prompt": "Hello world",
    "max_tokens": 50
  }'
```

**Check guidellm version:**
```bash
uv run guidellm --version
# Expected: 0.6.0
```

### Port Forward Dies

**Restart port forward:**
```bash
pkill -f "kubectl port-forward"
kubectl port-forward llada2-tp2-debug 8000:8000 &
```

---

## Cleanup

### Stop Server

```bash
kubectl exec llada2-tp2-debug -- pkill -f vllm.entrypoints.openai.api_server
```

### Delete Pod

```bash
kubectl delete pod llada2-tp2-debug
```

### Remove Local Port Forward

```bash
pkill -f "kubectl port-forward llada2-tp2-debug"
```

---

## Success Criteria

- [x] Server starts successfully with TP=2
- [x] Both GPU ranks (0 and 1) initialize and load model
- [x] Health endpoint responds
- [ ] All 6 benchmark scenarios complete without errors
- [ ] TP=2 throughput shows 1.8-2.0x improvement over TP=1
- [ ] Latency metrics remain within 10% of TP=1 baseline
- [ ] Structured outputs work correctly with TP=2

---

## Next Steps

After successful validation:
1. Document results in `docs/TP2_BENCHMARK_RESULTS.md`
2. Update `docs/KNOWN_LIMITATIONS.md` to mark TP=2 as validated
3. Update README.md to highlight TP=2 support
4. Commit TP implementation changes to PR #38
5. Add TP integration tests to test suite

---

## References

- [BENCHMARK_RESULTS_MULTI_REQUEST.md](BENCHMARK_RESULTS_MULTI_REQUEST.md) - TP=1 baseline results
- [tools/k8s/debug-pod-a100-tp2.yaml](../tools/k8s/debug-pod-a100-tp2.yaml) - Pod manifest
- [tools/run_tp2_benchmarks.sh](../tools/run_tp2_benchmarks.sh) - Benchmark script
- [Plan: Implement TP-Aware Per-Expert Weight Loading](../.claude/plans/let-s-plan-phase-7-agile-mochi.md)
