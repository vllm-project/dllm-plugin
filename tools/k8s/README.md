# K8s Testing & Benchmarking for dLLM Plugin

Scripts for testing dLLM structured outputs on A100 GPUs.

## Benchmark Scripts

### benchmark-structured-outputs.py

Production-ready benchmark script using guidellm to test structured outputs performance.

**Usage:**
```bash
# Copy to pod
kubectl cp tools/k8s/benchmark-structured-outputs.py llada2-debug:/tmp/
kubectl cp tools/k8s/generate-simple-benchmark-data.py llada2-debug:/tmp/

# Generate test data
kubectl exec llada2-debug -- python /tmp/generate-simple-benchmark-data.py

# Run benchmarks
kubectl exec llada2-debug -- python /tmp/benchmark-structured-outputs.py
```

Results are saved to `/tmp/benchmark_*.json` on the pod.

See [docs/structured-outputs-benchmark-results.md](../../docs/structured-outputs-benchmark-results.md) for comprehensive benchmark findings.

### generate-simple-benchmark-data.py

Generates guidellm-compatible JSONL data for benchmarking.

**Configuration:**
- NUM_REQUESTS: 100 (default)
- OUTPUT_TOKENS: 500 (fits in 2048 context)
- TARGET_PROMPT_TOKENS: 500

## Pod Setup

### setup-pod-and-benchmark.sh

Idempotent setup script that:
1. Creates/checks A100 pod status
2. Copies dllm-plugin code
3. Installs dependencies
4. Starts vLLM server with dLLM scheduler/worker

**Usage:**
```bash
bash tools/k8s/setup-pod-and-benchmark.sh
```

### start-vllm-server.sh

Standalone script to start vLLM server with dLLM plugin.

**Usage:**
```bash
kubectl exec llada2-debug -- bash /tmp/start-vllm-server.sh
```

## Pod Configurations

### debug-pod-a100.yaml

Standard A100 debug pod for dLLM testing.

**Features:**
- NVIDIA A100-SXM4-40GB GPU
- 2 hour TTL (activeDeadlineSeconds)
- Pre-configured environment variables for dLLM

**Usage:**
```bash
kubectl apply -f tools/k8s/debug-pod-a100.yaml
kubectl wait --for=condition=ready pod/llada2-debug --timeout=300s
```

### debug-pod-a100-vllm.yaml

A100 pod with vLLM pre-installed image.

**Usage:**
```bash
kubectl apply -f tools/k8s/debug-pod-a100-vllm.yaml
```

## Files

| File | Purpose | Status |
|------|---------|--------|
| `benchmark-structured-outputs.py` | Production benchmark script | ✅ Active |
| `generate-simple-benchmark-data.py` | JSONL data generator | ✅ Active |
| `setup-pod-and-benchmark.sh` | Pod setup automation | ✅ Active |
| `start-vllm-server.sh` | Server startup | ✅ Active |
| `debug-pod-a100.yaml` | Pod config | ✅ Active |
| `debug-pod-a100-vllm.yaml` | Pod config (vLLM image) | ✅ Active |
| `README.md` | This file | ✅ Active |

---

**Note:** Old benchmark scripts (`benchmark-baseline.sh`, `benchmark-compare.sh`, `benchmark-comprehensive.sh`) have been superseded by `benchmark-structured-outputs.py`.
