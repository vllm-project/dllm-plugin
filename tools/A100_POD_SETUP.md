# A100 Pod Setup Guide

This guide documents how to set up an A100 GPU pod for dLLM benchmarking.

## Prerequisites

- kubectl configured with access to A100 GPU cluster
- The `llada2-debug` pod running (see pod spec below)

## Quick Setup

Run the automated setup script:

```bash
./tools/setup_a100_pod.sh
```

This script will:
1. Copy dllm-plugin code to the pod
2. Install vLLM 0.6.6 and compatible dependencies
3. Install the dllm-plugin package
4. Start vLLM server with dLLM scheduler
5. Verify server health

## Manual Setup (if needed)

### 1. Create the Pod

```bash
kubectl apply -f tools/k8s/debug-pod-a100.yaml
```

Wait for pod to be running:
```bash
kubectl get pod llada2-debug -w
```

### 2. Copy Code

```bash
tar -czf - dllm_plugin pyproject.toml README.md | \
  kubectl exec -i llada2-debug -- tar -xzf - -C /workspace/dllm-plugin
```

### 3. Install Dependencies

```bash
kubectl exec llada2-debug -- bash -c "
  pip install vllm==0.6.6 --no-cache-dir
  pip install 'transformers<5.0' --no-cache-dir
  cd /workspace/dllm-plugin
  SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . --no-deps
"
```

### 4. Start vLLM Server

```bash
kubectl exec llada2-debug -- bash -c "
  export VLLM_PLUGINS=dllm
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  nohup python -m vllm.entrypoints.openai.api_server \
    --model inclusionAI/LLaDA2.0-mini \
    --max-model-len 2048 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
    > /tmp/vllm-server.log 2>&1 &
"
```

Wait 60 seconds for server to start.

### 5. Verify Server

```bash
kubectl exec llada2-debug -- python -c \
  "import requests; print(requests.get('http://localhost:8000/health').text)"
```

Expected output: `{"status":"ok"}`

## Port Forwarding

Forward port 8000 to access the server locally:

```bash
kubectl port-forward llada2-debug 8000:8000 > /tmp/port-forward.log 2>&1 &
```

Test locally:
```bash
curl http://localhost:8000/health
```

## Running Benchmarks

### Baseline Benchmark (torch.compile enabled by default)

```bash
tools/benchmark_optimization.sh baseline
```

### Disable torch.compile (for comparison)

```bash
# Restart server with torch.compile disabled
kubectl exec llada2-debug -- bash -c "
  pkill -f vllm
  export VLLM_PLUGINS=dllm
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  export VLLM_DLLM_DISABLE_COMPILE=true
  nohup python -m vllm.entrypoints.openai.api_server \
    --model inclusionAI/LLaDA2.0-mini \
    --max-model-len 2048 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
    > /tmp/vllm-server.log 2>&1 &
"

# Wait and benchmark
sleep 60
tools/benchmark_optimization.sh torch_compile_disabled
```

### Compare Results

```bash
python3 tools/extract_metrics.py \
  benchmarks/*/baseline.json \
  benchmarks/*/torch_compile_disabled.json
```

## Troubleshooting

### Check Server Logs

```bash
kubectl exec llada2-debug -- tail -100 /tmp/vllm-server.log
```

### Check torch.compile Status

Look for log message:
```
torch.compile enabled for routing on NVIDIA A100-SXM4-40GB (compute 8.0)
```

### Check GPU Memory

```bash
kubectl exec llada2-debug -- nvidia-smi
```

### Restart Server

```bash
kubectl exec llada2-debug -- pkill -f vllm
# Then re-run setup script or manual step 4
```

## Pod Lifecycle

The pod has a 2-hour TTL (`activeDeadlineSeconds: 7200`). After 2 hours, it will automatically terminate.

To extend the session, delete and recreate the pod:
```bash
kubectl delete pod llada2-debug
kubectl apply -f tools/k8s/debug-pod-a100.yaml
./tools/setup_a100_pod.sh
```

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_PLUGINS` | `dllm` | Enable dLLM plugin |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | Use V2 model runner |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Disable V1 multiprocessing |
| `VLLM_DLLM_DISABLE_COMPILE` | `true`/`false` | Control torch.compile on routing |

## Files on Pod

- `/workspace/dllm-plugin/` - Plugin source code
- `/tmp/vllm-server.log` - Server logs
- `/root/.cache/huggingface/` - Model cache (persists across restarts)
