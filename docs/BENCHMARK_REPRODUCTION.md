# Benchmark Reproduction Guide

Instructions for reproducing the Phase 9 E2E benchmarks on an A100 pod.

## Prerequisites

- Kubernetes cluster with A100-40GB GPUs
- `kubectl` configured
- HuggingFace access to `inclusionAI/LLaDA2.0-mini`

## Step 1: Create Pod

```bash
kubectl apply -f tools/investigation/pod-dinfer-benchmark.yaml
# Or use any A100 pod with:
#   image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
#   GPU: 1x A100-40GB
#   ephemeral-storage: 200Gi (for build + model weights)
kubectl wait --for=condition=Ready pod/<POD_NAME> --timeout=300s
```

## Step 2: Install Prerequisites

```bash
kubectl exec <POD_NAME> -- bash -c '
apt-get update -qq && apt-get install -y -qq git curl
pip install uv
'
```

## Step 3: Build vLLM Fork from Source

The plugin requires the
[dllm-fork](https://github.com/AlonKellner-RedHat/vllm/tree/dllm-fork)
of vLLM (4 commits on top of v0.20.2). Build it from source following
[docs.vllm.ai](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#full-build):

```bash
kubectl exec <POD_NAME> -- bash -c '
cd /workspace
git clone --branch dllm-fork https://github.com/AlonKellner-RedHat/vllm.git
cd vllm
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=6
uv pip install -e . --torch-backend=auto --system
'
```

This compiles all C++/CUDA extensions (~10-20 min on first build).

Verify:

```bash
kubectl exec <POD_NAME> -- python3 -c "import vllm; print(vllm.__version__)"
# Should print: 0.20.2.dev...
```

## Step 4: Install dllm-plugin

Copy and install the plugin (editable, no deps — vLLM already installed):

```bash
# From the repo root:
tar -czf - dllm_plugin pyproject.toml README.md | \
  kubectl exec -i <POD_NAME> -- tar -xzf - -C /workspace

kubectl exec <POD_NAME> -- bash -c '
cd /workspace
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . --no-deps
'
```

Verify:

```bash
kubectl exec <POD_NAME> -- python3 -c "import dllm_plugin; print('OK')"
```

## Step 5: Download Model

```bash
kubectl exec <POD_NAME> -- python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('inclusionAI/LLaDA2.0-mini', local_dir='/workspace/llada2-mini', resume_download=True)
"
```

## Step 6: Start vLLM Server

```bash
kubectl exec <POD_NAME> -- bash -c '
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TORCH_DYNAMO_DISABLE=1

vllm serve /workspace/llada2-mini \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker \
  > /workspace/server.log 2>&1 &
'
```

Wait for server health:

```bash
kubectl exec <POD_NAME> -- bash -c '
for i in $(seq 1 300); do
  curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo "Ready after ${i}s" && break
  sleep 1
done
'
```

Note: `--enforce-eager` is intentionally omitted to enable CUDA graph
capture (`UNIFORM_BATCH` mode).

## Step 7: Verify Coherent Output

```bash
kubectl exec <POD_NAME> -- curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/workspace/llada2-mini","prompt":"The capital of France is","max_tokens":64,"temperature":0}' \
  | python3 -m json.tool
```

Expected: coherent multi-sentence text (not garbage or repeated tokens).

## Step 8: Run Benchmarks

Install GuideLLM and run the 6-scenario suite:

```bash
kubectl exec <POD_NAME> -- pip install -q guidellm

# Sync free-form (baseline)
kubectl exec <POD_NAME> -- guidellm benchmark run \
  --target http://localhost:8000 \
  --model /workspace/llada2-mini \
  --request-format /v1/completions \
  --data "prompt_tokens=256,output_tokens=256,count=10000" \
  --backend-kwargs '{"extras":{"body":{"max_tokens":256}}}' \
  --profile synchronous \
  --max-seconds 300 \
  --output-path /workspace/benchmarks/sync_freeform.json
```

See `tools/run_phase9_benchmarks.sh` for all 6 scenarios.

## Step 9: Extract Metrics

```bash
kubectl exec <POD_NAME> -- python3 -c "
import json
with open('/workspace/benchmarks/sync_freeform.json') as f:
    data = json.load(f)
m = data['benchmarks'][0]['metrics']
ttft = m['time_to_first_token_ms']['successful']
itl = m['inter_token_latency_ms']['successful']
otps = m['output_tokens_per_second']['successful']
print(f'TTFT p50: {ttft[\"percentiles\"][\"p50\"]:.0f} ms')
print(f'ITL p50: {itl[\"percentiles\"][\"p50\"]:.1f} ms')
print(f'Output TPS: {otps[\"mean\"]:.1f}')
"
```

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_PLUGINS` | `dllm` | Load the dllm-plugin |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | Use MRV2 model runner |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Single-process mode |
| `TORCH_DYNAMO_DISABLE` | `1` | Disable torch.dynamo (incompatible) |

## Server Arguments

| Argument | Value | Purpose |
|----------|-------|---------|
| `--max-model-len` | `1024` | Max sequence length |
| `--max-num-seqs` | `8` | Max concurrent requests |
| `--gpu-memory-utilization` | `0.9` | GPU memory fraction |
| `--no-enable-prefix-caching` | - | Disabled (non-causal attention) |
| `--scheduler-cls` | `dllm_plugin.Scheduler` | Plugin scheduler |
| `--worker-cls` | `dllm_plugin.Worker` | Plugin worker |

Note: `--enforce-eager` is NOT used — CUDA graphs are enabled via
`UNIFORM_BATCH` mode (declared in `llada2_attention_backend.py`).

## Optimizations Enabled

1. **CUDA Graphs** (`UNIFORM_BATCH`): Model forward captured as graph,
   attention metadata runs eager
2. **Deferred GPU-CPU sync**: `.cpu().tolist()` deferred to step boundary
   (not every denoising iteration)
3. **Vectorized commit**: `scatter_()` replaces Python loop for
   `num_sampled`
4. **Triton remasking** (optional): Fused kernel auto-selected when
   available, falls back to PyTorch ops
