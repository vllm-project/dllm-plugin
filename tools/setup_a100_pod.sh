#!/usr/bin/env bash
# Setup script for A100 pod with dLLM plugin
#
# This script:
# 1. Copies dllm-plugin code to the pod
# 2. Installs dependencies (vLLM 0.20.1, compatible transformers)
# 3. Installs the dllm-plugin
# 4. Starts vLLM server with dLLM scheduler
#
# Usage:
#   ./tools/setup_a100_pod.sh

set -euo pipefail

POD_NAME="${POD_NAME:-llada2-debug}"
MODEL="${MODEL:-inclusionAI/LLaDA2.0-mini}"

echo "=========================================="
echo "Setting up A100 pod: $POD_NAME"
echo "Model: $MODEL"
echo "=========================================="

# Step 1: Clean and prepare workspace
echo ""
echo "[1/5] Preparing workspace..."
kubectl exec $POD_NAME -- bash -c "rm -rf /workspace/dllm-plugin && mkdir -p /workspace/dllm-plugin"

# Step 2: Copy code
echo "[2/5] Copying dllm-plugin code..."
tar -czf - dllm_plugin pyproject.toml README.md 2>/dev/null | \
  kubectl exec -i $POD_NAME -- tar -xzf - -C /workspace/dllm-plugin

# Step 3: Install dependencies
echo "[3/5] Installing dependencies (vLLM 0.20.1, transformers<5.0)..."
kubectl exec $POD_NAME -- bash -c "
  pip install vllm==0.20.1 --no-cache-dir -q 2>&1 | tail -5 && \
  pip install 'transformers<5.0' --no-cache-dir -q 2>&1 | tail -5
"

# Step 4: Install dllm-plugin
echo "[4/5] Installing dllm-plugin..."
kubectl exec $POD_NAME -- bash -c \
  "cd /workspace/dllm-plugin && SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e . --no-deps -q"

# Step 5: Start vLLM server
echo "[5/5] Starting vLLM server..."
kubectl exec $POD_NAME -- bash -c "
  # Kill any existing vLLM processes
  pkill -9 -f 'python.*vllm' || true
  sleep 2

  # Start server with vllm serve CLI
  # NOTE: Plugin entry point name is 'dllm' (not 'dllm_plugin')
  export VLLM_PLUGINS=dllm
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  nohup vllm serve $MODEL \
    --max-model-len 2048 \
    --max-num-seqs 32 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
    --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker \
    > /tmp/vllm-server.log 2>&1 &
"

echo ""
echo "Waiting for server to start (60 seconds)..."
sleep 60

# Check if server is up
echo ""
echo "Checking server health..."
if kubectl exec $POD_NAME -- python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" 2>/dev/null; then
  echo "✓ Server is ready!"
else
  echo "⚠ Server health check failed. Check logs:"
  echo "  kubectl exec $POD_NAME -- tail -100 /tmp/vllm-server.log"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Port-forward: kubectl port-forward $POD_NAME 8000:8000"
echo "  2. Run benchmark: tools/benchmark_optimization.sh baseline"
echo "  3. Check logs: kubectl exec $POD_NAME -- tail -f /tmp/vllm-server.log"
echo "=========================================="
