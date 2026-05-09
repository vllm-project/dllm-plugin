#!/bin/bash
# Complete TP=2 validation setup script
# Automates pod deployment, code copy, server start, and benchmark execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
POD_NAME="llada2-tp2-debug"

echo "=================================================="
echo "TP=2 Validation Setup Script"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Pod name: $POD_NAME"
echo ""

# Step 1: Deploy K8s pod
echo "=== Step 1: Deploy K8s Pod with 2x A100 GPUs ==="
kubectl apply -f "$SCRIPT_DIR/k8s/debug-pod-a100-tp2.yaml"

echo "Waiting for pod to be ready (timeout: 300s)..."
kubectl wait --for=condition=Ready pod/$POD_NAME --timeout=300s || {
    echo "ERROR: Pod failed to become ready"
    kubectl describe pod/$POD_NAME
    exit 1
}

echo "✓ Pod ready"
echo ""

# Step 2: Verify GPUs
echo "=== Step 2: Verify GPUs ==="
kubectl exec $POD_NAME -- nvidia-smi --list-gpus

GPU_COUNT=$(kubectl exec $POD_NAME -- nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -ne 2 ]; then
    echo "ERROR: Expected 2 GPUs, found $GPU_COUNT"
    exit 1
fi

echo "✓ 2 GPUs available"
echo ""

# Step 3: Copy code to pod
echo "=== Step 3: Copy Code to Pod ==="
cd "$PROJECT_ROOT"

# Create tarball excluding unnecessary files
tar -czf /tmp/dllm-plugin.tar.gz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='benchmarks' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  dllm_plugin/ tests/ pyproject.toml README.md

echo "Copying code to pod..."
kubectl cp /tmp/dllm-plugin.tar.gz $POD_NAME:/tmp/dllm-plugin.tar.gz

echo "Extracting code in pod..."
kubectl exec $POD_NAME -- bash -c "
  cd /workspace && \
  tar -xzf /tmp/dllm-plugin.tar.gz && \
  rm /tmp/dllm-plugin.tar.gz
"

rm /tmp/dllm-plugin.tar.gz

echo "✓ Code copied and extracted"
echo ""

# Step 4: Start vLLM server with TP=2
echo "=== Step 4: Start vLLM Server with TP=2 ==="
kubectl exec $POD_NAME -- bash -c "
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

echo "Server starting... (waiting 60s for initialization)"
sleep 60

# Check if server is running
echo "Checking server logs..."
kubectl exec $POD_NAME -- tail -30 /tmp/vllm-tp2.log | grep -q "Application startup complete" || {
    echo "WARNING: Server may not have started successfully"
    echo "Last 50 lines of server log:"
    kubectl exec $POD_NAME -- tail -50 /tmp/vllm-tp2.log
    echo ""
    echo "Check full logs with: kubectl exec $POD_NAME -- tail -100 /tmp/vllm-tp2.log"
    exit 1
}

echo "✓ Server started successfully"
echo ""

# Step 5: Verify server health
echo "=== Step 5: Verify Server Health ==="
kubectl exec $POD_NAME -- curl -sf http://localhost:8000/health > /dev/null || {
    echo "ERROR: Health endpoint not responding"
    kubectl exec $POD_NAME -- tail -50 /tmp/vllm-tp2.log
    exit 1
}

echo "✓ Server health check passed"
echo ""

# Step 6: Setup port forwarding
echo "=== Step 6: Setup Port Forwarding ==="
echo "Starting port-forward on 8000:8000..."

# Kill existing port-forward if any
pkill -f "kubectl port-forward $POD_NAME" 2>/dev/null || true

kubectl port-forward $POD_NAME 8000:8000 > /tmp/kubectl-port-forward.log 2>&1 &
PORT_FORWARD_PID=$!

sleep 3

# Verify port-forward is working
curl -sf http://localhost:8000/health > /dev/null || {
    echo "ERROR: Port forwarding not working"
    cat /tmp/kubectl-port-forward.log
    exit 1
}

echo "✓ Port forwarding active (PID: $PORT_FORWARD_PID)"
echo ""

# Step 7: Test local access
echo "=== Step 7: Test Local Access ==="
curl -s http://localhost:8000/v1/models | jq -r '.data[0].id'

echo "✓ Server accessible from localhost"
echo ""

echo "=================================================="
echo "TP=2 Setup Complete!"
echo "=================================================="
echo ""
echo "Server is running at: http://localhost:8000"
echo "Pod name: $POD_NAME"
echo "Port-forward PID: $PORT_FORWARD_PID"
echo ""
echo "Next steps:"
echo "  1. Run benchmarks: ./tools/run_tp2_benchmarks.sh"
echo "  2. View server logs: kubectl exec $POD_NAME -- tail -f /tmp/vllm-tp2.log"
echo "  3. Monitor GPUs: kubectl exec $POD_NAME -- nvidia-smi"
echo ""
echo "Cleanup when done:"
echo "  - Stop port-forward: kill $PORT_FORWARD_PID"
echo "  - Delete pod: kubectl delete pod $POD_NAME"
echo ""
echo "For complete validation guide, see docs/TP2_VALIDATION_GUIDE.md"
