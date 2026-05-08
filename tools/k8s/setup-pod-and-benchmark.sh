#!/bin/bash
# Idempotent pod setup script for dLLM benchmarking
# This script can be run multiple times safely
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
POD_NAME="llada2-debug"

echo "=================================================="
echo "dLLM Plugin Benchmark Setup"
echo "=================================================="
echo "Pod: $POD_NAME"
echo "Image: vllm/vllm-openai:latest (v0.20.1)"
echo ""

# Function to check if pod exists
pod_exists() {
    kubectl get pod "$POD_NAME" &>/dev/null
}

# Function to check if pod is ready
pod_ready() {
    kubectl get pod "$POD_NAME" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null | grep -q "True"
}

# Step 1: Ensure pod exists and is ready
echo "[1/6] Checking pod status..."
if ! pod_exists; then
    echo "  → Creating pod..."
    kubectl apply -f "$SCRIPT_DIR/debug-pod-a100-vllm.yaml"
    echo "  → Waiting for pod to be ready (timeout: 5 minutes)..."
    kubectl wait --for=condition=ready pod/"$POD_NAME" --timeout=300s
    echo "  ✓ Pod created and ready"
elif ! pod_ready; then
    echo "  → Pod exists but not ready, waiting..."
    kubectl wait --for=condition=ready pod/"$POD_NAME" --timeout=300s
    echo "  ✓ Pod ready"
else
    echo "  ✓ Pod already running and ready"
fi
echo ""

# Step 2: Copy dllm-plugin code
echo "[2/6] Copying dllm-plugin code..."
kubectl exec "$POD_NAME" -- mkdir -p /workspace/dllm-plugin 2>/dev/null || true
kubectl cp "$REPO_ROOT/dllm_plugin" "$POD_NAME:/workspace/dllm-plugin/dllm_plugin/"
kubectl cp "$REPO_ROOT/pyproject.toml" "$POD_NAME:/workspace/dllm-plugin/pyproject.toml"
echo "  ✓ Code copied"
echo ""

# Step 3: Install dllm-plugin (vLLM already in image)
echo "[3/6] Installing dllm-plugin..."
kubectl exec "$POD_NAME" -- bash -c "
    cd /workspace/dllm-plugin
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_DLLM_PLUGIN='0.1.0.dev'
    pip install -e . -q
" 2>&1 | grep -v "WARNING: Running pip as the 'root' user" | grep -v "It is recommended to use a virtual environment" || true
echo "  ✓ dllm-plugin installed"
echo ""

# Step 4: Install guidellm (with numpy constraint for vLLM compatibility)
echo "[4/6] Installing guidellm..."
kubectl exec "$POD_NAME" -- bash -c "
    if ! pip list | grep -q guidellm; then
        pip install 'numpy<2.0.0' -q
        pip install guidellm -q
    fi
" 2>&1 | grep -v "WARNING: Running pip as the 'root' user" | grep -v "It is recommended to use a virtual environment" | grep -v "dependency conflicts" || true
echo "  ✓ guidellm installed"
echo ""

# Step 5: Copy benchmark scripts
echo "[5/6] Copying benchmark scripts..."
kubectl cp "$SCRIPT_DIR/benchmark-baseline.sh" "$POD_NAME:/workspace/benchmark-baseline.sh"
kubectl cp "$SCRIPT_DIR/benchmark-structured-outputs.sh" "$POD_NAME:/workspace/benchmark-structured-outputs.sh"
kubectl cp "$SCRIPT_DIR/generate-structured-output-data.py" "$POD_NAME:/workspace/generate-structured-output-data.py"
kubectl exec "$POD_NAME" -- chmod +x /workspace/benchmark-baseline.sh
kubectl exec "$POD_NAME" -- chmod +x /workspace/benchmark-structured-outputs.sh
echo "  ✓ Benchmark scripts copied"
echo ""

# Step 6: Start vLLM server
echo "[6/6] Starting vLLM server..."

# Kill any existing vLLM processes (ignore errors)
kubectl exec "$POD_NAME" -- pkill -9 -f vllm 2>/dev/null || true
kubectl exec "$POD_NAME" -- nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2

# Create startup script in pod
kubectl exec "$POD_NAME" -- bash -c 'cat > /tmp/start-vllm.sh << '\''SCRIPT'\''
#!/bin/bash
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model inclusionAI/LLaDA2.0-mini \
    --max-model-len 2048 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
    --worker-cls dllm_plugin.Worker \
    > /tmp/vllm-server.log 2>&1 &
echo $! > /tmp/vllm-server.pid
SCRIPT'

# Make it executable and run it
kubectl exec "$POD_NAME" -- chmod +x /tmp/start-vllm.sh
kubectl exec "$POD_NAME" -- /tmp/start-vllm.sh

echo "  → Waiting for server to start (checking every 10s, max 180s)..."
for i in {1..18}; do
    if kubectl exec "$POD_NAME" -- python3 -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" 2>/dev/null; then
        echo "  ✓ vLLM server started successfully"
        break
    fi
    if [ $i -eq 18 ]; then
        echo "  ✗ Server failed to start within 180 seconds"
        echo ""
        echo "Check logs with:"
        echo "  kubectl exec $POD_NAME -- tail -100 /tmp/vllm-server.log"
        exit 1
    fi
    sleep 10
done
echo ""

echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Server is running at http://localhost:8000 (inside pod)"
echo ""
echo "To run benchmarks:"
echo "  # Baseline (no structured outputs)"
echo "  kubectl exec $POD_NAME -- bash /workspace/benchmark-baseline.sh"
echo ""
echo "  # Structured outputs"
echo "  kubectl exec $POD_NAME -- bash /workspace/benchmark-structured-outputs.sh"
echo ""
echo "To compare results:"
echo "  $SCRIPT_DIR/benchmark-compare.sh"
echo ""
echo "To check server logs:"
echo "  kubectl exec $POD_NAME -- tail -f /tmp/vllm-server.log"
echo ""
echo "To exec into pod:"
echo "  kubectl exec -it $POD_NAME -- bash"
echo ""
