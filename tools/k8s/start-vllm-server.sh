#!/bin/bash
# Startup script for LLaDA2.0-mini vLLM server with dLLM plugin
# Usage: kubectl exec llada2-debug -- bash /workspace/start-vllm-server.sh

set -e

echo "=== Installing system dependencies ==="
apt-get update -qq
apt-get install -y -qq git curl > /dev/null 2>&1

echo "=== Installing uv package manager ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    # Also update bashrc for future shells
    echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

echo "=== Setting up dLLM plugin ==="
cd /workspace

# Check if pyproject.toml exists (copied via tarball extraction)
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in /workspace!"
    echo "Please copy dllm-plugin files first"
    exit 1
fi

echo "=== Installing Python dependencies ==="
# Set version for setuptools-scm (since we don't have .git directory)
# Package name is "vllm-dllm-plugin" which normalizes to VLLM_DLLM_PLUGIN
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_DLLM_PLUGIN="0.1.0.dev"
uv sync --group dev --extra vllm

echo "=== Cleaning up any stuck GPU processes ==="
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 || true
pkill -9 -f vllm || true
sleep 2

echo "=== Starting vLLM server ==="
# IMPORTANT: Plugin entry point name is "dllm" (not "dllm_plugin")
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Use vllm serve CLI (simpler than python -m vllm.entrypoints.openai.api_server)
nohup vllm serve inclusionAI/LLaDA2.0-mini \
    --max-model-len 2048 \
    --max-num-seqs 32 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
    --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker \
    > /tmp/vllm-server.log 2>&1 &

echo "=== Waiting for server to be ready ==="
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ vLLM server is ready!"
        exit 0
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

echo "✗ Server failed to start within 60 seconds"
echo "Last 50 lines of server log:"
tail -50 /tmp/vllm-server.log
exit 1
