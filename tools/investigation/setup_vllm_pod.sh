#!/bin/bash
# Setup investigation-vllm pod for exhaustive tensor capture.
# Idempotent — safe to run multiple times.
set -euo pipefail

POD_NAME="investigation-vllm"
TARGET_DIR="/workspace"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "================================================================"
echo "vLLM/dllm-plugin Investigation Pod Setup"
echo "================================================================"

# Step 1: Verify pod
echo "[1/5] Verifying pod..."
POD_STATUS=$(kubectl get pod "$POD_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
if [ "$POD_STATUS" != "Running" ]; then
    echo "Pod not running (status: $POD_STATUS). Creating..."
    kubectl apply -f "$SCRIPT_DIR/pod-vllm.yaml"
    echo "Waiting for pod to start..."
    kubectl wait --for=condition=Ready pod/"$POD_NAME" --timeout=300s
fi
echo "Pod $POD_NAME is running"

# Step 2: Copy investigation scripts
echo "[2/5] Copying investigation scripts..."
kubectl exec "$POD_NAME" -- mkdir -p "$TARGET_DIR/scripts" 2>/dev/null || true
for f in "$SCRIPT_DIR/scripts/"*.py "$SCRIPT_DIR/scripts/"*.json; do
    [ -f "$f" ] && kubectl cp "$f" "$POD_NAME:$TARGET_DIR/scripts/$(basename "$f")" 2>/dev/null || true
done
echo "Scripts copied"

# Step 3: Copy full plugin source
echo "[3/5] Copying dllm-plugin source..."
kubectl exec "$POD_NAME" -- mkdir -p \
    "$TARGET_DIR/dllm-plugin/dllm_plugin/models" \
    "$TARGET_DIR/dllm-plugin/dllm_plugin/attention" \
    "$TARGET_DIR/dllm-plugin/dllm_plugin/validation" \
    "$TARGET_DIR/dllm-plugin/dllm_plugin/remasking" 2>/dev/null || true

kubectl cp "$REPO_ROOT/pyproject.toml" "$POD_NAME:$TARGET_DIR/dllm-plugin/pyproject.toml" 2>/dev/null || true
kubectl cp "$REPO_ROOT/README.md" "$POD_NAME:$TARGET_DIR/dllm-plugin/README.md" 2>/dev/null || true

for f in __init__.py config.py vllm_compat.py forward_context.py validation_utils.py \
         validation.py gpu_capability.py scheduler.py runtime_scheduler.py worker.py \
         gpu_model_runner.py vllm_gpu_model_runner_fork.py vllm_types.py \
         grammar_utils.py runtime_worker.py engine_core_draft_hook.py; do
    [ -f "$REPO_ROOT/dllm_plugin/$f" ] && \
        kubectl cp "$REPO_ROOT/dllm_plugin/$f" "$POD_NAME:$TARGET_DIR/dllm-plugin/dllm_plugin/$f" 2>/dev/null || true
done

for f in llada2.py llada2_attention.py llada2_attention_backend.py mock_llada2.py __init__.py; do
    [ -f "$REPO_ROOT/dllm_plugin/models/$f" ] && \
        kubectl cp "$REPO_ROOT/dllm_plugin/models/$f" "$POD_NAME:$TARGET_DIR/dllm-plugin/dllm_plugin/models/$f" 2>/dev/null || true
done

for f in __init__.py virtual_batches.py concatenated_virtual_batch.py; do
    [ -f "$REPO_ROOT/dllm_plugin/attention/$f" ] && \
        kubectl cp "$REPO_ROOT/dllm_plugin/attention/$f" "$POD_NAME:$TARGET_DIR/dllm-plugin/dllm_plugin/attention/$f" 2>/dev/null || true
done

for f in __init__.py chunked_attention_hooks.py; do
    [ -f "$REPO_ROOT/dllm_plugin/validation/$f" ] && \
        kubectl cp "$REPO_ROOT/dllm_plugin/validation/$f" "$POD_NAME:$TARGET_DIR/dllm-plugin/dllm_plugin/validation/$f" 2>/dev/null || true
done

for f in __init__.py base.py handoff.py llada2_default.py; do
    [ -f "$REPO_ROOT/dllm_plugin/remasking/$f" ] && \
        kubectl cp "$REPO_ROOT/dllm_plugin/remasking/$f" "$POD_NAME:$TARGET_DIR/dllm-plugin/dllm_plugin/remasking/$f" 2>/dev/null || true
done

echo "Plugin source copied"

# Step 4: Install dependencies
echo "[4/5] Installing dependencies..."
kubectl exec "$POD_NAME" -- bash -c '
set -e
pip install -q --upgrade pip 2>/dev/null

# Install vLLM 0.20.1
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "  Installing vLLM..."
    pip install -q vllm
fi

pip install -q huggingface_hub 2>/dev/null

# Install dllm-plugin in editable mode
echo "  Installing dllm-plugin..."
cd /workspace/dllm-plugin
pip install -q -e . 2>/dev/null || pip install -q -e ".[dev]" 2>/dev/null || true

python3 -c "
import torch, vllm
print(f\"  torch: {torch.__version__}\")
print(f\"  vllm: {vllm.__version__}\")
print(f\"  CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"})\")
try:
    from dllm_plugin.models.llada2 import LLaDA2ForCausalLM
    print(f\"  dllm_plugin: installed\")
except ImportError as e:
    print(f\"  dllm_plugin: FAILED ({e})\")
"
'
echo "Dependencies installed"

# Step 5: Download model
echo "[5/5] Downloading model..."
kubectl exec "$POD_NAME" -- bash -c '
if [ -f "/workspace/llada2-mini/config.json" ]; then
    echo "  Model already downloaded"
    du -sh /workspace/llada2-mini
else
    echo "  Downloading inclusionAI/LLaDA2.0-mini..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(\"inclusionAI/LLaDA2.0-mini\", local_dir=\"/workspace/llada2-mini\",
                  local_dir_use_symlinks=False, resume_download=True)
print(\"  Model downloaded\")
"
fi
'

echo ""
echo "================================================================"
echo "vLLM pod ready. Run captures with:"
echo "  kubectl exec $POD_NAME -- python3 /workspace/scripts/capture_vllm_all.py"
echo "================================================================"
