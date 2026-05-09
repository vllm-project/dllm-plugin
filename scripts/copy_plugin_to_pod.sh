#!/bin/bash
# Copy dllm plugin to Kubernetes pod for testing
# Phase 7: Plugin deployment

set -e

POD_NAME="${POD_NAME:-llada2-dev}"
NAMESPACE="${NAMESPACE:-default}"
PLUGIN_DIR="${PLUGIN_DIR:-$(pwd)}"

echo "Copying dllm plugin to pod..."
echo "  Pod: $POD_NAME"
echo "  Namespace: $NAMESPACE"
echo "  Source: $PLUGIN_DIR"

# Create temp directory on pod
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- mkdir -p /tmp/dllm

# Copy plugin files
kubectl cp "$PLUGIN_DIR/dllm_plugin" "$NAMESPACE/$POD_NAME:/tmp/dllm/"
kubectl cp "$PLUGIN_DIR/pyproject.toml" "$NAMESPACE/$POD_NAME:/tmp/dllm/"
kubectl cp "$PLUGIN_DIR/README.md" "$NAMESPACE/$POD_NAME:/tmp/dllm/"

# Install plugin in editable mode
echo "Installing plugin in editable mode..."
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- bash -c "cd /tmp/dllm && pip install -e . --no-build-isolation"

echo "Plugin installed successfully!"
echo ""
echo "Verify installation:"
echo "  kubectl exec -n $NAMESPACE $POD_NAME -- python -c 'import dllm_plugin; print(dllm_plugin.__version__)'"
