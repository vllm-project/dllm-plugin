#!/bin/bash
# Cleanup script for TP=2 validation environment
# Stops server, removes port-forward, and optionally deletes pod

set -e

POD_NAME="llada2-tp2-debug"

echo "=================================================="
echo "TP=2 Validation Cleanup"
echo "=================================================="
echo ""

# Stop port-forward
echo "=== Stopping Port Forward ==="
if pkill -f "kubectl port-forward $POD_NAME" 2>/dev/null; then
    echo "✓ Port forward stopped"
else
    echo "ℹ No port forward found"
fi
echo ""

# Stop vLLM server
echo "=== Stopping vLLM Server ==="
if kubectl get pod $POD_NAME &>/dev/null; then
    kubectl exec $POD_NAME -- pkill -f vllm.entrypoints.openai.api_server 2>/dev/null || {
        echo "ℹ Server not running or already stopped"
    }
    echo "✓ Server stopped"
else
    echo "ℹ Pod not found, skipping server stop"
fi
echo ""

# Optionally delete pod
echo "=== Delete Pod ==="
echo "Do you want to delete the pod '$POD_NAME'? (y/N)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    if kubectl get pod $POD_NAME &>/dev/null; then
        kubectl delete pod $POD_NAME
        echo "✓ Pod deleted"
    else
        echo "ℹ Pod not found, nothing to delete"
    fi
else
    echo "ℹ Pod kept (you can delete it later with: kubectl delete pod $POD_NAME)"
fi
echo ""

echo "=================================================="
echo "Cleanup Complete"
echo "=================================================="
echo ""
echo "To start a fresh validation, run:"
echo "  ./tools/setup_tp2_validation.sh"
