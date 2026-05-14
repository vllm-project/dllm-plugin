#!/bin/bash
# Orchestrate the full numerical investigation.
#
# Usage:
#   ./tools/investigation/run_investigation.sh
#
# This script:
# 1. Deploys both K8s pods (if not running)
# 2. Sets up both environments
# 3. Runs exhaustive captures on both pods (in parallel)
# 4. Copies captures locally
# 5. Runs comparison and analysis
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCAL_CAPTURES="$REPO_ROOT/investigation_captures"

DINFER_POD="investigation-dinfer"
VLLM_POD="investigation-vllm"

echo "================================================================"
echo "LLaDA2.0-mini Numerical Investigation"
echo "dInfer (vLLM 0.10.2) vs dllm-plugin (vLLM 0.20.1)"
echo "================================================================"

# ─── Phase 0: Setup ──────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 0: Environment Setup                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"

echo ""
echo "[0.1] Setting up dInfer pod..."
bash "$SCRIPT_DIR/setup_dinfer_pod.sh"

echo ""
echo "[0.2] Setting up vLLM pod..."
bash "$SCRIPT_DIR/setup_vllm_pod.sh"

# ─── Phase 2+3: Capture (parallel) ──────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 2+3: Exhaustive Tensor Capture (parallel)            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

echo ""
echo "[2] Starting dInfer captures (all scenarios, all layers, all levels)..."
kubectl exec "$DINFER_POD" -- python3 /workspace/scripts/capture_dinfer_all.py --all-scenarios &
DINFER_PID=$!

echo "[3] Starting vLLM captures (all scenarios, all layers, all levels)..."
kubectl exec "$VLLM_POD" -- python3 /workspace/scripts/capture_vllm_all.py --all-scenarios &
VLLM_PID=$!

echo ""
echo "Both captures running in parallel. Waiting for completion..."
echo "  dInfer PID: $DINFER_PID"
echo "  vLLM PID:   $VLLM_PID"

# Wait for both to complete
DINFER_EXIT=0
VLLM_EXIT=0
wait $DINFER_PID || DINFER_EXIT=$?
wait $VLLM_PID || VLLM_EXIT=$?

echo ""
if [ $DINFER_EXIT -ne 0 ]; then
    echo "[WARN] dInfer capture exited with code $DINFER_EXIT"
fi
if [ $VLLM_EXIT -ne 0 ]; then
    echo "[WARN] vLLM capture exited with code $VLLM_EXIT"
fi

# ─── Copy captures locally ───────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Copying captures to local machine                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

mkdir -p "$LOCAL_CAPTURES"

echo "[4.1] Copying dInfer captures..."
kubectl cp "$DINFER_POD:/workspace/captures" "$LOCAL_CAPTURES/" 2>/dev/null || {
    echo "[WARN] kubectl cp failed for dInfer, trying tar method..."
    kubectl exec "$DINFER_POD" -- tar czf /tmp/dinfer_captures.tar.gz -C /workspace captures/
    kubectl cp "$DINFER_POD:/tmp/dinfer_captures.tar.gz" "$LOCAL_CAPTURES/dinfer_captures.tar.gz"
    cd "$LOCAL_CAPTURES" && tar xzf dinfer_captures.tar.gz && rm dinfer_captures.tar.gz
}

echo "[4.2] Copying vLLM captures..."
kubectl cp "$VLLM_POD:/workspace/captures" "$LOCAL_CAPTURES/" 2>/dev/null || {
    echo "[WARN] kubectl cp failed for vLLM, trying tar method..."
    kubectl exec "$VLLM_POD" -- tar czf /tmp/vllm_captures.tar.gz -C /workspace captures/
    kubectl cp "$VLLM_POD:/tmp/vllm_captures.tar.gz" "$LOCAL_CAPTURES/vllm_captures.tar.gz"
    cd "$LOCAL_CAPTURES" && tar xzf vllm_captures.tar.gz && rm vllm_captures.tar.gz
}

echo ""
echo "Captures stored at: $LOCAL_CAPTURES"
find "$LOCAL_CAPTURES" -name "*.pt" | wc -l | xargs -I{} echo "  Total .pt files: {}"
du -sh "$LOCAL_CAPTURES" | awk '{print "  Total size: " $1}'

# ─── Phase 4: Comparison ────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 4: Comprehensive Comparison (CPU-only)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

python3 "$SCRIPT_DIR/scripts/compare_checkpoints.py" \
    --captures-dir "$LOCAL_CAPTURES" \
    --all-scenarios \
    --output-dir "$LOCAL_CAPTURES/comparisons"

# ─── Phase 6: Replay Isolation ───────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 6: Replay Isolation                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Note: replay needs model weights, so runs on a pod
for scenario in first_block nth_block_p32; do
    echo "[6] Replay isolation for $scenario..."
    # Copy comparison captures to vllm pod for replay
    kubectl cp "$LOCAL_CAPTURES" "$VLLM_POD:/workspace/investigation_captures" 2>/dev/null || true
    kubectl exec "$VLLM_POD" -- python3 /workspace/scripts/replay_operations.py \
        --captures-dir /workspace/investigation_captures \
        --scenario "$scenario" \
        --output-dir /workspace/investigation_captures/comparisons || true
    # Copy replay results back
    kubectl cp "$VLLM_POD:/workspace/investigation_captures/comparisons" \
        "$LOCAL_CAPTURES/comparisons" 2>/dev/null || true
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Investigation Complete                                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results at: $LOCAL_CAPTURES/comparisons/"
echo ""
echo "Key files:"
echo "  $LOCAL_CAPTURES/comparisons/COMPARISON_SUMMARY.md"
echo "  $LOCAL_CAPTURES/comparisons/first_block_report.json"
echo "  $LOCAL_CAPTURES/comparisons/first_block_replay.json"
echo ""
echo "To view summary:"
echo "  cat $LOCAL_CAPTURES/comparisons/COMPARISON_SUMMARY.md"
