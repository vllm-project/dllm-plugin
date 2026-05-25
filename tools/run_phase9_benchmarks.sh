#!/usr/bin/env bash
# Phase 9 Benchmark Suite — 6 scenarios with py-spy profiling
#
# Runs on an A100 pod with vLLM + dllm-plugin installed.
# Each scenario: start vLLM under py-spy → run guidellm → stop → extract metrics.
#
# Usage (on the pod):
#   bash /workspace/scripts/run_phase9_benchmarks.sh
#
# Prerequisites:
#   pip install py-spy guidellm
#   vllm + dllm-plugin installed and working

set -euo pipefail

MODEL="${MODEL:-/workspace/llada2-mini}"
PORT=8000
TARGET="http://localhost:$PORT"
OUTDIR="/workspace/benchmarks/phase9"
REGEX_PATTERN='[A-Z][a-z]+( [A-Z][a-z]+)*'
MAX_SECONDS=300
DATA="prompt_tokens=500,output_tokens=500,count=10000"

export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

mkdir -p "$OUTDIR"

start_server() {
    local scenario="$1"
    echo "  Starting vLLM with py-spy (scenario: $scenario)..."
    pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
    sleep 3

    py-spy record -f raw -o "$OUTDIR/${scenario}.raw" \
        --subprocesses --full-filenames -r 1 -- \
        vllm serve "$MODEL" \
            --max-model-len 1024 \
            --max-num-seqs 8 \
            --port "$PORT" \
            --trust-remote-code \
            --gpu-memory-utilization 0.9 \
            --no-enable-prefix-caching \
            --scheduler-cls dllm_plugin.Scheduler \
            --worker-cls dllm_plugin.Worker \
        > "$OUTDIR/${scenario}_server.log" 2>&1 &
    SERVER_PID=$!

    echo "  Waiting for server (PID $SERVER_PID)..."
    for i in $(seq 1 300); do
        if curl -sf "$TARGET/health" > /dev/null 2>&1; then
            echo "  Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server did not start in 300s"
    cat "$OUTDIR/${scenario}_server.log" | tail -20
    return 1
}

stop_server() {
    echo "  Stopping server..."
    pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
    sleep 3
}

run_guidellm() {
    local scenario="$1"
    shift
    echo "  Running guidellm ($scenario)..."
    guidellm benchmark run \
        --target "$TARGET" \
        --model "$MODEL" \
        --data "$DATA" \
        --max-requests 10000 \
        --max-seconds "$MAX_SECONDS" \
        --output-path "$OUTDIR/${scenario}.json" \
        "$@"
}

extract() {
    local scenario="$1"
    if [ -f "/workspace/dllm-plugin/tools/extract_metrics.py" ]; then
        python3 /workspace/dllm-plugin/tools/extract_metrics.py \
            "$OUTDIR/${scenario}.json" 2>/dev/null || echo "  (extraction failed)"
    else
        echo "  (extract_metrics.py not found)"
    fi
}

echo "======================================================"
echo "Phase 9 Benchmark Suite — $(date)"
echo "Model: $MODEL"
echo "6 scenarios × py-spy profiling"
echo "======================================================"
echo ""

# Scenario 1: Synchronous + Free-form
echo "=== [1/6] Synchronous + Free-form ==="
start_server "1_sync_freeform"
run_guidellm "1_sync_freeform" --profile synchronous
extract "1_sync_freeform"
stop_server
echo ""

# Scenario 2: Synchronous + Structured
echo "=== [2/6] Synchronous + Structured ==="
start_server "2_sync_structured"
run_guidellm "2_sync_structured" --profile synchronous \
    --backend-kwargs "{\"extras\": {\"body\": {\"guided_regex\": \"$REGEX_PATTERN\"}}}"
extract "2_sync_structured"
stop_server
echo ""

# Scenario 3: 5 RPS + Free-form
echo "=== [3/6] Constant 5 RPS + Free-form ==="
start_server "3_5rps_freeform"
run_guidellm "3_5rps_freeform" --profile constant --rate 5
extract "3_5rps_freeform"
stop_server
echo ""

# Scenario 4: 5 RPS + Structured
echo "=== [4/6] Constant 5 RPS + Structured ==="
start_server "4_5rps_structured"
run_guidellm "4_5rps_structured" --profile constant --rate 5 \
    --backend-kwargs "{\"extras\": {\"body\": {\"guided_regex\": \"$REGEX_PATTERN\"}}}"
extract "4_5rps_structured"
stop_server
echo ""

# Scenario 5: 20 RPS + Free-form
echo "=== [5/6] Constant 20 RPS + Free-form ==="
start_server "5_20rps_freeform"
run_guidellm "5_20rps_freeform" --profile constant --rate 20
extract "5_20rps_freeform"
stop_server
echo ""

# Scenario 6: 20 RPS + Structured
echo "=== [6/6] Constant 20 RPS + Structured ==="
start_server "6_20rps_structured"
run_guidellm "6_20rps_structured" --profile constant --rate 20 \
    --backend-kwargs "{\"extras\": {\"body\": {\"guided_regex\": \"$REGEX_PATTERN\"}}}"
extract "6_20rps_structured"
stop_server
echo ""

echo "======================================================"
echo "All 6 scenarios complete."
echo "Results: $OUTDIR/"
echo "  JSON:     $OUTDIR/*.json"
echo "  Profiles: $OUTDIR/*.raw"
echo "  Logs:     $OUTDIR/*_server.log"
echo "======================================================"

# Summary table
echo ""
echo "=== SUMMARY ==="
for f in "$OUTDIR"/*.json; do
    name=$(basename "$f" .json)
    echo "--- $name ---"
    extract "$name"
    echo ""
done
