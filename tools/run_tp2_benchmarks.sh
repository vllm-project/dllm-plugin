#!/bin/bash
# Run 6 benchmark configurations for TP=2 validation
# Matches the configurations from BENCHMARK_RESULTS_MULTI_REQUEST.md

set -e

TARGET="http://localhost:8000"
MODEL="inclusionAI/LLaDA2.0-mini"
OUTPUT_DIR="benchmarks/tp2_validation"
REGEX_PATTERN="[A-Z][a-z]+( [A-Z][a-z]+)*"

echo "=================================================="
echo "TP=2 Validation Benchmarks"
echo "=================================================="
echo "Target: $TARGET"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Verify server is healthy
echo "Checking server health..."
if ! curl -sf "$TARGET/health" > /dev/null; then
    echo "ERROR: Server not responding at $TARGET"
    exit 1
fi
echo "✓ Server healthy"
echo ""

# Scenario 1: Synchronous + Free-form
echo "=== [1/6] Synchronous + Free-form ==="
uv run guidellm benchmark run \
  --target "$TARGET" \
  --model "$MODEL" \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile synchronous \
  --max-requests 10000 \
  --max-seconds 300 \
  --output-path "$OUTPUT_DIR/1_sync_freeform.json"

echo ""
echo "=== [2/6] Synchronous + Structured ==="
uv run guidellm benchmark run \
  --target "$TARGET" \
  --model "$MODEL" \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile synchronous \
  --max-requests 10000 \
  --max-seconds 300 \
  --backend-kwargs "{\"extras\": {\"body\": {\"guided_regex\": \"$REGEX_PATTERN\"}}}" \
  --output-path "$OUTPUT_DIR/2_sync_structured.json"

echo ""
echo "=== [3/6] Constant 5 RPS + Free-form ==="
uv run guidellm benchmark run \
  --target "$TARGET" \
  --model "$MODEL" \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile constant \
  --rate 5 \
  --max-requests 10000 \
  --max-seconds 300 \
  --output-path "$OUTPUT_DIR/3_5rps_freeform.json"

echo ""
echo "=== [4/6] Constant 5 RPS + Structured ==="
uv run guidellm benchmark run \
  --target "$TARGET" \
  --model "$MODEL" \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile constant \
  --rate 5 \
  --max-requests 10000 \
  --max-seconds 300 \
  --backend-kwargs "{\"extras\": {\"body\": {\"guided_regex\": \"$REGEX_PATTERN\"}}}" \
  --output-path "$OUTPUT_DIR/4_5rps_structured.json"

echo ""
echo "=== [5/6] Constant 10 RPS + Free-form ==="
uv run guidellm benchmark run \
  --target "$TARGET" \
  --model "$MODEL" \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile constant \
  --rate 10 \
  --max-requests 10000 \
  --max-seconds 300 \
  --output-path "$OUTPUT_DIR/5_10rps_freeform.json"

echo ""
echo "=== [6/6] Constant 10 RPS + Structured ==="
uv run guidellm benchmark run \
  --target "$TARGET" \
  --model "$MODEL" \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --profile constant \
  --rate 10 \
  --max-requests 10000 \
  --max-seconds 300 \
  --backend-kwargs "{\"extras\": {\"body\": {\"guided_regex\": \"$REGEX_PATTERN\"}}}" \
  --output-path "$OUTPUT_DIR/6_10rps_structured.json"

echo ""
echo "=================================================="
echo "All benchmarks complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================="
