#!/usr/bin/env bash
# Benchmark a single optimization with GuideLLM
#
# Usage:
#   ./tools/benchmark_optimization.sh baseline
#   ./tools/benchmark_optimization.sh torch_compile benchmarks/custom_dir
#
# Environment variables:
#   MODEL_ID - Model to benchmark (default: inclusionAI/LLaDA2.0-mini)
#   TARGET - Server URL (default: http://localhost:8000)
#   BENCHMARK_SECONDS - Benchmark duration (default: 180)
#   PROMPT_TOKENS - Input length (default: 256)
#   OUTPUT_TOKENS - Output length (default: 1000)

set -euo pipefail

OPTIMIZATION_NAME="${1:-baseline}"
OUTPUT_DIR="${2:-benchmarks/$(date +%Y%m%d_%H%M%S)}"
MODEL_ID="${MODEL_ID:-inclusionAI/LLaDA2.0-mini}"
TARGET="${TARGET:-http://localhost:8000}"
BENCHMARK_SECONDS="${BENCHMARK_SECONDS:-180}"
PROMPT_TOKENS="${PROMPT_TOKENS:-256}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1000}"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Benchmarking: $OPTIMIZATION_NAME"
echo "Model: $MODEL_ID"
echo "Target: $TARGET"
echo "Duration: ${BENCHMARK_SECONDS}s"
echo "Input: ${PROMPT_TOKENS} tokens, Output: ${OUTPUT_TOKENS} tokens"
echo "Output: $OUTPUT_DIR/$OPTIMIZATION_NAME.json"
echo "=========================================="

# Run GuideLLM benchmark
guidellm benchmark \
  --target "$TARGET" \
  --model "$MODEL_ID" \
  --profile synchronous \
  --max-seconds "$BENCHMARK_SECONDS" \
  --data "prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
  --processor "$MODEL_ID" \
  --processor-args '{"trust_remote_code": true}' \
  > "$OUTPUT_DIR/$OPTIMIZATION_NAME.json"

# Extract key metrics
if [ -f "tools/extract_metrics.py" ]; then
    python3 tools/extract_metrics.py \
      "$OUTPUT_DIR/$OPTIMIZATION_NAME.json" \
      > "$OUTPUT_DIR/${OPTIMIZATION_NAME}_summary.txt"

    echo ""
    echo "=========================================="
    echo "Summary:"
    echo "=========================================="
    cat "$OUTPUT_DIR/${OPTIMIZATION_NAME}_summary.txt"
else
    echo "Warning: tools/extract_metrics.py not found, skipping summary extraction"
fi

echo ""
echo "Done. Full results: $OUTPUT_DIR/$OPTIMIZATION_NAME.json"
