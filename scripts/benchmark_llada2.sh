#!/bin/bash
# Benchmark LLaDA2.0-mini with dllm plugin using guidellm
# Phase 7: Virtual batch attention performance testing

set -e

# Configuration
TARGET="${TARGET:-http://localhost:8000}"
MODEL="${MODEL:-inclusionAI/LLaDA2.0-mini}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"

echo "Running guidellm benchmarks..."
echo "  Target: $TARGET"
echo "  Model: $MODEL"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Test 1: Short sequences (32 prompt + 32 output)
echo "=== Test 1: Short sequences (32 + 32 tokens) ==="
guidellm benchmark run \
  --target "$TARGET" \
  --model "$MODEL" \
  --data '{"prompt_tokens": 32, "output_tokens": 32}' \
  --profile constant \
  --rate 1 \
  --max-requests 10 \
  --request-format /v1/completions \
  --output-dir "$OUTPUT_DIR" \
  --outputs short_bench.json,short_bench.csv

echo ""
echo "=== Test 2: Medium sequences (32 + 900 tokens) ==="
guidellm benchmark run \
  --target "$TARGET" \
  --model "$MODEL" \
  --data '{"prompt_tokens": 32, "output_tokens": 900}' \
  --profile synchronous \
  --max-requests 5 \
  --request-format /v1/completions \
  --output-dir "$OUTPUT_DIR" \
  --outputs medium_bench.json,medium_bench.csv

echo ""
echo "=== Test 3: Long sequences (1000 + 1000 tokens) ==="
guidellm benchmark run \
  --target "$TARGET" \
  --model "$MODEL" \
  --data '{"prompt_tokens": 1000, "output_tokens": 1000}' \
  --profile synchronous \
  --max-requests 5 \
  --request-format /v1/completions \
  --output-dir "$OUTPUT_DIR" \
  --outputs long_bench.json,long_bench.csv

echo ""
echo "Benchmark complete! Results saved to $OUTPUT_DIR"
echo ""
echo "Summary:"
echo "  - short_bench.{json,csv}: 32+32 tokens, constant rate"
echo "  - medium_bench.{json,csv}: 32+900 tokens, synchronous"
echo "  - long_bench.{json,csv}: 1000+1000 tokens, synchronous"
