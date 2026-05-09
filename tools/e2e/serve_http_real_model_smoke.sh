#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# HTTP smoke test for real LLaDA2.0 model (Phase 7).
#
# **Phase 7 deliverable** (issue #25): Validate that real LLaDA2.0-mini model
# can serve HTTP requests via vLLM server with dLLM plugin.
#
# Tests:
# - Server starts successfully
# - Health endpoint responds with 200
# - Chat completions endpoint accepts requests
# - Response has expected JSON structure
#
# Usage:
#   ./tools/e2e/serve_http_real_model_smoke.sh
#
# Environment variables:
#   LLADA2_REAL_MODEL_ID: HuggingFace model ID (default: inclusionAI/LLaDA2.0-mini)
#   VLLM_DLLM_HTTP_SMOKE_PORT: Server port (default: 8767)
#   VLLM_ATTENTION_BACKEND: Attention backend (FLASH_ATTN or FLASHINFER)

set -euo pipefail

# Configuration
MODEL_ID="${LLADA2_REAL_MODEL_ID:-inclusionAI/LLaDA2.0-mini}"
PORT="${VLLM_DLLM_HTTP_SMOKE_PORT:-8767}"
TIMEOUT_HEALTH="${VLLM_DLLM_HEALTH_TIMEOUT:-120}"

# Environment setup for dLLM plugin
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DLLM_USE_MOCK_MODEL=0  # Use real model

echo "========================================="
echo "LLaDA2.0 Real Model HTTP Smoke Test"
echo "========================================="
echo "Model:              ${MODEL_ID}"
echo "Port:               ${PORT}"
echo "Attention backend:  ${VLLM_ATTENTION_BACKEND:-auto}"
echo "========================================="

# Start vLLM server in background
echo "[1/4] Starting vLLM server..."
uv run vllm serve "${MODEL_ID}" \
  --tokenizer "${MODEL_ID}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --enforce-eager \
  --max-model-len 256 \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Cleanup function
cleanup() {
  echo ""
  echo "[Cleanup] Terminating server (PID ${SERVER_PID})..."
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  echo "[Cleanup] Done"
}
trap cleanup EXIT INT TERM

# Wait for health endpoint
echo "[2/4] Waiting for server health..."
HEALTH_URL="http://127.0.0.1:${PORT}/health"
for i in $(seq 1 "${TIMEOUT_HEALTH}"); do
  code=$(curl -sS -o /dev/null -w "%{http_code}" "${HEALTH_URL}" 2>/dev/null || echo "000")
  if [[ "${code}" == "200" ]]; then
    echo "✓ Server healthy (${i}s)"
    break
  fi
  if [[ $((i % 10)) -eq 0 ]]; then
    echo "  ... waiting ${i}/${TIMEOUT_HEALTH}s"
  fi
  sleep 1
done

if [[ "${code}" != "200" ]]; then
  echo "✗ Health check failed after ${TIMEOUT_HEALTH}s (code: ${code})"
  exit 1
fi

# Send chat completion request
echo "[3/4] Sending chat completion request..."
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
CHAT_BODY=$(mktemp)

code=$(curl -sS -o "${CHAT_BODY}" -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llada2-test",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1,
    "temperature": 0
  }' \
  "${CHAT_URL}" 2>/dev/null || echo "000")

if [[ "${code}" != "200" ]]; then
  echo "✗ Chat completion failed (HTTP ${code})"
  echo "Response body:"
  cat "${CHAT_BODY}"
  rm -f "${CHAT_BODY}"
  exit 1
fi

echo "✓ Chat completion successful (HTTP 200)"

# Validate response structure
echo "[4/4] Validating response structure..."
python3 -c '
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

# Validate required fields
assert "choices" in data, "Missing choices field"
assert len(data["choices"]) >= 1, "Empty choices array"

choice = data["choices"][0]
assert "message" in choice or "text" in choice, "Missing message/text in choice"

print("✓ Response structure valid")
print(f"  - choices: {len(data[\"choices\"])}")
if "usage" in data:
    print(f"  - tokens: {data[\"usage\"]}")
' "${CHAT_BODY}"

rm -f "${CHAT_BODY}"

echo ""
echo "========================================="
echo "✓ ALL TESTS PASSED"
echo "========================================="
echo "Real LLaDA2.0 model HTTP smoke test successful!"
echo ""
