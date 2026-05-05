#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Maintainer smoke: vLLM OpenAI HTTP server + mock dLLM stack (issue #35 / PR review).
#
# Requirements: repository root as cwd, Linux + CUDA, `uv sync --group dev --extra vllm`.
# Uses curl only (no GuideLLM). Stops the server on exit. Readiness and chat
# requests assert **HTTP 200** explicitly via ``curl -w '%{http_code}'`` (not only
# transport success). ``max_tokens`` stays at 1 so this stays a shallow decode
# smoke (aligned with ``tests/test_vllm_mock_integration.py`` on tight L4 KV).
#
# Env (common):
#   VLLM_PLUGINS=dllm
#   VLLM_USE_V2_MODEL_RUNNER=1
#   VLLM_ENABLE_V1_MULTIPROCESSING=0
#   VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK=1   (set below; use on legacy wheels)
# Optional:
#   VLLM_DLLM_HTTP_SMOKE_PORT (default 8765)
#   VLLM_DLLM_HTTP_SMOKE_MODEL_NAME (default dllm-plugin-mock-http-smoke)
#   DLLM_TEST_GPU_MEMORY_UTILIZATION / DLLM_TEST_KV_CACHE_MEMORY_BYTES (see tests/gpu_memory.py)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${ROOT}"

export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK=1

MODEL_DIR="${ROOT}/tests/fixtures/mock_llada2_hf_config"
PORT="${VLLM_DLLM_HTTP_SMOKE_PORT:-8765}"
SERVED_NAME="${VLLM_DLLM_HTTP_SMOKE_MODEL_NAME:-dllm-plugin-mock-http-smoke}"

GPU_UTIL="${DLLM_TEST_GPU_MEMORY_UTILIZATION:-0.9}"
KV_RAW="${DLLM_TEST_KV_CACHE_MEMORY_BYTES:-}"
SERVE_EXTRA=(--gpu-memory-utilization "${GPU_UTIL}")
kvl=$(printf "%s" "${KV_RAW}" | tr "[:upper:]" "[:lower:]")
if [[ -z "${KV_RAW}" ]]; then
  SERVE_EXTRA+=(--kv-cache-memory-bytes 268435456)
elif [[ "${kvl}" == "none" || "${kvl}" == "null" ]]; then
  :
else
  SERVE_EXTRA+=(--kv-cache-memory-bytes "${KV_RAW}")
fi

SERVE_PID=""
CHAT_BODY=""
cleanup() {
  rm -f "${CHAT_BODY}"
  if [[ -n "${SERVE_PID}" ]] && kill -0 "${SERVE_PID}" 2>/dev/null; then
    kill "${SERVE_PID}" 2>/dev/null || true
    wait "${SERVE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "serve_http_smoke: starting vllm serve on 127.0.0.1:${PORT} (model ${SERVED_NAME})" >&2
uv run vllm serve "${MODEL_DIR}" \
  --tokenizer "${MODEL_DIR}" \
  --served-model-name "${SERVED_NAME}" \
  --no-async-scheduling \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --enforce-eager \
  --load-format dummy \
  --max-model-len 128 \
  --max-num-seqs 1 \
  --tensor-parallel-size 1 \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker \
  "${SERVE_EXTRA[@]}" &
SERVE_PID=$!

ready=0
for _ in $(seq 1 120); do
  code=$(curl -sS -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" || echo "000")
  if [[ "${code}" == "200" ]]; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "serve_http_smoke: /health did not return HTTP 200 in time" >&2
  exit 1
fi

echo "serve_http_smoke: POST /v1/chat/completions" >&2
CHAT_BODY=$(mktemp)
code=$(
  curl -sS -o "${CHAT_BODY}" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" \
    "http://127.0.0.1:${PORT}/v1/chat/completions" || echo "000"
)
if [[ "${code}" != "200" ]]; then
  echo "serve_http_smoke: expected HTTP 200 from /v1/chat/completions, got ${code}" >&2
  cat "${CHAT_BODY}" >&2 || true
  exit 1
fi

python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); assert "choices" in j and len(j["choices"])>=1, j' "${CHAT_BODY}"

echo "serve_http_smoke: OK" >&2
