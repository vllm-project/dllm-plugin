#!/bin/bash
# Start vLLM server with LLaDA2.0-mini and dllm plugin
# Phase 7: Virtual batch attention implementation

set -e

# Configuration
MODEL="inclusionAI/LLaDA2.0-mini"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"

echo "Starting vLLM server with dllm plugin..."
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  Max model length: $MAX_MODEL_LEN"
echo "  GPU memory utilization: $GPU_MEMORY_UTIL"

# Required environment variables
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1

# Start server
vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --trust-remote-code \
  --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
  --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker
