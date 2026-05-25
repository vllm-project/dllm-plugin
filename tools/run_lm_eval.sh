#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Phase 9.2: Run lm-eval GSM8K with both dInfer and dllm-plugin
#
# Prerequisites (on A100 pod):
#   pip install lm-eval>=0.4.0 accelerate
#   pip install -e /workspace/dllm-plugin
#   pip install dinfer  # or clone github.com/inclusionAI/dInfer
#
# Usage:
#   bash tools/run_lm_eval.sh

set -euo pipefail

MODEL="${MODEL:-inclusionAI/LLaDA2.0-mini}"
OUTDIR="${OUTDIR:-results/phase9.2}"
TASKS_DIR="$(cd "$(dirname "$0")/.." && pwd)/evaluations/tasks"

mkdir -p "$OUTDIR"

echo "======================================================"
echo "Phase 9.2: lm-eval GSM8K Comparison"
echo "Model: $MODEL"
echo "Tasks dir: $TASKS_DIR"
echo "Output: $OUTDIR"
echo "======================================================"

# --- dInfer baseline ---
echo ""
echo "=== [1/2] dInfer baseline ==="
if python -c "from dinfer import BlockDiffusionLLM" 2>/dev/null; then
    python -m lm_eval \
        --model dInfer_eval \
        --model_args "model_path=$MODEL,gen_length=2048,block_length=32,threshold=0.90,parallel_decoding=threshold,use_bd=True,cache=prefix,use_compile=False,tp_size=1,parallel=dp,model_type=llada2,use_credit=False,cont_weight=0,master_port=23456" \
        --tasks gsm8k_llada_mini \
        --include_path "$TASKS_DIR" \
        --output_path "$OUTDIR/dinfer" \
        --confirm_run_unsafe_code \
        --apply_chat_template 2>&1 | tee "$OUTDIR/dinfer.log"
    echo "dInfer results: $OUTDIR/dinfer/"
else
    echo "SKIP: dInfer not installed. Install with: pip install dinfer"
    echo "  or: pip install git+https://github.com/inclusionAI/dInfer.git"
fi

# --- dllm-plugin ---
echo ""
echo "=== [2/2] dllm-plugin (vLLM) ==="
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python tools/run_lm_eval_plugin.py \
    --model dllm_plugin_eval \
    --model_args "model_path=$MODEL,gen_length=2048,max_model_len=4096" \
    --tasks gsm8k_llada_mini \
    --include_path "$TASKS_DIR" \
    --output_path "$OUTDIR/dllm_plugin" \
    --confirm_run_unsafe_code \
    --apply_chat_template 2>&1 | tee "$OUTDIR/dllm_plugin.log"
echo "dllm-plugin results: $OUTDIR/dllm_plugin/"

# --- Compare ---
echo ""
echo "=== Comparison ==="
DINFER_JSON=$(find "$OUTDIR/dinfer" -name "results*.json" 2>/dev/null | head -1)
PLUGIN_JSON=$(find "$OUTDIR/dllm_plugin" -name "results*.json" 2>/dev/null | head -1)

if [ -n "$DINFER_JSON" ] && [ -n "$PLUGIN_JSON" ]; then
    python tools/compare_lm_eval_results.py "$DINFER_JSON" "$PLUGIN_JSON"
else
    echo "Cannot compare — one or both result files missing"
    [ -z "$DINFER_JSON" ] && echo "  Missing: dInfer results"
    [ -z "$PLUGIN_JSON" ] && echo "  Missing: dllm-plugin results"
fi

echo ""
echo "======================================================"
echo "Done. Results in: $OUTDIR/"
echo "======================================================"
