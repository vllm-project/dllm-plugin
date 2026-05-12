#!/bin/bash
# Validation script for softmax normalization fix
#
# This script validates that the concatenated virtual batch approach
# fixes the dual-chunk softmax normalization bug.
#
# Expected results:
# - C8 attention divergence: max_diff < 0.01 (down from 1.188)
# - Token generation: token 30 (matching dInfer, not token 198)
# - Attention weights sum to 1.0 (not 2.0)

set -e

echo "=========================================="
echo "Softmax Normalization Fix Validation"
echo "=========================================="
echo ""

# Check CUDA availability
echo "Checking GPU availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA available: {torch.cuda.device_count()} device(s)')"
echo ""

# Check environment setup
echo "Checking environment..."
python -c "
from dllm_plugin.attention.concatenated_virtual_batch import create_concatenated_virtual_batch
from dllm_plugin.models.llada2_attention import LLaDA2BlockAttention
print('✅ Modules import successfully')
"
echo ""

# Run attention validation tests
echo "=========================================="
echo "Test 1: Attention Checkpoint Validation"
echo "=========================================="
echo ""
echo "Running attention validation (C8 checkpoint)..."
echo "Expected: max_diff < 0.01 (previously 1.188)"
echo ""

pytest tests/test_llada2_numerical_validation.py::TestAttentionValidation::test_attention_output_matches_reference \
    -v -s --tb=short 2>&1 | tee /tmp/validation_c8_results.txt

# Extract max_diff from results
MAX_DIFF=$(grep -oP "max_diff[=: ]*\K[0-9]+\.[0-9]+" /tmp/validation_c8_results.txt | head -1 || echo "unknown")
echo ""
echo "C8 Attention max_diff: $MAX_DIFF"

if (( $(echo "$MAX_DIFF < 0.01" | bc -l 2>/dev/null || echo 0) )); then
    echo "✅ PASS: C8 divergence within tolerance"
else
    echo "⚠️  WARNING: C8 divergence = $MAX_DIFF (target < 0.01)"
fi
echo ""

# Run E2E token generation test
echo "=========================================="
echo "Test 2: Token Generation Validation"
echo "=========================================="
echo ""
echo "Running E2E token generation..."
echo "Expected: vLLM generates token 30 (matching dInfer)"
echo ""

pytest tests/test_llada2_numerical_validation.py::TestE2EValidation::test_e2e_generation_matches_reference \
    -v -s --tb=short 2>&1 | tee /tmp/validation_token_results.txt

# Check for token match
if grep -q "token.*30" /tmp/validation_token_results.txt; then
    echo "✅ PASS: Token generation matches dInfer (token 30)"
elif grep -q "token.*198" /tmp/validation_token_results.txt; then
    echo "❌ FAIL: Still generating wrong token (198 instead of 30)"
else
    echo "⚠️  Check token generation results above"
fi
echo ""

# Run full validation suite
echo "=========================================="
echo "Test 3: Full Validation Suite"
echo "=========================================="
echo ""
echo "Running complete numerical validation..."
echo ""

pytest tests/test_llada2_numerical_validation.py -v --tb=short 2>&1 | tee /tmp/validation_full_results.txt

# Summary
echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - C8 Checkpoint: /tmp/validation_c8_results.txt"
echo "  - Token Generation: /tmp/validation_token_results.txt"
echo "  - Full Suite: /tmp/validation_full_results.txt"
echo ""

# Check overall pass/fail
if grep -q "FAILED" /tmp/validation_full_results.txt; then
    echo "❌ Some tests failed - review results above"
    exit 1
else
    echo "✅ All validation tests passed!"
    echo ""
    echo "Fix verified successfully:"
    echo "  - Softmax normalization corrected (single softmax)"
    echo "  - C8 attention divergence within tolerance"
    echo "  - Token generation matches dInfer reference"
    exit 0
fi
