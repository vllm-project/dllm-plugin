# Known Limitations - Phase 7 & 8

**Last Updated:** 2026-05-09  
**Scope:** LLaDA2.0 real model implementation (Phase 7 + 8)

This document tracks known limitations, unvalidated assumptions, and deferred work for the Phase 7+8 release.

---

## P0 - Critical Limitations (User-Facing Impact)

### 1. MoE Router Numerical Precision (FP32 Default, BF16 Opt-In)

**Status:** Phase 7 defaults to FP32 router (safe) with BF16 experimental opt-in via `VLLM_LLADA2_BF16_ROUTER=1`.

**Default Behavior (FP32 Router):**
- Router computation uses FP32 precision following DeepSeek V3 and Qwen2-MoE patterns
- Sigmoid activation computed in FP32 then cast back to input dtype
- Validated pattern from other vLLM MoE models

**Experimental BF16 Mode:**
Set `VLLM_LLADA2_BF16_ROUTER=1` to use BF16 router (faster but unvalidated):
- Router computation uses same dtype as hidden_states (typically BF16)
- Potential risks:
  - Expert selection bias if sigmoid saturates in low precision
  - Routing entropy loss if scores collapse or lose precision  
  - Silent correctness degradation (no NaN/inf failure mode)
- Logs warning at first use

**Why BF16 is unvalidated:**
- No comparison against FP32 router baseline in production workloads
- No validation against HuggingFace reference implementation
- Sigmoid (vs softmax) stability in BF16 is untested for group-limited routing

**Comparison to other models:**
- DeepSeek V3: FP32 router required (vllm PR #14027) due to softmax precision issues
- Qwen2-MoE: FP32 gating (vllm/models/qwen2_moe.py:167)
- LLaDA2.0: FP32 default (this implementation), BF16 experimental

**Validation Plan (Phase 9, issue #39):**
1. Compare BF16 vs FP32 router expert selection distributions
2. Measure KL divergence between BF16 and FP32 routing decisions  
3. Validate against SGlang or HuggingFace reference (if available)
4. Run lm-eval benchmarks to detect quality degradation

**Impact:** 
- FP32 default: Safe, validated pattern, slight compute overhead
- BF16 experimental: Faster but may produce subtly incorrect outputs due to expert mis-routing

**Recommendation:** Use default FP32 for production. Only enable BF16 for experimentation with careful validation.

**Tracking:** Issue #39 (Phase 9 numerical validation will determine if BF16 is safe)

---

### 2. CUDAGraph Optimization Disabled (Issue #40)

**Status:** CUDAGraph support explicitly disabled in Phase 7 MVP.

**Reason:** LLaDA2.0's dual-chunk attention creates runtime-dependent metadata (heterogeneous prefix lengths), which breaks CUDAGraph's static graph assumption.

**Performance Impact:** ~10-15% higher inter-token latency (ITL) vs CUDAGraph-enabled models:
- Most vLLM models achieve 10-30% ITL reduction via CUDAGraph
- LLaDA2.0 currently recreates attention metadata on every forward pass
- Decode phase could benefit most from static CUDAGraph optimization

**Why disabled:**
1. Forward-time metadata creation has dynamic shapes (per-request prefix lengths)
2. CUDAGraph requires static metadata and static shapes
3. Upstream chunked_local_attention also disables CUDAGraph for similar reasons

**Comparison to other models:**
- Mixtral: ✅ CUDAGraph supported
- Qwen2-MoE: ✅ CUDAGraph supported
- DeepSeek V3: ✅ CUDAGraph supported
- LLaDA2.0: ❌ Disabled due to virtual batch metadata

**Investigation Plan (Post-MVP):**
1. Explore build-time metadata approach (vs current forward-time)
2. Evaluate static metadata for decode phase (prefix lengths fixed after prefill)
3. Consider two-path optimization: CUDAGraph for decode, dynamic for prefill

**Workaround:** Use torch.compile (if beneficial) or accept ITL overhead.

**Long-term Fix:** Tracked in issue #40 for Phase 8.4+ optimization.

**Impact:** Production workloads experience 10-15% slower streaming responses (higher ITL) compared to CUDAGraph-enabled models.

---

### 3. torch.compile Infrastructure - No Performance Benefit

**Status:** Phase 8 delivers torch.compile **integration** via `@support_torch_compile` decorator, but A/B benchmarks show **no measurable performance improvement** for current configuration.

**A/B Benchmark Results (LLaDA2.0-mini, A100-40GB):**

| Metric | Baseline (OFF) | torch.compile (ON) | Delta |
|--------|----------------|-------------------|-------|
| Output Tokens/sec | 179.1 | 177.8 | **-0.7%** (regression) |
| TTFT | 1753.4 ms | 1713.0 ms | -2.3% (within noise) |
| ITL | 3.9 ms | 3.9 ms | 0.0% (no change) |

**Why no benefit:**
1. Small model size (LLaDA2.0-mini) - limited optimization opportunity
2. Eager execution mode (`--enforce-eager`) disables CUDAGraph
3. Single-request batching (`max_num_seqs=1`) - no parallelism
4. MoE compute already optimal with TRITON kernels

**What Phase 8 delivered:**
- ✅ Infrastructure: `@support_torch_compile` decorator properly integrated
- ✅ Follows vLLM best practices for compilation
- ✅ Foundation for future optimization on larger models
- ❌ **NOT** a performance optimization for current configuration

**Re-evaluation Plan:**
- Larger models: LLaDA2.0-medium/large where compilation may help
- Multi-request batching: `max_num_seqs > 1` (Phase 7.1 supports this)
- Alternative backends: CUTLASS FusedMoE (Phase 8.3)

**Recommendation:** Operators should **not** expect performance gains from torch.compile in Phase 7+8. This is infrastructure work for future optimization.

**Tracking:** Future phases will re-evaluate compilation benefits.

---

## P1 - Important Limitations (May Affect Advanced Usage)

### 4. Pipeline Parallelism (PP > 1) Not Supported

**Status:** Explicitly disabled with fail-fast validation.

**Reason:** LLaDA2.0's dual-chunk attention and MoE routing require custom PP split logic not implemented in Phase 7 MVP.

**Workaround:** Use Tensor Parallelism (`--tensor-parallel-size`) for multi-GPU scaling.

**Error Message:**
```
ValueError: Pipeline parallelism (PP > 1) is not supported for LLaDA2ForCausalLM 
in Phase 7 MVP. Use tensor parallelism (--tensor-parallel-size) for multi-GPU 
inference. PP support may be added in a future phase.
```

**Impact:** Multi-node inference requires TP, not PP. TP=4 tested and working.

---

### 5. Tensor Parallelism (TP > 1) Supported (Phase 8.2)

**Status:** ✅ TP > 1 supported via per-expert weight loading (as of Phase 8.2).

**Implementation:** Expert weights loaded individually with `expert_id` parameter, enabling vLLM's standard TP sharding hooks to distribute experts across ranks.

**Pattern:** Matches Qwen2-MoE, Mixtral, and DeepSeek V3 implementations.

**Implementation Details:**
```python
# Per-expert loading with expert_id parameter (Phase 8.2)
for expert_id in range(self.num_experts):
    # ... prepare expert weights ...
    weight_loader_w13(param_w13, w13_weight, expert_id=expert_id)
    weight_loader_w2(param_w2, down_weight, expert_id=expert_id)
```

**Tested configurations:**
- ✅ TP=1 (single GPU)
- ✅ TP=2 (dual GPU - validated in integration tests)
- ✅ TP=4 (quad GPU - validated in integration tests)
- ⚠️ TP=8+ (not tested but should work with even expert distribution)

**Limitations:**
- Expert count should divide evenly by TP size for optimal distribution
- Recommended TP sizes for 256 experts: TP=1, 2, 4, 8, 16, 32, 64, 128, 256
- Uneven distributions (e.g., TP=3 with 256 experts) will log warning but still work

**Example warning for uneven distribution:**
```
WARNING: TP size 3 does not evenly divide 256 experts. Expert distribution may be unbalanced across ranks.
```

**Multi-GPU setup:**
```bash
# Start vLLM with TP=2
uv run python -m vllm.entrypoints.openai.api_server \
    --model inclusionAI/LLaDA2.0-mini \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --port 8000 \
    --trust-remote-code
```

**Tracking:** TP > 1 support completed in Phase 8.2 (PR #38). Further validation on larger models tracked in future phases.

---

## P2 - Minor Limitations (Documentation/Optimization)

### 6. First-Block Generation Moved to Model Runner

**Status:** First-block draft initialization moved from scheduler to model runner in Phase 7.

**Trade-off:** 
- ✅ Aligns with one interpretation of vLLM pattern
- ⚠️ Makes model runner stateful (stores `_dllm_first_block_requests` dict)
- ⚠️ Inconsistent with scheduler statelessness principle

**Impact:** Low - works correctly but architectural clarity could improve.

**Future:** May revert to scheduler-based initialization if architectural review prefers stateless runner.

---

### 7. GPU Capability Detection - Unused in Phase 7

**Status:** Comprehensive GPU capability detection (`dllm_plugin/gpu_capability.py`) implemented but not used.

**Reason:** Phase 8.2+ will use for CUTLASS/FP8 backend selection. Phase 7 uses default backends.

**Impact:** None - dead code but prepares for future optimization.

---

## Validation Deferred to Phase 9

The following validations are **intentionally deferred** to Phase 9 (issue #39):

1. **MoE routing numerical correctness** (see limitation #1 above)
2. **lm-eval benchmark scores** vs HuggingFace baseline
3. **SGlang reference comparison** (if available)
4. **Group-limited routing load balance** analysis
5. **Routed scaling factor (2.5x)** validation against paper spec

**Rationale:** Phase 7 establishes functional implementation with integration tests. Phase 9 validates numerical correctness and quality.

---

## How to Report Issues

If you encounter behavior related to these limitations:

1. **Check this document** to see if it's a known limitation
2. **Check issue tracker** for existing tracking issues (#39, #40, etc.)
3. **File new issue** if limitation causes production impact not documented here
4. **Include:**
   - Which limitation from this doc
   - Your configuration (model size, TP, batch size)
   - Observed vs expected behavior
   - Logs/error messages

---

**Maintained by:** dLLM Plugin Team  
**Review Frequency:** Updated each phase milestone  
**Last Review:** Phase 7+8 (2026-05-09)
