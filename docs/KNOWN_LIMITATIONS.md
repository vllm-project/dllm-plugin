# Known Limitations

**Last Updated:** 2026-05-18  
**Scope:** LLaDA2.0 dllm-plugin (Phases 7-9)

This document tracks known limitations, unvalidated assumptions, and deferred work.

---

## CUDA Graph Support

CUDAGraph mode is `UNIFORM_BATCH` — the model forward (embedding → layers →
lm_head) is graph-captured, while attention metadata (virtual batch, slot
mapping remap) runs in eager mode per step. This provides partial graph
benefits but not full capture.

**Remaining blockers for full graph capture:**
- `prepare_inputs()` creates `torch.tensor()` from Python lists (prompt tail
  injection on the first block). This is outside the graph region but adds
  CPU→GPU sync.
- The `DraftTokenIds` scheduler interface requires `tolist()` per step (one
  GPU→CPU sync). Eliminating this requires fork changes to accept GPU tensors.

## First-Block Slot Mapping Remap

The first block requires slot mapping remapping to overwrite frozen prefix KV
with values recomputed in the full-block context. This produces different RoPE
positions than continuation positions. Position continuation was tested and
produces degraded output — the frozen prefix KV from a short prefill dominates
attention and the model repeats prompt tokens.

## Multi-Request First-Block Recomputation

Full-block recomputation (`prepare_inputs` position/input_ids override,
`prepare_attn` slot mapping remap) only supports `num_reqs=1`. Multi-request
batches skip first-block recomputation with a logged warning. This affects
new requests in batches with `max_num_seqs > 1`.

---

## P0 - Critical Limitations (User-Facing Impact)

### 1. MoE Router Numerical Precision

**Status:** Resolved. MoE routing is now delegated to vLLM's `FusedMoE`,
which handles router precision internally (FP32 by default, following
DeepSeek V3 and Qwen2-MoE patterns).

The previous `VLLM_LLADA2_BF16_ROUTER` env var has been removed — it was
specific to the manual routing implementation that was replaced by FusedMoE
delegation. FusedMoE uses `scoring_func="sigmoid"` with FP32 precision for
the gate computation, matching the validated pattern from other vLLM MoE
models.

**Tracking:** Issue #42 criterion resolved by FusedMoE delegation

---

### 2. CUDAGraph Support (UNIFORM_BATCH mode)

**Status:** CUDAGraph is supported via `UNIFORM_BATCH` mode. The model
forward pass is graph-captured; all ModelState hooks (`before_step()`,
`prepare_inputs()`, `prepare_attn()`) and `DiffusionSampler` run in eager
mode before/after graph replay.

Block diffusion decode steps have constant batch shape (1 request, 32
tokens), which is compatible with graph replay. Python dicts and
`torch.tensor()` calls in eager hooks do not break graph capture.

**`@support_torch_compile` removal:** The `@support_torch_compile` decorator
was intentionally removed from `LLaDA2ForCausalLM` because block diffusion
requires variable attention metadata per step (slot_mapping remap, seq_lens
override, prefix_lengths) which is incompatible with full `torch.compile`
graph capture. Phase 8 A/B benchmarks showed no measurable performance
benefit. This is aligned with upstream vLLM direction. For reference, dInfer
applies `torch.compile` to `model.forward` and `get_transfer_index_threshold`
separately (`serving.py:370`, `parallel_strategy.py:353`); the plugin's
`UNIFORM_BATCH` CUDAGraph approach is the vLLM-native equivalent.

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

### 5. Tensor Parallelism (TP > 1) Supported with Model Size Considerations (Phase 8.2)

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

**Validated configurations:**
- ✅ TP=1 (single GPU) - Recommended for LLaDA2.0-mini
- ✅ TP=2 (dual GPU - implementation validated, see performance considerations below)
- ✅ TP=4 (quad GPU - implementation validated)
- ⚠️ TP=8+ (not tested but should work with even expert distribution)

**⚠️ IMPORTANT: Model Size Considerations**

TP introduces communication overhead (NCCL all-reduce, cross-GPU synchronization) that only benefits large models where computation >> communication.

**TP=2 Benchmark Results (LLaDA2.0-mini on 2x A100-40GB):**

| Metric | TP=1 | TP=2 | Change |
|--------|------|------|--------|
| Throughput | 0.5 req/s | 0.37 req/s | **-26%** |
| Token throughput | 501 tok/s | 383 tok/s | **-24%** |
| TTFT | 17ms | 507ms | **+2,847%** |
| ITL | 4.0ms | 4.3ms | +8% |

**Why TP=2 is slower for LLaDA2.0-mini:**
- TP communication overhead: ~500ms (NCCL all-reduce)
- LLaDA2.0-mini computation: ~17ms (prefill)
- **Result:** Overhead >> Computation → Negative scaling

**TP Scaling Threshold:**
```
TP beneficial when: Computation Time > 10x Communication Overhead

For ~500ms TP overhead:
  Models >70B parameters typically benefit from TP
  Models <70B parameters: TP overhead exceeds benefits
  
Examples:
  - GPT-3 175B: TP beneficial
  - LLaMA 70B: TP marginally beneficial
  - LLaDA2.0-mini 30GB: Use TP=1
```

**Recommendation:**
- **LLaDA2.0-mini:** Use TP=1 for optimal performance
- **Larger models (>70B):** TP=2/4/8 will show positive scaling
- **Memory constraints:** Use TP even with small models if they don't fit on single GPU

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
# Start vLLM with TP=2 (only recommended for large models)
uv run python -m vllm.entrypoints.openai.api_server \
    --model inclusionAI/LLaDA2.0-mini \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --port 8000 \
    --trust-remote-code
```

**Documentation:**
- **TP=2 validation guide:** [docs/TP2_VALIDATION_GUIDE.md](TP2_VALIDATION_GUIDE.md)
- **TP=2 benchmark results:** [docs/TP2_BENCHMARK_RESULTS.md](TP2_BENCHMARK_RESULTS.md)

**Tracking:** TP > 1 support completed in Phase 8.2 (PR #38). Benchmark results validate TP implementation correctness and expected performance characteristics.

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
