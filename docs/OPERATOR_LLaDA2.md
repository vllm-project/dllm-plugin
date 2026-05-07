# Operator guide (LLaDA2.0 real model, Phase 7)

This guide describes the **LLaDA2.0 real model** operator path for `vllm-dllm-plugin`
with production-ready inference (Phase 7) and mock-stack testing (Phases 5-6).

**Phase 7 update:** Real LLaDA2.0 model with MoE architecture now available.
See [Phase 7: Real Model](#phase-7-real-llada20-model) section below.

## Prerequisites

- Linux/CUDA environment with working GPU (Phase 6 integration test is GPU-gated).
- Plugin repo checked out and synced with vLLM extra:

```bash
uv sync --group dev --extra vllm
```

- vLLM compatibility/minimum tested range follows `pyproject.toml`
  (`vllm>=0.20.0,<0.21`); track pin and hook confidence updates via issue `#2`.

- **PyPI wheels vs companion plumbing:** A matching **`pyproject.toml` pin** does not
  automatically mean every **`dllm_*`** field on **`SchedulerOutput`** or related
  **`EngineCore`** batch paths exist on the wheel you installed. Until maintainers
  confirm parity with stock PyPI builds, treat **GPU / Helm** integration runs as the
  authority for structured-output E2E; merging plugin issues **#9** / **#10** does not
  assert “works on PyPI vLLM alone” without that verification (see **#2**).

- **Companion vLLM PR / commit:** When an upstream (or fork) change set exists for the
  engine / `SchedulerOutput` plumbing, link it from issue [**#2**](https://github.com/vllm-project/dllm-plugin/issues/2)
  and paste the URL into the **active plugin PR** so operators are not left searching
  for the “other half” of the integration.

- **Behavioral tests + EngineCore shim (issue [#35](https://github.com/vllm-project/dllm-plugin/issues/35)):** see [TESTING_DLLM_SEMANTICS.md](TESTING_DLLM_SEMANTICS.md) for pytest markers, CI matrix, and the PR **#36391** test-only patch.

- vLLM plugin loading enabled:

```bash
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
```

### v1 vs v2 model runner (mock-stack path)

| Runner | Mock-stack support | Notes |
|--------|-------------------|--------|
| **v2** (`VLLM_USE_V2_MODEL_RUNNER=1`) | **Supported** | Required for the Phase 6 integration test, CI (`vllm-extra`), and operator docs. Adapters are written against vLLM's **v2 model-runner** hooks (`DllmRuntimeWorker` subclasses vLLM's worker on that stack). |
| **v1** (`VLLM_USE_V2_MODEL_RUNNER=0` or unset) | **Unsupported** | Runtime adapters target the **v2 model-runner** stack only. Expect incorrect scheduler/worker pairing or runtime failures rather than silent corruption; do not use for mock-stack validation. |

There is **no supported fallback** from v2 to v1 for the mock-stack path: if your environment cannot enable the v2 model runner, treat the mock-stack integration test and operator workflow as **not applicable** until v2 is available—do not expect partial correctness on v1.

**Two-phase execution:** On v2, inference does **not** attach final tokens in `execute_model`; the worker’s model runner returns `None` after forward and performs dLLM remasking in **phase two** (`sample_tokens` → `sample`). Structured-output grammar bitmasks arrive on that path (`GrammarOutput`), consistent with AR and vanilla spec-decode. Do **not** enable Eagle (or similar draft-model speculative decoding) together with the dLLM plugin stack for the same requests—the stacks are mutually exclusive for MVP.

Keep `VLLM_ENABLE_V1_MULTIPROCESSING=0` for the documented integration test to avoid multiprocessing differences on single-process bring-up.

**Async scheduling:** Mock-stack integration and GPU grammar tests use **`async_scheduling=False`**. The runner still branches on **`use_async_scheduling`** for dLLM block batches, but **async scheduling + structured outputs + dLLM** has **no CI coverage**—treat it as **unsupported for MVP** until explicit tests exist (milestone [**#19**](https://github.com/vllm-project/dllm-plugin/issues/19) follow-up).

**Pipeline parallelism (PP):** On dLLM architectures, block sampling uses a wider per-request token row than vanilla AR sampling when speculative decoding is off (``DRAFT_SIZE`` vs ``num_speculative_steps + 1``). The plugin aligns ``pp_receive`` / ``pp_broadcast`` tensor widths so ranks agree with ``torch.distributed.broadcast`` shape rules. Multi-rank PP + dLLM is still **lightly exercised** compared to single-GPU mock CI—treat full PP stacks as higher risk until you have your own smoke runs.

### Draft handoff naming (#10)

The engine still calls the worker’s **`take_draft_token_ids()`**. On the v2 stack,
:class:`~dllm_plugin.gpu_model_runner.DllmGPUModelRunner` exposes
**`take_dllm_draft_token_ids()`** (not the upstream runner hook name used by Eagle-style
spec decode) so dLLM **`DraftTokenIds`** stay semantically separate from vanilla
speculative drafts. :class:`~dllm_plugin.runtime_worker.DllmRuntimeWorker` prefers that
method when present, then falls back to **`super().take_draft_token_ids()`**.

### Strict stack validation toggle

Issue [#4](https://github.com/vllm-project/dllm-plugin/issues/4) validation is **on by default**. To disable it when `strict` is left unset (e.g. temporary debugging), set:

```bash
export VLLM_DLLM_STRICT_STACK_VALIDATION=0
```

Prefer fixing scheduler/worker/model wiring; disabling checks can mask misconfiguration. Explicit `strict=True` / `strict=False` in code overrides the environment.

## Required runtime wiring

Use runtime adapters via CLI overrides:

```bash
vllm serve <model> \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker
```

Short aliases (`dllm_plugin.Scheduler` / `dllm_plugin.Worker`) match `DllmRuntimeScheduler` / `DllmRuntimeWorker`. vLLM class resolution expects **dotted** names (`module.Class`), not `module:Class`.

Strict stack validation (`dllm_plugin.validation.assert_compatible_stack`)
fails fast when scheduler/worker/model architecture combinations are incompatible.

Checks resolve scheduler and worker types and compare them to the **concrete**
adapter classes (`DllmRuntimeScheduler`, `DllmRuntimeWorker`) by fully-qualified
name. **Subclasses** of those adapters are rejected until validation is relaxed
or extended—by design for MVP mock-stack gatekeeping; forks should adjust
validation if they introduce subclassed workers/schedulers.

When strict validation is **on** (default), constructing `DllmRuntimeWorker` also
runs `assert_runtime_worker_v2_model_runner`: if the v2 model runner is not enabled
(`VLLM_USE_V2_MODEL_RUNNER=1`), startup raises **ValueError** (issue [**#10**](https://github.com/vllm-project/dllm-plugin/issues/10)).
With strict off, the same mismatch emits a **warning** instead.

## EngineCore draft hook (runtime) and HTTP smoke

On some PyPI **vLLM 0.20.x** wheels, `EngineCore` still ties the draft-token hook to
speculative decoding until [vLLM PR #36391](https://github.com/vllm-project/vllm/pull/36391)
ships in your build. For **`vllm serve`** and other engine processes, set:

```bash
export VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK=1
```

so `register_dllm()` applies the same **string-fragile** runtime patch described in
`docs/CONTRACTS.md` and `dllm_plugin.engine_core_draft_hook`. Track pins and upstream
in [issue #2](https://github.com/vllm-project/dllm-plugin/issues/2). Disable **all**
patching (tests or runtime) with `VLLM_DLLM_SKIP_ENGINE_CORE_DRAFT_HOOK_PATCH=1`.

**HTTP smoke (mock stack, GPU):** from the plugin repository root, after
`uv sync --group dev --extra vllm`:

```bash
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
bash tools/e2e/serve_http_smoke.sh
```

The script starts `vllm serve` on `127.0.0.1` (port `8765` by default, overridable via
`VLLM_DLLM_HTTP_SMOKE_PORT`), waits for `/health`, posts to `/v1/chat/completions` with
`curl`, asserts a JSON `choices` field, then stops the server. It sets
`VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK=1` internally for legacy wheels. The Helm chart
`tools/helm/dllm-plugin-gpu-test` runs this script after pytest when
`tests.runServeHttpSmoke` is `true` (default).

## First block initialization

- The scheduler initializes `Request.spec_token_ids` for new requests when empty.
- Block size is global: `dllm_plugin.config.DRAFT_SIZE`.
- Override block size consistently across scheduler/worker/remasking by setting
  `VLLM_DLLM_DRAFT_SIZE` **before importing** plugin modules.

## Integration test (mock stack)

Run the concrete vLLM runtime integration test:

```bash
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
uv run pytest -q tests/test_vllm_mock_integration.py
```

Expected behavior:

- **CPU / GitHub-hosted CI:** `tests/test_vllm_mock_integration_cpu_smoke.py` runs in the
  default `vllm-extra` job: it builds `EngineArgs`, calls `create_engine_config`, resolves
  scheduler/worker classes, and runs `assert_compatible_stack` (no CUDA required). This
  catches upstream API or qualname drift even when the GPU test is skipped.
- **Non-GPU hosts:** the full `LLM.generate` test in `test_vllm_mock_integration.py` is
  skipped (`requires CUDA GPU`).
- **GPU hosts:** that test executes one end-to-end step through vLLM runtime objects with
  plugin scheduler/worker adapters and mock model config.
- **Depth:** the GPU test asserts `LLM.generate` returns token ids; it does **not**
  check remasking block shape, `dllm_block_logits` consumption, or other remask
  invariants—tighter assertions are optional follow-ups beyond this Phase 6 smoke.

For PR/release evidence, include GPU job status plus a persisted log query
(`gcloud logging read ... labels.\"k8s-pod/job-name\"=\"<job-name>\"`) so the
integration result remains auditable after pod cleanup.

## Helm GPU job (`tools/helm/dllm-plugin-gpu-test`)

The chart defaults include tolerations for **`nvidia.com/gpu`** (in the Job template) and
the **jounce.io L4** GPU pool (`scheduling.extraTolerations` in `values.yaml`). If your
cluster does not use those taints, clear or replace them (see
`tools/helm/dllm-plugin-gpu-test/README.md`).

The Job runs **`tests.pytestPaths`** from `values.yaml`, including **mock-stack GPU smoke**,
**`DllmGPUModelRunner` monkeypatch + regex structured output**, and **two-phase MRV2**
contract tests—override `tests.pytestPaths` if you need a narrower run. When
**`tests.runServeHttpSmoke`** is enabled (default), the Job also runs
**`tools/e2e/serve_http_smoke.sh`** after pytest.

## Notes

- This guide covers **mock-stack** MVP only (Phases 2-6), not real LLaDA2 weights.

### Structured outputs (Phase 4 / issues [#9](https://github.com/vllm-project/dllm-plugin/issues/9), [#10](https://github.com/vllm-project/dllm-plugin/issues/10))

- **Mutually exclusive:** Do **not** enable vanilla **speculative decoding** alongside the
  dLLM draft-block path; reuse of spec-decode-shaped fields is for dLLM blocks only (see
  ``docs/DESIGN_MVP.md`` §7).
- **Frontier bitmask:** For structured-output requests, grammar masks apply at the
  **frontier** token (first invalid grammar position in the draft block). Valid-prefix
  bookkeeping follows vLLM’s ``StructuredOutputManager.grammar_bitmask`` semantics;
  fixed ``DRAFT_SIZE`` blocks are preserved — drafts are not grammar-truncated in
  ``update_draft_token_ids``.
- **Repair budget:** ``grammar_extra_transfer`` may increase per-step transfer counts when
  a grammar-invalid tail exists (see ``Llada2DefaultRemaskingPolicy``).
- **Strict frontier-only vs multi-frontier:** Default MVP path masks **one** frontier row
  per step; remasking earlier positions without re-evaluating grammar is out of scope
  unless documented otherwise.
- **vLLM pin:** Precomputed grammar metadata on ``SchedulerOutput`` and relaxed draft-token
  hooks in ``EngineCore`` require the matching **vLLM** revision that includes those
  ``dllm_*`` fields and ``post_step`` / batch-queue updates—coordinate upgrades with
  ``pyproject.toml`` bounds (issue [#2](https://github.com/vllm-project/dllm-plugin/issues/2)).
- **Bitmask buffer sizing:** If ``speculative_config.num_speculative_tokens`` is unset,
  raise it to at least ``DRAFT_SIZE - 1`` when using structured outputs at scale, or rely on
  a vLLM build that extends grammar-bitmask allocation for large dLLM blocks.
- **`num_invalid_spec_tokens`:** The plugin leaves this map empty when refreshing drafts for
  dLLM-shaped batches (documented in scheduler code). Safe today; re-check if upstream begins
  relying on this field for non–spec-decode behavior on mixed batches.
- **Two-stage grammar (GPU + frontier):** vLLM applies the batch grammar bitmask on GPU
  logits; the plugin may apply an additional **frontier-row** mask on CPU-materialized
  block logits before remasking — both target the same frontier semantics (first invalid
  grammar position).
- **Test-only env:** ``VLLM_DLLM_SKIP_FIRST_BLOCK_SEED=1`` skips seeding the first dLLM
  draft block for new requests (used by GPU grammar tests). Do **not** set this in
  production-like deployments. Full first-block seed + regex SO can remain sensitive to
  bitmask row allocation vs draft scheduling until upstream alignment improves (issue **#2**).
- **Async + SO:** Same stance as the v2 runner section above—**not** MVP-validated; keep
  ``async_scheduling=False`` for assurance unless you own the risk.

---

## Phase 7: Real LLaDA2.0 Model

**Issue:** [#12](https://github.com/vllm-project/dllm-plugin/issues/12), [#11](https://github.com/vllm-project/dllm-plugin/issues/11), [#25](https://github.com/vllm-project/dllm-plugin/issues/25)  
**Milestone:** [Phase 7 - Real Model Integration](https://github.com/vllm-project/dllm-plugin/issues/19)

Phase 7 adds production-ready LLaDA2.0 inference with:
- **256-expert MoE** architecture with group-limited routing
- **Block-style non-causal attention** for diffusion-based generation
- **Shared expert** (always active) + routed experts (top-k selected)
- **Tensor parallelism (TP)** support for multi-GPU inference

### Model Selection

By default, `LLADA2_ARCHITECTURE_NAME` now points to the real model:

```bash
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1

# Use real LLaDA2.0 model (default in Phase 7)
vllm serve inclusionAI/LLaDA2.0-mini \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker
```

To use mock model for testing (Phases 2-6 behavior):

```bash
export VLLM_DLLM_USE_MOCK_MODEL=1  # Override to mock
```

### Supported Models

**Phase 7 tested models:**
- `inclusionAI/LLaDA2.0-mini` (smallest, recommended for testing)
- Other LLaDA2.0 family models from [HuggingFace collection](https://huggingface.co/collections/inclusionAI/llada-20)

**Model requirements:**
- HuggingFace config with `architectures` containing `"LLaDA2ForCausalLM"`
- MoE parameters: `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`
- Optional: `num_shared_experts`, `n_group`, `topk_group`, `routed_scaling_factor`

### Multi-GPU Inference

#### Tensor Parallelism (Supported)

LLaDA2.0 supports tensor parallelism for multi-GPU scaling:

```bash
vllm serve inclusionAI/LLaDA2.0-mini \
  --tensor-parallel-size 2 \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker
```

**Validation:**
- TP size must not exceed `num_experts` (256 for LLaDA2.0)
- Expert weights are sharded across TP ranks
- Router/gate weights are replicated (not sharded)

#### Pipeline Parallelism (Not Supported - Phase 7)

Pipeline parallelism (PP > 1) is **not supported** in Phase 7 MVP:

```bash
# This will FAIL:
vllm serve inclusionAI/LLaDA2.0-mini \
  --pipeline-parallel-size 2  # ValueError: PP > 1 not supported
```

**Reason:** Simpler implementation, TP covers most use cases, can be added in future phase.

Use `--tensor-parallel-size` for multi-GPU inference instead.

### Attention Backends

LLaDA2.0 block-style attention works with both:

**FlashAttention** (default):
```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

**FlashInfer** (alternative):
```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
```

Both backends support `is_causal=False` / `causal=False` required for non-causal block attention.

### GPU Requirements

**Minimum:** L4 16GB (for LLaDA2.0-mini)  
**Recommended:** A100-40GB (preferred for testing)  
**Large models:** H100-80GB (spot instances for models >40GB)

Example with explicit memory configuration:

```bash
vllm serve inclusionAI/LLaDA2.0-mini \
  --gpu-memory-utilization 0.9 \
  --max-model-len 256 \
  --enforce-eager \
  --scheduler-cls dllm_plugin.Scheduler \
  --worker-cls dllm_plugin.Worker
```

### Configuration Parameters

**Block size** (DRAFT_SIZE):
```bash
export VLLM_DLLM_DRAFT_SIZE=32  # Default: 32 tokens per block
```

**MoE architecture defaults** (from HF config, can override):
- `num_experts`: 256
- `num_experts_per_tok`: 8
- `num_shared_experts`: 1
- `moe_intermediate_size`: 512
- `n_group`: 8 (expert groups)
- `topk_group`: 4 (groups to select)
- `routed_scaling_factor`: 2.5

### Testing

**Unit tests** (no GPU):
```bash
pytest tests/test_llada2_attention.py
pytest tests/test_llada2_real_model.py
```

**GPU integration tests**:
```bash
# Requires CUDA GPU (A100/L4/H100)
pytest -v -m dllm_gpu_integration tests/test_llada2_gpu_integration.py
```

**HTTP smoke test**:
```bash
./tools/e2e/serve_http_real_model_smoke.sh
```

### Troubleshooting

**Problem:** `ValueError: Pipeline parallelism (PP > 1) not supported`  
**Solution:** Use `--tensor-parallel-size` instead of `--pipeline-parallel-size`

**Problem:** `ValueError: Tensor parallelism size cannot exceed number of experts`  
**Solution:** Reduce `--tensor-parallel-size` to ≤ 256 (num_experts)

**Problem:** Out of memory (OOM) errors  
**Solution:** 
- Use smaller model (LLaDA2.0-mini)
- Reduce `--max-model-len` (e.g., 256 instead of 2048)
- Increase `--gpu-memory-utilization` (e.g., 0.9)

**Problem:** Want to use mock model for testing  
**Solution:** Set `export VLLM_DLLM_USE_MOCK_MODEL=1`

### Design Documentation

- **Attention design:** [docs/ATTENTION_DESIGN.md](ATTENTION_DESIGN.md)
- **MoE architecture:** See vLLM reference implementations:
  - [Mixtral](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mixtral.py)
  - [Qwen2 MoE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py)
  - [DeepSeek V2](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py)

### Known Limitations (Phase 7 MVP)

**Not implemented:**
- **Pipeline parallelism (PP > 1)** - use TP instead
- **Multi-request batching** - Only `max_num_seqs=1` supported for virtual batch attention
  - Heterogeneous prefix lengths across multiple requests not yet supported
  - Server will raise `NotImplementedError` if `num_reqs > 1` is attempted
  - Future work: Phase 7.1 will add multi-request support
- Custom CUDA kernels for attention - using FlashAttention/FlashInfer
- Prefix caching under block-style masks
- Advanced grammar integrations beyond basic support

**Configuration constraints:**
- **KV cache block size:** Defaults to 16 tokens/block (standard vLLM)
  - Currently not queried from `cache_config`, uses default value
  - May break if vLLM changes default block size in future versions
  - Future work: Query from vLLM's cache configuration

**Testing limitations:**
- **Structural validation only** - Integration tests verify API contracts, not output correctness
- **Phase 9 required** - Numerical correctness validation (lm-eval, reference comparison) deferred
- See [dllm-plugin issue #40](https://github.com/vllm-project/dllm-plugin/issues/40) for Phase 9 plan

**Future enhancements** (post-MVP):
- **Phase 7.1:** Multi-request batching with heterogeneous prefix lengths
- Full PP support if use cases emerge
- Optimized attention kernels (fused prefix + block in single pass)
- Sparse/windowed attention for very long contexts
- **Phase 9:** Output correctness validation and reference comparisons

