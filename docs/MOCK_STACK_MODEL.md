# Mock stack model (Phases 2–6)

**Not production inference.** This path exists so scheduler, worker, remasking,
and validation can integrate **before** real LLaDA2 weights and attention land
(issue #24; real model is issue #12 / Phase 7).

## Registered architecture names

`register_dllm()` adds two keys to vLLM’s ``ModelRegistry`` (same lazy class target):

| Constant | Value | Role |
|----------|-------|------|
| `LLADA2_ARCHITECTURE_NAME` | `LLaDA2ForCausalLM` | Placeholder prod-style name until HF mapping (#12). |
| `DLLM_MOCK_STACK_MODEL_ID` | `DllmMockLlada2StackTest` | Explicit stack-test id. |

Lazy FQCN: `vllm_dllm_plugin.config.DLLM_MOCK_MODEL_CLASS_FQCN` →
`vllm_dllm_plugin.models.mock_llada2:DllmMockLlada2ForCausalLM`.

### Placeholder architecture risk (`LLaDA2ForCausalLM`)

Until Phase 7 ([#12](https://github.com/vllm-project/dllm-plugin/issues/12)), any HF
`config.json` that lists `LLaDA2ForCausalLM` resolves to **this mock** (degenerate
logits on token id `0`), not a real LLaDA2 model. Prefer `DllmMockLlada2StackTest`
in configs when you want an **obviously** non-production id.

If a future vLLM core build registers the same architecture string, load order
wins; this plugin **skips** registration when the name is already present, so
it will **not** override core.

### Pipeline parallel (PP)

The mock branches on `get_pp_group()` for basic staging, but non-last ranks use a
**zero residual** stub and there is **no** `make_empty_intermediate_tensors` (or
related helpers) yet. Treat PP > 1 as **unsupported** for this stub until a
follow-up wires real residuals and PP factory hooks.

## HuggingFace `config.json` surface (minimal)

Your config should list one of the architecture strings above under
`architectures`, and include at least:

```json
{
  "architectures": ["LLaDA2ForCausalLM"],
  "hidden_size": 32,
  "vocab_size": 256,
  "model_type": "llada2_mock"
}
```

Sizes are hints for tensor shapes; the mock does not implement a real transformer.

## Forward / logits contract (for #10 / #13)

- **Last PP stage:** `compute_logits` returns a 2-D float tensor
  `[num_tokens, vocab_size]` with deterministic mass on token index `0`.
- **Non-last PP stage:** `compute_logits` returns `None` (vLLM convention).
- **`forward`:** returns hidden states (or `IntermediateTensors` for PP) with
  hidden width `hidden_size`.

## Operator expectations

Requires `VLLM_PLUGINS=dllm` and a vLLM build that loads general plugins.

**CI:** The default workflow runs lint/tests without the `vllm` extra; a second
job syncs with `--extra vllm` on Python 3.12 and runs the full test suite so
registration and mock import are exercised on PRs. If that job fails (e.g. no
wheel for the runner), use the **Optional vLLM smoke** workflow or
`uv sync --group dev --extra vllm` locally.
