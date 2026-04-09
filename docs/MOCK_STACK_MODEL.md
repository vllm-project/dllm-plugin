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

Requires `VLLM_PLUGINS=dllm` and a vLLM build that loads general plugins. Default
CI for this repo does **not** install the `vllm` extra; run optional workflows or
`uv sync --group dev --extra vllm` locally to exercise registration and this module.
