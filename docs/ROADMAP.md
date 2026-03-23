# dLLM plugin — roadmap

Phased work after the **repository skeleton**. Each item should link to **public** trackers (GitHub issues in [`vllm-project/dllm-plugin`](https://github.com/vllm-project/dllm-plugin)), Hugging Face artifacts, or papers—not private planning docs.

| Phase | Focus |
|-------|--------|
| **Skeleton** | Importable package, `register()` stub, CI, docs (current milestone). |
| **Core hook + plugin stack** | `DllmScheduler` / `DllmWorker`, validation, integration with the vLLM draft-token hook from [vllm#36155](https://github.com/vllm-project/vllm/issues/36155). |
| **LLaDA2.0** | End-to-end model module, default remasking, first operator docs. |
| **Grammar / structured output** | Safer interaction with structured output and async paths; scheduler overrides. |
| **Kernel optimizations** | Prefer **virtual sub-requests** with FlashAttention non-causal (and causal) chunks—same decomposition pattern as [chunked local attention](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/chunked_local_attention.py) in core vLLM—before custom CUDA; see [DESIGN_MVP.md §9](DESIGN_MVP.md#9-attention-and-execution-mvp). |
| **Additional architectures** | WeDLM, SDAR, Fast-dLLMv2, and related public models—behind clear capability flags. |
| **Prefix caching / semi-causal masks** | Cache behavior under block masks; interacts with how virtual chunks share KV; document limitations. |
| **Benchmarking & reports** | Reproducible scripts, comparison to AR baselines, published numbers. |

Contributions should follow [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/) norms (sign-off, PR hygiene) and this repo’s [CONTRIBUTING.md](../CONTRIBUTING.md).
