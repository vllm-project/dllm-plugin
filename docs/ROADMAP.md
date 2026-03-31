# dLLM plugin — roadmap

Phased work after the **repository skeleton**. Each item should link to **public** trackers (GitHub issues in [`vllm-project/dllm-plugin`](https://github.com/vllm-project/dllm-plugin)), Hugging Face artifacts, or papers—not private planning docs.

| Phase | Focus | Public trackers | Status cue |
|-------|--------|-----------------|------------|
| **Skeleton** | Importable package, `register_dllm()` stub, CI, docs (current milestone). | [#18](https://github.com/vllm-project/dllm-plugin/issues/18), [#2](https://github.com/vllm-project/dllm-plugin/issues/2), [#20](https://github.com/vllm-project/dllm-plugin/issues/20), [#19](https://github.com/vllm-project/dllm-plugin/issues/19) | active |
| **Core hook + plugin stack** | `DllmScheduler` / `DllmWorker`, validation, integration with the vLLM draft-token hook from [vllm#36155](https://github.com/vllm-project/vllm/issues/36155). | [#8](https://github.com/vllm-project/dllm-plugin/issues/8), [#9](https://github.com/vllm-project/dllm-plugin/issues/9), [#10](https://github.com/vllm-project/dllm-plugin/issues/10), [#4](https://github.com/vllm-project/dllm-plugin/issues/4), [#2](https://github.com/vllm-project/dllm-plugin/issues/2) | active |
| **LLaDA2.0** | End-to-end model module, default remasking, first operator docs. | [#3](https://github.com/vllm-project/dllm-plugin/issues/3), [#5](https://github.com/vllm-project/dllm-plugin/issues/5), [#6](https://github.com/vllm-project/dllm-plugin/issues/6), [#7](https://github.com/vllm-project/dllm-plugin/issues/7), [#11](https://github.com/vllm-project/dllm-plugin/issues/11), [#12](https://github.com/vllm-project/dllm-plugin/issues/12), [#13](https://github.com/vllm-project/dllm-plugin/issues/13), [#14](https://github.com/vllm-project/dllm-plugin/issues/14), [#16](https://github.com/vllm-project/dllm-plugin/issues/16), [#17](https://github.com/vllm-project/dllm-plugin/issues/17) | active |
| **Grammar / structured output** | Safer interaction with structured output and async paths; scheduler overrides. | [#9](https://github.com/vllm-project/dllm-plugin/issues/9) (MVP guardrails only; broader work deferred) | backlog |
| **Kernel optimizations** | Prefer **virtual sub-requests** with FlashAttention non-causal (and causal) chunks—same decomposition pattern as [chunked local attention](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/chunked_local_attention.py) in core vLLM—before custom CUDA; see [DESIGN_MVP.md §9](DESIGN_MVP.md#9-attention-and-execution-mvp). | [#11](https://github.com/vllm-project/dllm-plugin/issues/11) (attention spike) | backlog |
| **Additional architectures** | WeDLM, SDAR, Fast-dLLMv2, and related public models—behind clear capability flags. | No dedicated issues yet (create when entering this phase) | backlog |
| **Prefix caching / semi-causal masks** | Cache behavior under block masks; interacts with how virtual chunks share KV; document limitations. | No dedicated issues yet (create when entering this phase) | backlog |
| **Benchmarking & reports** | Reproducible scripts, comparison to AR baselines, published numbers. | No dedicated issues yet (create when entering this phase) | backlog |

Status cue convention: `active` means current milestone implementation focus; `backlog` means tracked future work that should be pruned/refreshed as scope changes.

Contributions should follow [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/) norms (sign-off, PR hygiene) and this repo’s [CONTRIBUTING.md](../CONTRIBUTING.md).
