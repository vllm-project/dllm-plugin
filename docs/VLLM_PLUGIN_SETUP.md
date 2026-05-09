# vLLM Plugin Setup for dLLM

**Last Updated:** 2026-05-09

## TL;DR - Simple Setup

The dLLM plugin works with vLLM's standard plugin system. Just set the environment variable and use the CLI:

```bash
export VLLM_PLUGINS=dllm  # Entry point name is "dllm" (not "dllm_plugin")
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

vllm serve inclusionAI/LLaDA2.0-mini \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
  --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker
```

That's it! No Python wrappers, no manual registration needed.

### Tensor Parallelism (TP > 1)

For multi-GPU deployments with TP=2 or higher:

```bash
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

vllm serve inclusionAI/LLaDA2.0-mini \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 2 \
  --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
  --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker
```

**Validated configurations:** TP=1, TP=2, TP=4

**For complete TP=2 setup, benchmarking, and validation instructions, see [TP2_VALIDATION_GUIDE.md](TP2_VALIDATION_GUIDE.md).**

---

## How It Works

### Plugin Registration

The dLLM plugin registers with vLLM via setuptools entry points in [pyproject.toml](../pyproject.toml):

```toml
[project.entry-points."vllm.general_plugins"]
dllm = "dllm_plugin:register_dllm"
```

**Key point:** The entry point name is `dllm`, NOT `dllm_plugin`. This is what you set in `VLLM_PLUGINS`.

### Automatic Loading

vLLM automatically calls `load_general_plugins()` when:
1. `make_arg_parser()` is called (during server startup)
2. `EngineArgs.create_engine_config()` is called
3. Model registry is accessed
4. Engine core and workers initialize

This means setting `VLLM_PLUGINS=dllm` is sufficient - the plugin is loaded automatically before the model architecture is resolved.

### Architecture Names Registered

When `register_dllm()` runs, it registers two architecture names:

```python
LLADA2_ARCHITECTURE_NAME = "LLaDA2ForCausalLM"
LLADA2_HF_ARCHITECTURE_NAME = "LLaDA2MoeModelLM"  # From HuggingFace config
```

Both point to `dllm_plugin.models.llada2:LLaDA2ForCausalLM`.

---

## Common Mistakes

### ❌ WRONG: Using `dllm_plugin` as plugin name
```bash
export VLLM_PLUGINS=dllm_plugin  # WRONG - this won't work
```

The entry point name in `pyproject.toml` is `dllm`, not `dllm_plugin`.

### ❌ WRONG: Manual Python wrapper for registration
```python
# NOT NEEDED - vLLM loads plugins automatically
from dllm_plugin import register_dllm
register_dllm()
```

This was the approach taken initially but is unnecessary. vLLM's plugin system handles registration.

### ❌ WRONG: Complex multiprocessing workarounds
The plugin is loaded in all processes (main, engine core, workers) automatically via vLLM's `load_general_plugins()` calls.

---

## Environment Variables

### Required for dLLM

```bash
VLLM_PLUGINS=dllm                      # Load dLLM plugin
VLLM_USE_V2_MODEL_RUNNER=1             # Use V2 model runner (required for dLLM)
VLLM_ENABLE_V1_MULTIPROCESSING=0       # Disable V1 multiprocessing
```

### Optional

```bash
VLLM_DLLM_USE_MOCK_MODEL=1            # Use mock model instead of real LLaDA2 (testing)
VLLM_DLLM_DRAFT_SIZE=32               # Override default block size (default: 32)
VLLM_DLLM_APPLY_ENGINE_CORE_DRAFT_HOOK=1  # Apply EngineCore draft hook (if needed)
```

---

## Kubernetes Deployment

Pod manifest ([tools/k8s/debug-pod-a100-vllm.yaml](../tools/k8s/debug-pod-a100-vllm.yaml)):

```yaml
env:
  - name: VLLM_PLUGINS
    value: dllm  # Correct entry point name
  - name: VLLM_USE_V2_MODEL_RUNNER
    value: "1"
  - name: VLLM_ENABLE_V1_MULTIPROCESSING
    value: "0"
```

Startup script ([tools/k8s/start-vllm-server.sh](../tools/k8s/start-vllm-server.sh)):
```bash
#!/bin/bash
export VLLM_PLUGINS=dllm
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

vllm serve inclusionAI/LLaDA2.0-mini \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler \
  --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker
```

---

## Benchmarking with guidellm

### Structured Outputs via backend-kwargs

To benchmark with structured outputs (e.g., `guided_regex`), use guidellm's `--backend-kwargs` parameter:

```bash
guidellm benchmark run \
  --target http://localhost:8000 \
  --model inclusionAI/LLaDA2.0-mini \
  --profile constant \
  --rate 5 \
  --max-requests 10000 \
  --max-seconds 300 \
  --data "prompt_tokens=500,output_tokens=500,count=10000" \
  --backend-kwargs '{"extras": {"body": {"guided_regex": "[A-Z][a-z]+( [A-Z][a-z]+)*"}}}' \
  --output-path benchmark_results.json
```

**How it works:**
- `extras.body` gets merged into the HTTP request body via `OpenAIHTTPBackend`'s `model_combine()` method
- Any vLLM-compatible parameter can be passed this way (e.g., `temperature`, `top_p`, `guided_json`, etc.)
- See [BENCHMARK_RESULTS_MULTI_REQUEST.md](BENCHMARK_RESULTS_MULTI_REQUEST.md) for full results

### Synthetic Data Generation

guidellm supports synthetic data generation via the `--data` parameter:

```bash
--data "prompt_tokens=500,output_tokens=500,count=10000"
```

This generates synthetic prompts/outputs matching the specified token counts, eliminating the need for custom dataset preparation.

---

## Verification

Check if plugin is loaded correctly:

```bash
export VLLM_PLUGINS=dllm
python3 -c "
from vllm.plugins import load_general_plugins
load_general_plugins()

from vllm import ModelRegistry
print('LLaDA2MoeModelLM registered:', 'LLaDA2MoeModelLM' in ModelRegistry.get_supported_archs())
print('LLaDA2ForCausalLM registered:', 'LLaDA2ForCausalLM' in ModelRegistry.get_supported_archs())
"
```

Expected output:
```
LLaDA2MoeModelLM registered: True
LLaDA2ForCausalLM registered: True
```

---

## Multi-Request Batching

Phase 7.1 enables multi-request batching with heterogeneous prefix lengths. Configure with `--max-num-seqs`:

```bash
--max-num-seqs 32  # Allow up to 32 concurrent requests in a batch
```

**Expected performance improvement:** 5-10x throughput vs `max_num_seqs=1` under load.

See [docs/PHASE7.1_MULTI_REQUEST_VALIDATION.md](PHASE7.1_MULTI_REQUEST_VALIDATION.md) for validation details.

---

## Troubleshooting

### Plugin not loading
1. **Check environment variable is set BEFORE vLLM starts:**
   ```bash
   echo $VLLM_PLUGINS  # Should print "dllm"
   ```

2. **Verify plugin is installed:**
   ```bash
   python3 -c "from importlib.metadata import entry_points; eps = entry_points(group='vllm.general_plugins'); print([ep.name for ep in eps if 'dllm' in ep.name])"
   # Should print: ['dllm']
   ```

3. **Check plugin registration function exists:**
   ```bash
   python3 -c "from dllm_plugin import register_dllm; print('OK')"
   # Should print: OK
   ```

### Architecture not recognized

If vLLM says `Model architectures ['LLaDA2MoeModelLM'] are not supported`:
- Plugin didn't load or `register_dllm()` wasn't called
- Check `VLLM_PLUGINS=dllm` is set (not `dllm_plugin`)
- Verify plugin is installed with `pip install -e .`

### GPU memory issues

```
ValueError: Free memory on device cuda:0 (...) is less than desired GPU memory utilization
```

- Kill old server processes: `pkill -9 python3`
- Clean GPU: `nvidia-smi` and kill listed processes
- Reduce `--gpu-memory-utilization` (try 0.7 or 0.6)

---

## References

- Plugin entry point: [pyproject.toml](../pyproject.toml)
- Registration function: [dllm_plugin/__init__.py](../dllm_plugin/__init__.py)
- vLLM plugin loader: vLLM source `/vllm/plugins/__init__.py`
- Architecture config: [dllm_plugin/config.py](../dllm_plugin/config.py)
- Phase 7 docs: [PHASE7_DESIGN_DECISIONS.md](PHASE7_DESIGN_DECISIONS.md)
- Phase 7.1 validation: [PHASE7.1_MULTI_REQUEST_VALIDATION.md](PHASE7.1_MULTI_REQUEST_VALIDATION.md)
