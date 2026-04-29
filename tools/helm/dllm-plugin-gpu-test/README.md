# Helm chart: GPU integration job (template)

This chart runs **GPU pytest targets** from `tests.pytestPaths` (see `values.yaml`): by default that includes **mock-stack end-to-end** (`test_vllm_mock_integration.py`), **MRV2 subclass injection + regex structured output** (`test_vllm_gpu_mrv2_monkeypatch_grammar.py`), and **two-phase contract** checks (`test_two_phase_dllm.py`). The Job template always tolerates `nvidia.com/gpu` (NoSchedule). **Default `values.yaml`** also includes the **jounce.io L4 GPU pool** toleration (`jounce.io/nodetype=L4:NoSchedule`); if your nodes are not tainted that way, set `scheduling.extraTolerations: []` or replace with your fleet’s taints. Use `scheduling.nodeSelector` when you need a specific accelerator label.

The job **clones `git.repoUrl` / `git.branch` from GitHub** inside the container; only commits **pushed** to that remote branch are exercised.

## Fork-and-adjust

1. Copy or vendor this chart under your infra repo.
2. Set `scheduling.nodeSelector` / `scheduling.extraTolerations` in `values.yaml` to match **your** accelerator nodes (cloud provider labels, spot tiers, etc.).
3. Override `image`, `git.repoUrl`, `git.branch`, and resource requests as needed.

Example for GKE L4-style scheduling (illustrative—verify against your cluster labels):

```yaml
scheduling:
  nodeSelector:
    cloud.google.com/gke-accelerator: "nvidia-l4"
  extraTolerations: []
```

The default `extraTolerations` target that **jounce.io L4** pattern. If your pool uses a different taint, edit `values.yaml` or the pod stays `Pending`. Confirm with `kubectl describe node <gpu-node> | grep -A2 Taints`.

See also `docs/OPERATOR_LLaDA2.md` (Helm subsection).
