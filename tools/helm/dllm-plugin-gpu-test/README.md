# Helm chart: GPU integration job (template)

This chart runs `tests/test_vllm_mock_integration.py` in a CUDA container. **Default values are cluster-portable:** only a standard GPU toleration is applied; **node selectors and extra tolerations are empty** until you set them for your fleet.

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

If GPU nodes use a **dedicated pool taint** (e.g. `jounce.io/nodetype=L4:NoSchedule` alongside `nvidia.com/gpu`), add a matching **toleration** under `scheduling.extraTolerations` or the pod stays `Pending` despite correct `nodeSelector`. Confirm with `kubectl describe node <gpu-node> | grep -A2 Taints`.

See also `docs/OPERATOR_LLaDA2.md` (Helm subsection).
