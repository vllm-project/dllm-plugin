#!/bin/bash
# Deploy LLaDA2 dev pod on Kubernetes with A100 GPU
# Phase 7: GPU testing environment

set -e

POD_NAME="${POD_NAME:-llada2-dev}"
NAMESPACE="${NAMESPACE:-default}"

echo "Deploying LLaDA2 dev pod..."
echo "  Pod name: $POD_NAME"
echo "  Namespace: $NAMESPACE"

# Create pod manifest
cat > /tmp/llada2-dev-pod.yaml <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  labels:
    app: llada2-dev
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 7200  # 2 hour TTL
  containers:
  - name: dev
    image: vllm/vllm-openai:v0.20.1
    command: ["/bin/bash", "-c", "sleep 7200"]
    resources:
      requests:
        nvidia.com/gpu: 1
        memory: "40Gi"
        cpu: "8"
      limits:
        nvidia.com/gpu: 1
        memory: "40Gi"
        cpu: "8"
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 20Gi
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-a100
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  - key: jounce.io/nodetype
    operator: Equal
    value: "A100-40"
    effect: NoSchedule
EOF

# Apply pod
kubectl apply -f /tmp/llada2-dev-pod.yaml -n "$NAMESPACE"

echo "Waiting for pod to be ready..."
kubectl wait --for=condition=Ready pod/$POD_NAME -n "$NAMESPACE" --timeout=300s

echo "Pod ready!"
echo ""
echo "To copy plugin to pod:"
echo "  ./scripts/copy_plugin_to_pod.sh"
echo ""
echo "To start server on pod:"
echo "  kubectl exec -n $NAMESPACE $POD_NAME -- bash -c 'cd /tmp/dllm && VLLM_PLUGINS=dllm VLLM_USE_V2_MODEL_RUNNER=1 vllm serve inclusionAI/LLaDA2.0-mini --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.85 --trust-remote-code --scheduler-cls dllm_plugin.runtime_scheduler.DllmRuntimeScheduler --worker-cls dllm_plugin.runtime_worker.DllmRuntimeWorker'"
echo ""
echo "To set up port forwarding:"
echo "  kubectl port-forward -n $NAMESPACE $POD_NAME 8000:8000"
