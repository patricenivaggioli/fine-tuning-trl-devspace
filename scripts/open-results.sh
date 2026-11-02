#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
command -v oc >/dev/null || { echo "oc CLI is required" >&2; exit 1; }
namespace="${1:-}"
if [[ -z "$namespace" ]]; then
  namespace="$(oc project -q 2>/dev/null || true)"
fi
if [[ ! "$namespace" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  echo "Pass the GPU project name or select one with oc project." >&2
  exit 1
fi
if [[ "$(oc -n "$namespace" get job gemma4-banking77 -o jsonpath='{.status.succeeded}')" != "1" ]]; then
  echo "The training Job has not completed successfully." >&2
  exit 1
fi
sed "s/__NAMESPACE__/$namespace/g" deploy/results-pod.yaml | oc -n "$namespace" apply -f -
oc -n "$namespace" wait --for=condition=Ready pod/gemma4-results --timeout=10m
mkdir -p runs
oc -n "$namespace" cp gemma4-results:/work/report.json runs/report.json
oc -n "$namespace" cp gemma4-results:/work/report.md runs/report.md
oc -n "$namespace" cp gemma4-results:/work/train/train_summary.json runs/train_summary.json
echo "Reports copied to runs/. TensorBoard:"
echo "  oc -n $namespace port-forward pod/gemma4-results 6006:6006"
echo "When finished, remove the reader pod with:"
echo "  oc -n $namespace delete pod gemma4-results"
