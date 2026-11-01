#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
command -v oc >/dev/null || { echo "oc CLI is required" >&2; exit 1; }
namespace="$(oc project -q)"
if [[ ! "$namespace" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  echo "Choose an OpenShift project before running this script." >&2
  exit 1
fi
if oc -n "$namespace" get job gemma4-banking77 >/dev/null 2>&1; then
  echo "Job gemma4-banking77 already exists. Preserve its results or delete it deliberately before rerunning." >&2
  exit 1
fi

# Send only the image inputs. A binary build from the whole repository could
# upload hidden files, caches, or local credentials that are not in the image.
build_context="$(mktemp -d)"
trap 'rm -r -- "$build_context"' EXIT
mkdir -p "$build_context/banking_lab"
cp Dockerfile requirements-gpu.txt "$build_context/"
cp banking_lab/*.py "$build_context/banking_lab/"
if ! oc -n "$namespace" get buildconfig gemma4-banking77 >/dev/null 2>&1; then
  oc -n "$namespace" new-build --binary --strategy=docker --name=gemma4-banking77
fi
oc -n "$namespace" start-build gemma4-banking77 --from-dir="$build_context" --follow --wait
oc -n "$namespace" apply -f deploy/artifacts-pvc.yaml
sed "s/__NAMESPACE__/$namespace/g" deploy/train-job.yaml | oc -n "$namespace" apply -f -
echo "Submitted Job in $namespace. Follow it with:"
echo "  oc -n $namespace logs -f job/gemma4-banking77"
echo "  oc -n $namespace get job gemma4-banking77"
