#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
command -v oc >/dev/null || { echo "oc CLI is required" >&2; exit 1; }
namespace="${1:-}"
if [[ -z "$namespace" ]]; then
  namespace="$(oc project -q 2>/dev/null || true)"
fi
if [[ ! "$namespace" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  echo "Pass a GPU project name or select one with oc project. A user-authenticated oc session is required." >&2
  exit 1
fi
for resource in buildconfigs.build.openshift.io builds.build.openshift.io imagestreams.image.openshift.io jobs.batch persistentvolumeclaims; do
  if ! permission="$(oc auth can-i create "$resource" -n "$namespace")"; then
    echo "Could not check permissions in $namespace. Check your OpenShift login and VPN connection." >&2
    exit 1
  fi
  if [[ "$permission" != "yes" ]]; then
    echo "Current oc identity cannot create $resource in $namespace. Use your OpenShift user login, not the Dev Spaces workspace service account." >&2
    exit 1
  fi
done
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
