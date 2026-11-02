#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
command -v python3.11 >/dev/null || { echo "Python 3.11 is required in Dev Spaces" >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  python3.11 -m venv .venv
fi
.venv/bin/python -m pip install --no-cache-dir -r requirements-dev.txt
echo "Ready: .venv/bin/python -m banking_lab.prepare --out runs/data"
