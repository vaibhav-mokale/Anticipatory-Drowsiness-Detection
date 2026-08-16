#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Drowsiness Detection setup"

if [[ ! -f checkpoints/best.pth ]]; then
  echo "ERROR: checkpoints/best.pth missing. Train via 01_Train.ipynb first."
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

echo "==> Setup complete. Starting app..."
python -m app "$@"
