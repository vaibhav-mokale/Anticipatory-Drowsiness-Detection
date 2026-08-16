#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Building DrowsinessDetection (PyInstaller)"

if [[ ! -f checkpoints/best.pth ]]; then
  echo "ERROR: checkpoints/best.pth missing."
  exit 1
fi
if [[ ! -f assets/icon.ico ]]; then
  echo "ERROR: assets/icon.ico missing (same icon as road-hypnosis-detection)."
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

pyinstaller setup.spec --clean --noconfirm

echo "==> Done: dist/DrowsinessDetection"
ls -lh dist/ || true
