#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo -e "${BLUE}Starting IDAS setup...${NC}"

IF_DAT="shape_predictor_68_face_landmarks.dat"
if [ ! -f "$IF_DAT" ]; then
    echo -e "${BLUE}Downloading facial landmarks model...${NC}"
    wget -O shape_predictor_68_face_landmarks.dat.bz2 \
        "https://github.com/italojs/facetorch/raw/main/models/shape_predictor_68_face_landmarks.dat.bz2"
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2
else
    echo -e "${GREEN}Facial landmarks model found.${NC}"
fi

echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv .venv

echo -e "${BLUE}Installing dependencies...${NC}"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f assets/figures/ear_mar_monitoring.png ]; then
    echo -e "${BLUE}Generating reference EAR/MAR figure...${NC}"
    python3 - <<'PY'
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out = Path('assets/figures/ear_mar_monitoring.png')
out.parent.mkdir(parents=True, exist_ok=True)
max_hist = 50
ear = [0.30 + 0.02 * np.sin(i / 4) for i in range(max_hist)]
mar = [0.15 + 0.05 * np.cos(i / 5) for i in range(max_hist)]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3.5), dpi=140)
fig.patch.set_facecolor('#FCFAFA')
for ax, data, title, thresh, color in (
    (ax1, ear, 'Eye Aspect Ratio (EAR)', 0.25, '#5E81AC'),
    (ax2, mar, 'Mouth Aspect Ratio (MAR)', 0.60, '#BF616A'),
):
    ax.plot(range(len(data)), data, color=color, linewidth=2)
    ax.axhline(thresh, color='#B42318', linestyle='--', alpha=0.8)
    ax.set_title(title, fontsize=10, fontname='serif')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out, bbox_inches='tight')
PY
fi

echo -e "${GREEN}Setup complete. Run: source .venv/bin/activate && python3 -m app${NC}"
