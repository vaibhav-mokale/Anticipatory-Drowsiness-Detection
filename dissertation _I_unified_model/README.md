# Drowsiness Detection

![dataset used](https://drive.google.com/drive/folders/1wFVS0pCfBcZpz-Eq6Ht9lvM3mdC0_tX1)
Real-time webcam drowsiness detection with a CustomTkinter GUI. Uses a trained hybrid **CNN_LSTM_ViT** checkpoint (`MobileNetV3-Large` + BiLSTM + ViT-Tiny).

Logic and GUI are split: `app/` owns model / camera / audio / stats; `app/gui/` is presentation only. Same entry works as `python -m app` and as a frozen binary via PyInstaller.

Icon / favicon: `assets/icon.ico` (same as the road-hypnosis-detection archive).

## Features

- Live webcam preview with face box overlay
- Face-crop inference (Haar + upper-center ROI fallback)
- Threshold slider, audio alert on sustained drowsiness
- Nerd stats panel: Important / All toggle, scroll, select, Copy
- Export: annotated frame + diagnostics panel PNG, plus JSON sidecar

## Requirements

- Python 3.10+
- Desktop Tk (`python3-tk` on Debian/Ubuntu)
- Webcam
- `checkpoints/best.pth` (produced by `01_Train.ipynb`)

## Quick start (Python)

```bash
cd drowsiness-detection
chmod +x setup.sh
./setup.sh
```

Manual:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m app
```

Windows: `setup.bat`

### Flags

| Flag | Meaning |
|------|---------|
| `--headless` | Load model and exit (smoke check) |
| `--webcam N` | OpenCV camera index (default `0`) |
| `--threshold 0.45` | Drowsy probability cutoff |

## Build standalone binary

Uses the same `assets/icon.ico` as road-hypnosis-detection. Prefer **CPU** torch wheels; CUDA makes a huge bundle.

### GitHub Actions (Windows EXE)

Workflow: [`.github/workflows/build-windows-exe.yml`](.github/workflows/build-windows-exe.yml)

- **Manual:** Actions → `build windows exe` → Run workflow
- **Release:** push a tag like `v1.0.0` (uploads the exe to the GitHub Release)

Artifact name: `DrowsinessDetection-windows` (`DrowsinessDetection.exe`). Friend only needs Windows + webcam, then double-click the exe.

### Local build

```bash
chmod +x build.sh
./build.sh
```

Or:

```bash
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pyinstaller setup.spec --clean --noconfirm
```

Output: `dist/DrowsinessDetection` (Linux) or `dist/DrowsinessDetection.exe` (Windows).

On Linux, PyInstaller ignores `.ico` for the ELF binary icon (Windows/macOS only). The same `assets/icon.ico` is still bundled and used as the window icon at runtime.

Bundled datas: `checkpoints/best.pth`, `checkpoints/meta.json`, `assets/alert.mp3`, `assets/icon.ico`.

Smoke check the freeze:

```bash
./dist/DrowsinessDetection --headless
```

Paths resolve through `app.resources.resource_path` for both source and frozen runs.

## Layout

```
app/                 runtime package (model, inference, face, gui, stats)
assets/              alert.mp3, icon.ico
checkpoints/         best.pth (+ meta.json)
exports/             Export button output (gitignored)
main.py              thin entry for PyInstaller
setup.spec           one-file freeze
setup.sh / setup.bat dev bootstrap + run
build.sh             freeze helper
01_Train.ipynb       train
02_Run.ipynb         legacy notebook inference
03_Evaluation.ipynb  eval
```

## Notes

- Default drowsy threshold is `0.45` (tune in UI).
- Alert fires after consecutive drowsy frames (config: `consecutive_drowsy_frames`).
- Training notebooks and `datasets/` are research artifacts; the shipped app only needs `best.pth`.
