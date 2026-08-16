@echo off
setlocal
cd /d "%~dp0"

echo ==^> Drowsiness Detection setup

if not exist "checkpoints\best.pth" (
  echo ERROR: checkpoints\best.pth missing. Train via 01_Train.ipynb first.
  exit /b 1
)

if not exist ".venv" (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

echo ==^> Setup complete. Starting app...
python -m app %*
