# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Drowsiness Detection (CPU torch).

Build:
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  pyinstaller setup.spec --clean --noconfirm
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

datas = [
    ('checkpoints/best.pth', 'checkpoints'),
    ('checkpoints/meta.json', 'checkpoints'),
    ('assets/alert.mp3', 'assets'),
    ('assets/icon.ico', 'assets'),
    ('assets/figures/Training_Validation_Accuracy.png', 'assets/figures'),
    ('assets/figures/Training_Loss.png', 'assets/figures'),
    ('assets/figures/Confusion_Matrix.png', 'assets/figures'),
    ('assets/figures/ROC_Curve.png', 'assets/figures'),
    ('assets/figures/Precision_Recall_Curve.png', 'assets/figures'),
]
binaries = []
hiddenimports = [
    'timm',
    'torchvision',
    'customtkinter',
    'PIL',
    'pygame',
    'platformdirs',
    'pkg_resources',
    'app',
    'app.gui',
    'app.gui.window',
    'app.model',
    'app.inference',
    'app.face',
    'app.stats',
    'app.camera',
    'app.audio',
    'app.config',
    'app.resources',
]

for pkg in ('timm', 'customtkinter', 'torchvision', 'platformdirs'):
    try:
        pkg_datas, pkg_bins, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_bins
        hiddenimports += pkg_hidden
    except Exception:
        pass

try:
    datas += collect_data_files('cv2')
except Exception:
    pass

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter.test', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DrowsinessDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/icon.ico'],
)
