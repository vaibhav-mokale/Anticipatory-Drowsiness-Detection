from __future__ import annotations

import os
import sys


def resource_path(relative_path: str) -> str:
    """Resolve asset path for both `python -m app` and frozen PyInstaller EXE."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base, relative_path)
