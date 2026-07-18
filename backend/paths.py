"""Resolve project / packaged data paths (dev repo vs PyInstaller onedir)."""

from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """Directory that contains `data/` and (in prod) `ui/dist`.

    - Dev: real_stream_SDAnime/
    - PyInstaller onedir: exe folder or `_internal` (PyInstaller 6+)
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
        for candidate in (exe_dir, exe_dir / "_internal", meipass):
            if (candidate / "data").is_dir() or (candidate / "ui" / "dist").is_dir():
                return candidate
        return meipass
    # backend/paths.py -> backend/ -> real_stream_SDAnime/
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    return app_root() / "data"


def models_dir() -> Path:
    return data_dir() / "models"


def refs_dir() -> Path:
    return data_dir() / "refs"


def outputs_dir() -> Path:
    d = app_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ui_dist_dir() -> Path:
    """Return packaged Vite output.

    PyInstaller 6 puts datas under ``_internal/``, while the build script may
    also copy ``data/`` next to the exe. Prefer any layout that actually has
    ``ui/dist/index.html`` so the webview is not a bare API 404.
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
        for candidate in (exe_dir, exe_dir / "_internal", meipass, app_root()):
            dist = candidate / "ui" / "dist"
            if (dist / "index.html").is_file():
                return dist
    return app_root() / "ui" / "dist"


def model_path() -> Path:
    return models_dir() / "AnythingV5V3_v5PrtRE.safetensors"


def ip_adapter_dir() -> Path:
    return models_dir() / "ip-adapter"


def ckpt_dir() -> Path:
    return models_dir() / "finetuned" / "checkpoints"


def param_stats_path() -> Path:
    return models_dir() / "finetuned" / "param_stats.json"


def default_ref_path() -> Path:
    return refs_dir() / "train_char_1.png"
