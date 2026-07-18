# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir spec for SDAnime Pose.

Must ship package metadata (.dist-info). Diffusers/HF check
``importlib.metadata.version("transformers")`` and treat missing metadata as
"not installed", even when the package code is present.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

ROOT = Path(SPECPATH).resolve().parent  # packaging/ -> real_stream_SDAnime/
UI_DIST = ROOT / "ui" / "dist"
DATA = ROOT / "data"

datas = []
binaries = []
hiddenimports = [
    "encodings.idna",
    "encodings.utf_8",
    "encodings.ascii",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "starlette",
    "pydantic",
    "multipart",
    "python_multipart",
    "webview",
    "PIL",
    "backend",
    "backend.api",
    "backend.engine",
    "backend.paths",
    "backend.training",
    "backend.training.pose_adapter",
    "backend.utils",
    "backend.utils.params",
    "backend.tracking",
    "backend.tracking.openseeface",
    "backend.tracking.service",
    "diffusers",
    "diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion",
    "diffusers.schedulers.scheduling_dpmsolver_multistep",
    "transformers",
    "transformers.models.clip",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "tokenizers",
    "torch",
]

# Package metadata required by HF/diffusers availability checks.
for _pkg in (
    "transformers",
    "diffusers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "tokenizers",
    "torch",
    "numpy",
    "Pillow",
    "opencv-python",
    "onnxruntime",
    "regex",
    "tqdm",
    "requests",
    "packaging",
    "filelock",
    "pyyaml",
    "sentencepiece",
):
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        pass

# Collect full runtime trees for the ML stack (code + datas + binaries).
for _pkg in ("transformers", "diffusers", "tokenizers", "accelerate", "safetensors"):
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except Exception:
        pass

if UI_DIST.is_dir():
    datas.append((str(UI_DIST), "ui/dist"))
if (DATA / "refs").is_dir():
    datas.append((str(DATA / "refs"), "data/refs"))
finetuned = DATA / "models" / "finetuned"
if finetuned.is_dir():
    datas.append((str(finetuned), "data/models/finetuned"))

# Bundled OpenSeeFace (local mode). Server mode does not require it at runtime.
osf_dir = ROOT / "tools" / "OpenSeeFace"
if osf_dir.is_dir() and (osf_dir / "facetracker.py").is_file():
    datas.append((str(osf_dir), "tools/OpenSeeFace"))

try:
    datas += collect_data_files("webview")
except Exception:
    pass

a = Analysis(
    [str(ROOT / "backend" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "rth_idna.py")],
    excludes=[
        "tkinter",
        "matplotlib",
        "notebook",
        "IPython",
        "pytest",
        "tensorboard",
        "tensorflow",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SDAnimePose",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SDAnimePose",
)
