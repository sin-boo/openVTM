#!/usr/bin/env bash
# One-shot Linux server setup + start (Vast / cloud GPU).
# - Creates .venv if missing
# - Installs Python deps (CUDA torch when GPU is present)
# - Downloads models from Hugging Face into data/models/
# - Starts API in server mode on 0.0.0.0:8765
set -euo pipefail
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")"

export HF_REPO="${HF_REPO:-sinBoo1/models-VT-prototype}"
HOST="${SDANIME_HOST:-0.0.0.0}"
PORT="${SDANIME_PORT:-8765}"
VENV_DIR="${VENV_DIR:-.venv}"

export SDANIME_SERVER_MODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

echo "==> SDAnime Pose server setup (Linux)"
echo "    repo root: $(pwd)"
echo "    HF models: ${HF_REPO}"
echo "    bind: ${HOST}:${PORT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ and retry."
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel setuptools

echo "==> Installing PyTorch"
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio \
    || pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio \
    || pip install torch
else
  echo "WARNING: nvidia-smi not found — installing CPU torch (slow / may be unusable)."
  pip install torch
fi

echo "==> Installing requirements.txt"
pip install -r requirements.txt

mkdir -p data/models data/refs outputs

NEED_DOWNLOAD=0
if [[ ! -f data/models/AnythingV5V3_v5PrtRE.safetensors ]]; then
  NEED_DOWNLOAD=1
fi
if [[ ! -f data/models/ip-adapter/models/ip-adapter_sd15.bin ]]; then
  NEED_DOWNLOAD=1
fi
if [[ ! -f data/models/finetuned/checkpoints/pose_adapter_step_020000.pt ]] \
   && [[ ! -f data/models/finetuned/checkpoints/pose_adapter_latest.pt ]]; then
  NEED_DOWNLOAD=1
fi

if [[ "${NEED_DOWNLOAD}" -eq 1 ]]; then
  echo "==> Downloading models from Hugging Face (${HF_REPO})"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  fi
  python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo = os.environ.get("HF_REPO", "sinBoo1/models-VT-prototype")
dest = Path("data/models")
dest.mkdir(parents=True, exist_ok=True)
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
print(f"snapshot_download {repo} -> {dest.resolve()}")
snapshot_download(
    repo_id=repo,
    repo_type="model",
    local_dir=str(dest),
    token=token,
)
print("download complete")
PY
else
  echo "==> Models already present under data/models/ — skip download"
fi

if [[ ! -f data/models/AnythingV5V3_v5PrtRE.safetensors ]]; then
  echo "ERROR: Anything V5 missing after download."
  echo "       Check HF repo access (public or set HF_TOKEN) and retry."
  exit 1
fi

echo "==> Starting API (server mode) on ${HOST}:${PORT}"
echo "    Health: http://127.0.0.1:${PORT}/api/health"
echo "    On Vast, use the mapped public IP:port for 8765 (see Open Ports)."
if [[ -n "${BROKER_URL:-}" && -n "${PUBLIC_URL:-}" && -n "${BROKER_SECRET:-}" ]]; then
  echo "    Broker: ${BROKER_URL}  PUBLIC_URL=${PUBLIC_URL}"
else
  echo "    Broker: off (set BROKER_URL, BROKER_SECRET, PUBLIC_URL to register)"
fi
exec python -m backend --ui none --server-mode --host "${HOST}" --port "${PORT}"
