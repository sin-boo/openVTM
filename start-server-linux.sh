#!/usr/bin/env bash
# One-shot Linux server setup + start (Vast / cloud GPU).
# - Creates .venv if missing
# - Installs Python deps (CUDA torch when GPU is present)
# - Downloads models from Hugging Face into data/models/
# - Prompts for web-terminal join token (saved in .broker.env)
# - Starts API in server mode on 0.0.0.0:8765
set -euo pipefail
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")"

export HF_REPO="${HF_REPO:-sinBoo1/models-VT-prototype}"
HOST="${SDANIME_HOST:-0.0.0.0}"
PORT="${SDANIME_PORT:-8765}"
VENV_DIR="${VENV_DIR:-.venv}"
BROKER_ENV_FILE="${BROKER_ENV_FILE:-.broker.env}"

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

# --- Web terminal broker config (interactive, saved locally) ---
load_broker_env() {
  if [[ -f "${BROKER_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck disable=SC1091
    source "${BROKER_ENV_FILE}"
    set +a
  fi
}

save_broker_env() {
  umask 077
  cat > "${BROKER_ENV_FILE}" <<EOF
# Local only — do not commit. Used by start-server-linux.sh
BROKER_URL=${BROKER_URL}
BROKER_TOKEN=${BROKER_TOKEN}
PUBLIC_URL=${PUBLIC_URL}
EOF
  echo "==> Saved broker settings to ${BROKER_ENV_FILE}"
}

normalize_token() {
  # Uppercase alphanumeric, max 7 chars
  echo "$1" | tr '[:lower:]' '[:upper:]' | tr -cd 'A-Z0-9' | cut -c1-7
}

prompt_broker_config() {
  load_broker_env

  echo ""
  echo "==> Web terminal (discovery broker)"
  echo "    This registers the GPU so https://webtermial.vercel.app can see it."
  echo ""

  local use_broker="${BROKER_ENABLE:-}"
  if [[ -z "${use_broker}" ]]; then
    if [[ -n "${BROKER_TOKEN:-}" && -n "${PUBLIC_URL:-}" ]]; then
      read -r -p "Use saved broker settings from ${BROKER_ENV_FILE}? [Y/n] " use_broker
      use_broker="${use_broker:-Y}"
    else
      read -r -p "Register with the web terminal? [Y/n] " use_broker
      use_broker="${use_broker:-Y}"
    fi
  fi

  case "${use_broker}" in
    n|N|no|NO)
      unset BROKER_URL BROKER_TOKEN PUBLIC_URL BROKER_SECRET || true
      echo "    Broker registration skipped."
      return 0
      ;;
  esac

  local default_url="${BROKER_URL:-https://webtermial.vercel.app}"
  local default_public="${PUBLIC_URL:-}"
  local default_token="${BROKER_TOKEN:-}"

  read -r -p "Broker URL [${default_url}]: " input_url
  BROKER_URL="${input_url:-$default_url}"
  BROKER_URL="${BROKER_URL%/}"

  while true; do
    if [[ -n "${default_token}" ]]; then
      read -r -p "Join token (7 chars) [saved ******* — Enter to keep]: " input_token
      if [[ -z "${input_token}" ]]; then
        BROKER_TOKEN="${default_token}"
      else
        BROKER_TOKEN="$(normalize_token "${input_token}")"
      fi
    else
      read -r -p "Join token (7 letters/digits from the web terminal): " input_token
      BROKER_TOKEN="$(normalize_token "${input_token}")"
    fi
    if [[ "${#BROKER_TOKEN}" -eq 7 ]]; then
      break
    fi
    echo "    Token must be exactly 7 A–Z / 0–9 characters. Try again."
    default_token=""
  done

  while true; do
    read -r -p "Public URL (Vast mapped port for 8765, e.g. http://IP:45323) [${default_public}]: " input_public
    PUBLIC_URL="${input_public:-$default_public}"
    PUBLIC_URL="${PUBLIC_URL%/}"
    if [[ "${PUBLIC_URL}" =~ ^https?:// ]]; then
      break
    fi
    echo "    Must start with http:// or https://"
    default_public=""
  done

  echo ""
  echo "    Broker URL : ${BROKER_URL}"
  echo "    Join token : ******* (saved, not shown)"
  echo "    Public URL : ${PUBLIC_URL}"
  read -r -p "Save and continue? [Y/n] " confirm
  confirm="${confirm:-Y}"
  case "${confirm}" in
    n|N|no|NO)
      echo "Aborted. Re-run when ready."
      exit 1
      ;;
  esac

  export BROKER_URL BROKER_TOKEN PUBLIC_URL
  save_broker_env
}

# Skip prompts if already fully provided via environment (non-interactive / CI)
if [[ -n "${BROKER_URL:-}" && -n "${PUBLIC_URL:-}" && ( -n "${BROKER_TOKEN:-}" || -n "${BROKER_SECRET:-}" ) && "${BROKER_NONINTERACTIVE:-}" == "1" ]]; then
  echo "==> Broker env already set (BROKER_NONINTERACTIVE=1) — skipping prompts"
  export BROKER_TOKEN="${BROKER_TOKEN:-$BROKER_SECRET}"
else
  prompt_broker_config
fi

echo "==> Starting API (server mode) on ${HOST}:${PORT}"
echo "    Health: http://127.0.0.1:${PORT}/api/health"
echo "    On Vast, use the mapped public IP:port for 8765 (see Open Ports)."
if [[ -n "${BROKER_URL:-}" && -n "${PUBLIC_URL:-}" && -n "${BROKER_TOKEN:-}" ]]; then
  echo "    Broker: ${BROKER_URL}"
  echo "    PUBLIC_URL=${PUBLIC_URL}"
  echo "    Join token set — will /handshake then /heartbeat every 2s"
else
  echo "    Broker: off"
fi
exec python -m backend --ui none --server-mode --host "${HOST}" --port "${PORT}"
