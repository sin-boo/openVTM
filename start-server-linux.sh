#!/usr/bin/env bash
# One-shot Linux GPU server setup + start (Vast / Salad / cloud, including RTX 5090).
# venv → CUDA torch → server deps → HF models → API (auto port if busy).
# Broker join-token prompt is ON by default (BROKER_PROMPT=0 to skip).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")"

export HF_REPO="${HF_REPO:-sinBoo1/models-VT-prototype}"
HOST="${SDANIME_HOST:-0.0.0.0}"
# Preferred port; if busy, script auto-picks the next free one unless SDANIME_PORT_STRICT=1.
PORT="${SDANIME_PORT:-8765}"
VENV_DIR="${VENV_DIR:-.venv}"
BROKER_ENV_FILE="${BROKER_ENV_FILE:-.broker.env}"
REQ_FILE="${REQ_FILE:-requirements.server.txt}"
if [[ ! -f "${REQ_FILE}" ]]; then
  REQ_FILE="requirements.txt"
fi

export SDANIME_SERVER_MODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# Force live progress in non-TTY / Vast web shells
export PYTHONUNBUFFERED=1
export PIP_PROGRESS_BAR=on
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TQDM_MININTERVAL=0.3

echo "==> SDAnime Pose server setup (Linux)"
echo "    repo root: $(pwd)"
echo "    HF models: ${HF_REPO}"
echo "    preferred bind: ${HOST}:${PORT}"

# --- helpers ---
have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Pick a free TCP listen port. Uses python (ss/lsof/netstat often missing on cloud images).
# Respects SDANIME_PORT as the preferred start; scans upward unless SDANIME_PORT_STRICT=1.
pick_listen_port() {
  local preferred="${1:-8765}"
  local host="${2:-0.0.0.0}"
  local strict="${SDANIME_PORT_STRICT:-0}"
  local chosen
  chosen="$(
    PREFERRED="${preferred}" HOST_BIND="${host}" STRICT="${strict}" python3 - <<'PY'
import os
import socket
import sys

preferred = int(os.environ.get("PREFERRED", "8765"))
host = os.environ.get("HOST_BIND", "0.0.0.0")
strict = os.environ.get("STRICT", "0") == "1"
span = 1 if strict else 64

def free(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()

for port in range(preferred, preferred + span):
    if free(port):
        print(port)
        raise SystemExit(0)
print(f"ERROR: no free port in {preferred}..{preferred + span - 1}", file=sys.stderr)
raise SystemExit(1)
PY
  )" || {
    echo "ERROR: could not find a free listen port (preferred ${preferred})."
    echo "       Free the old process or set SDANIME_PORT to an open port."
    exit 1
  }
  PORT="${chosen}"
  export PORT
  if [[ "${PORT}" != "${preferred}" ]]; then
    echo "==> Port ${preferred} busy — using ${PORT} instead"
    echo "    (set SDANIME_PORT_STRICT=1 to fail instead of auto-picking)"
  else
    echo "==> Listen port ${PORT} is free"
  fi
}

ensure_python() {
  if ! have_cmd python3; then
    echo "ERROR: python3 not found. Install Python 3.10+ and retry."
    exit 1
  fi
  local py_minor
  py_minor="$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
  local py_major
  py_major="$(python3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)"
  if [[ "${py_major}" -lt 3 ]] || [[ "${py_major}" -eq 3 && "${py_minor}" -lt 10 ]]; then
    echo "ERROR: Need Python 3.10+, found $(python3 --version 2>&1)."
    exit 1
  fi
}

maybe_apt_basics() {
  # Best-effort on Ubuntu/Debian images (Vast). Skip if no apt or not root.
  if [[ "${EUID:-$(id -u)}" -ne 0 ]] || ! have_cmd apt-get; then
    return 0
  fi
  if python3 -c 'import venv' 2>/dev/null; then
    return 0
  fi
  echo "==> Installing python3-venv (apt)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3-venv python3-pip >/dev/null
}

gpu_name() {
  if have_cmd nvidia-smi; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
  fi
}

gpu_compute_cap() {
  if have_cmd nvidia-smi; then
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' '
  fi
}

# RTX 50-series / Blackwell (sm_120) needs PyTorch cu128 — cu124 has no kernels.
needs_cu128() {
  local name cap major
  name="$(gpu_name || true)"
  cap="$(gpu_compute_cap || true)"
  if [[ -n "${cap}" ]]; then
    major="${cap%%.*}"
    if [[ "${major}" =~ ^[0-9]+$ ]] && [[ "${major}" -ge 12 ]]; then
      return 0
    fi
  fi
  if echo "${name}" | grep -qiE 'RTX[[:space:]]*50|5090|5080|5070|Blackwell'; then
    return 0
  fi
  return 1
}

install_torch() {
  echo "==> Installing PyTorch"
  echo "    (large CUDA wheels — watch the pip progress bar; extract can take several minutes)"
  # Prefer an explicit progress bar even when stdout is not a TTY (Vast web terminal).
  local pip_opts=(--upgrade --progress-bar on)
  if ! have_cmd nvidia-smi; then
    echo "WARNING: nvidia-smi not found — installing CPU torch (slow / may be unusable)."
    pip install "${pip_opts[@]}" torch torchvision torchaudio
    return 0
  fi

  local name cap
  name="$(gpu_name || true)"
  cap="$(gpu_compute_cap || true)"
  echo "    GPU: ${name:-unknown} (compute_cap=${cap:-?})"

  if needs_cu128; then
    echo "    Blackwell / sm_12x detected → CUDA 12.8 wheels (required for RTX 5090)"
    pip install "${pip_opts[@]}" torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128
  else
    echo "    Trying CUDA 12.4 wheels (fallback: 12.8 → 12.1)"
    pip install "${pip_opts[@]}" torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu124 \
      || pip install "${pip_opts[@]}" torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 \
      || pip install "${pip_opts[@]}" torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 \
      || pip install "${pip_opts[@]}" torch torchvision torchaudio
  fi
}

cuda_smoke_test() {
  echo "==> CUDA smoke test"
  python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"ERROR: cannot import torch: {exc}")
    sys.exit(1)

print(f"    torch={torch.__version__} cuda_built={torch.version.cuda}")
if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False — GPU not usable.")
    print("       Check nvidia drivers / NVIDIA Container Toolkit, then re-run.")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"    device={name} capability={cap[0]}.{cap[1]}")
try:
    x = torch.zeros(1, device="cuda")
    y = x + 1
    torch.cuda.synchronize()
    print(f"    smoke ok (tensor={float(y.item())})")
except Exception as exc:
    print(f"ERROR: CUDA kernel smoke failed: {exc}")
    print("       RTX 50-series needs cu128 PyTorch. Re-run after: pip uninstall -y torch torchvision torchaudio")
    sys.exit(1)
PY
}

models_complete() {
  local enc="data/models/ip-adapter/models/image_encoder"
  [[ -f data/models/AnythingV5V3_v5PrtRE.safetensors ]] \
    && [[ -f data/models/ip-adapter/models/ip-adapter_sd15.bin ]] \
    && [[ -f data/models/ip-adapter/models/ip-adapter-plus_sd15.bin ]] \
    && [[ -f "${enc}/config.json" ]] \
    && { [[ -f "${enc}/model.safetensors" ]] || [[ -f "${enc}/pytorch_model.bin" ]]; } \
    && { [[ -f data/models/finetuned/checkpoints/pose_adapter_step_020000.pt ]] \
         || [[ -f data/models/finetuned/checkpoints/pose_adapter_latest.pt ]]; }
}

# Download one file from the configured public HF repo.
# Force IPv4 because some Vast hosts advertise an unusable IPv6 route, causing
# Python requests/curl to stall or end TLS with SSL_UNEXPECTED_EOF.
hf_log() {
  # Always stderr + timestamp so Salad/Vast web terminals show activity
  # even when stdout is buffered or jobs run in parallel.
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2
}

hf_curl_file() {
  local rel="$1"
  local dest="$2"
  local base="${HF_ENDPOINT:-https://huggingface.co}"
  base="${base%/}"
  local escaped_rel="${rel// /%20}"
  local url="${base}/${HF_REPO}/resolve/main/${escaped_rel}"
  mkdir -p "$(dirname "${dest}")"
  if [[ -s "${dest}" ]]; then
    hf_log "skip (exists) ${rel}"
    return 0
  fi
  hf_log "START ${rel}"
  local tmp="${dest}.partial"
  # -# = progress meter on stderr (visible during long image_encoder / ckpt pulls)
  if ! curl -4 --http1.1 -L --fail -# \
      --connect-timeout 30 --retry 2 --retry-delay 2 \
      -o "${tmp}" "${url}"; then
    rm -f "${tmp}"
    hf_log "FAIL  ${rel}"
    return 1
  fi
  mv -f "${tmp}" "${dest}" || {
    # Parallel/racy second job: another curl may have already moved the file.
    if [[ -s "${dest}" ]]; then
      hf_log "DONE  ${rel} (already present after race)"
      rm -f "${tmp}"
      return 0
    fi
    hf_log "FAIL  ${rel} (mv missing partial)"
    return 1
  }
  local bytes
  bytes="$(wc -c < "${dest}" | tr -d ' ')"
  hf_log "DONE  ${rel} (${bytes} bytes)"
}

# Cap concurrent HF downloads (override with HF_DOWNLOAD_JOBS).
# Helps folders like image_encoder (many small files / TLS handshakes).
hf_wait_for_slot() {
  local max_jobs="$1"
  local ignore_pid="${2:-}"
  while true; do
    local running
    running="$(jobs -rp | wc -l | tr -d ' ')"
    if [[ -n "${ignore_pid}" ]] && kill -0 "${ignore_pid}" 2>/dev/null; then
      running=$((running - 1))
    fi
    if [[ "${running}" -lt "${max_jobs}" ]]; then
      return 0
    fi
    # Prefer wait -n (bash 4.3+); fall back to a short sleep.
    # Ignore job exit status here — failures are recorded in fail_dir.
    if ! wait -n 2>/dev/null; then
      sleep 0.2
    fi
  done
}

download_models() {
  mkdir -p data/models data/refs outputs

  if models_complete; then
    echo "==> Models already present under data/models/ — skip download"
    return 0
  fi

  local hf_base="${HF_ENDPOINT:-https://huggingface.co}"
  hf_base="${hf_base%/}"
  local manifest
  manifest="$(mktemp)"
  trap 'rm -f "${manifest}"' RETURN

  echo "==> Reading public Hugging Face repository manifest (forced IPv4)"
  echo "    (this can take up to ~30s if HF is slow — hang here means network/TLS)"
  if ! curl -4 --http1.1 -L --fail --silent --show-error \
      --connect-timeout 30 --retry 0 \
      -o "${manifest}" "${hf_base}/api/models/${HF_REPO}"; then
    echo "ERROR: Cannot read the public repository manifest over IPv4:"
    echo "       ${hf_base}/api/models/${HF_REPO}"
    echo "       This is a network/TLS problem on the host, not a download retry problem."
    exit 1
  fi
  echo "    manifest ok"

  local -a repo_files=()
  mapfile -t repo_files < <(
    python - "${manifest}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    payload = json.load(f)

for item in payload.get("siblings", []):
    name = item.get("rfilename")
    if name and name != ".gitattributes":
        print(name)
PY
  )
  rm -f "${manifest}"
  trap - RETURN

  if [[ "${#repo_files[@]}" -eq 0 ]]; then
    echo "ERROR: Public repository manifest contained no downloadable files."
    exit 1
  fi

  local max_jobs="${HF_DOWNLOAD_JOBS:-8}"
  if ! [[ "${max_jobs}" =~ ^[1-9][0-9]*$ ]]; then
    max_jobs=8
  fi
  if [[ "${max_jobs}" -gt 32 ]]; then
    max_jobs=32
  fi

  local need=0
  local rel
  for rel in "${repo_files[@]}"; do
    if [[ ! -s "data/models/${rel}" ]]; then
      need=$((need + 1))
    fi
  done

  echo "==> Downloading every file from the public Hugging Face repository"
  echo "    repo: ${HF_REPO}"
  echo "    dest: $(pwd)/data/models"
  echo "    files in repo: ${#repo_files[@]}  still need: ${need}"
  echo "    parallel jobs: ${max_jobs} (set HF_DOWNLOAD_JOBS to change)"
  echo "    progress: START/DONE lines + curl meters on stderr"

  local fail_dir
  fail_dir="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '${fail_dir}'" RETURN

  # Heartbeat: print growing .partial files so long curls are visible.
  (
    while [[ ! -f "${fail_dir}/_downloads_finished" ]]; do
      sleep 8
      [[ -f "${fail_dir}/_downloads_finished" ]] && break
      partials=0
      while IFS= read -r line; do
        partials=$((partials + 1))
        hf_log " partial ${line}"
      done < <(find data/models -type f -name '*.partial' -printf '%p (%s bytes)\n' 2>/dev/null || true)
      if [[ "${partials}" -eq 0 ]]; then
        hf_log "… waiting (no .partial files yet — connecting or between files)"
      else
        hf_log "… ${partials} download(s) in progress"
      fi
    done
  ) &
  local heartbeat_pid=$!

  local -a dl_pids=()
  for rel in "${repo_files[@]}"; do
    hf_wait_for_slot "${max_jobs}" "${heartbeat_pid}"
    (
      if ! hf_curl_file "${rel}" "data/models/${rel}"; then
        marker="$(printf '%s' "${rel}" | tr '/ ' '__')"
        printf '%s\n' "${rel}" > "${fail_dir}/${marker}"
      fi
    ) &
    dl_pids+=($!)
  done

  local pid
  for pid in "${dl_pids[@]}"; do
    wait "${pid}" || true
  done
  : > "${fail_dir}/_downloads_finished"
  wait "${heartbeat_pid}" 2>/dev/null || true

  local -a failed=()
  local marker_file
  for marker_file in "${fail_dir}"/*; do
    [[ -e "${marker_file}" ]] || continue
    [[ "$(basename "${marker_file}")" == "_downloads_finished" ]] && continue
    failed+=("$(cat "${marker_file}")")
  done
  if [[ "${#failed[@]}" -gt 0 ]]; then
    echo "ERROR: Failed downloading one or more files over forced IPv4:"
    printf '       %s\n' "${failed[@]}" | sort -u
    exit 1
  fi
  rm -rf "${fail_dir}"
  trap - RETURN

  if ! models_complete; then
    echo ""
    echo "ERROR: ${HF_REPO} downloaded, but required runtime model files are missing."
    echo "       Fix the public repository layout; this script does not use fallback repositories."
    exit 1
  fi
  echo "==> Model download complete"
}

verify_models() {
  echo "==> Verifying required model files"
  local ok=1
  local f
  for f in \
    data/models/AnythingV5V3_v5PrtRE.safetensors \
    data/models/ip-adapter/models/ip-adapter_sd15.bin \
    data/models/ip-adapter/models/ip-adapter-plus_sd15.bin
  do
    if [[ -f "$f" ]]; then
      echo "    OK  $f"
    else
      echo "    MISSING $f"
      ok=0
    fi
  done
  local enc="data/models/ip-adapter/models/image_encoder"
  if [[ -f "${enc}/config.json" ]] \
    && { [[ -f "${enc}/model.safetensors" ]] || [[ -f "${enc}/pytorch_model.bin" ]]; }; then
    echo "    OK  ${enc}/ (config + weights)"
  else
    echo "    MISSING ${enc}/ weights (need model.safetensors or pytorch_model.bin)"
    ok=0
  fi
  if [[ -f data/models/finetuned/checkpoints/pose_adapter_step_020000.pt ]]; then
    echo "    OK  pose_adapter_step_020000.pt"
  elif [[ -f data/models/finetuned/checkpoints/pose_adapter_latest.pt ]]; then
    echo "    OK  pose_adapter_latest.pt"
  else
    echo "    MISSING finetuned PoseAdapter checkpoint"
    ok=0
  fi
  if [[ -f data/models/finetuned/param_stats.json ]]; then
    echo "    OK  param_stats.json"
  else
    echo "    WARN param_stats.json missing (engine will use slider fallback)"
  fi
  if [[ -f data/refs/train_char_1.png ]]; then
    echo "    OK  data/refs/train_char_1.png"
  else
    echo "    WARN default ref missing (upload via /api/reference)"
  fi
  if [[ "${ok}" -ne 1 ]]; then
    echo "ERROR: required models missing after download."
    echo "       Check HF repo access (public or set HF_TOKEN) and retry."
    exit 1
  fi
}

load_broker_env() {
  if [[ -f "${BROKER_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck disable=SC1091
    source "${BROKER_ENV_FILE}"
    set +a
    echo "==> Loaded broker settings from ${BROKER_ENV_FILE}"
  fi
}

# Vast: PUBLIC_IPADDR + VAST_TCP_PORT_<internal>
auto_public_url() {
  if [[ -n "${PUBLIC_URL:-}" ]]; then
    return 0
  fi
  local ip mapped
  ip="${PUBLIC_IPADDR:-}"
  mapped_var="VAST_TCP_PORT_${PORT}"
  mapped="${!mapped_var:-}"
  if [[ -z "${ip}" && -f /var/lib/vastai_kaalia/host_ipaddr ]]; then
    ip="$(tr -d '[:space:]' </var/lib/vastai_kaalia/host_ipaddr || true)"
  fi
  if [[ -n "${ip}" && -n "${mapped}" ]]; then
    PUBLIC_URL="http://${ip}:${mapped}"
    export PUBLIC_URL
    echo "==> Auto PUBLIC_URL from Vast: ${PUBLIC_URL}"
  fi
}

normalize_token() {
  echo "$1" | tr '[:lower:]' '[:upper:]' | tr -cd 'A-Z0-9' | cut -c1-7
}

# Trim CR/LF/spaces (web-terminal pastes often include \r or look typed but read empty).
trim_ws() {
  printf '%s' "$1" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

# Read a line from the real console when possible (Salad/Vast web shells
# often have non-TTY stdin, so plain `read` never prompts).
tty_read() {
  local prompt="$1"
  local reply=""
  if [[ -r /dev/tty ]]; then
    # shellcheck disable=SC2162
    read -r -p "${prompt}" reply </dev/tty || true
  elif [[ -t 0 ]]; then
    read -r -p "${prompt}" reply || true
  else
    echo "WARNING: no TTY — cannot prompt interactively." >&2
    echo "         Export BROKER_URL BROKER_TOKEN PUBLIC_URL instead." >&2
  fi
  trim_ws "${reply}"
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

prompt_broker_config() {
  load_broker_env
  auto_public_url

  echo ""
  echo "==> Web terminal (discovery broker) — interactive"
  echo "    Use the same 7-char token as https://webtermial.vercel.app"
  echo "    PUBLIC_URL must be reachable from your PC (Salad gateway / Vast mapped port)."
  local default_url="${BROKER_URL:-https://webtermial.vercel.app}"
  local default_public="${PUBLIC_URL:-}"
  local default_token="${BROKER_TOKEN:-}"
  local input_url input_token input_public

  input_url="$(tty_read "Broker URL [${default_url}]: ")"
  BROKER_URL="${input_url:-$default_url}"
  BROKER_URL="${BROKER_URL%/}"

  while true; do
    if [[ -n "${default_token}" ]]; then
      input_token="$(tty_read "Join token (7 chars) [saved — Enter to keep]: ")"
      if [[ -z "${input_token}" ]]; then
        BROKER_TOKEN="${default_token}"
      else
        BROKER_TOKEN="$(normalize_token "${input_token}")"
      fi
    else
      input_token="$(tty_read "Join token (7 letters/digits): ")"
      BROKER_TOKEN="$(normalize_token "${input_token}")"
    fi
    if [[ "${#BROKER_TOKEN}" -eq 7 ]]; then
      break
    fi
    echo "    Token must be exactly 7 A–Z / 0–9 characters."
    default_token=""
  done

  while true; do
    if [[ -z "${default_public}" ]]; then
      echo "    Tip (Salad): use your container gateway URL that reaches port ${PORT}"
      echo "         e.g. https://your-name.salad.cloud   (must route to ${PORT})"
      echo "    If paste fails in this web terminal, Ctrl+C and run with env vars:"
      echo "         export BROKER_URL=https://webtermial.vercel.app"
      echo "         export BROKER_TOKEN=YOUR7CHR"
      echo "         export PUBLIC_URL=https://your-name.salad.cloud"
      echo "         ./start-server-linux.sh"
    fi
    input_public="$(tty_read "Public URL [${default_public}]: ")"
    PUBLIC_URL="$(trim_ws "${input_public:-$default_public}")"
    PUBLIC_URL="${PUBLIC_URL%/}"
    if [[ "${PUBLIC_URL}" =~ ^https?:// ]]; then
      break
    fi
    echo "    Must start with http:// or https:// (got: '${PUBLIC_URL}')"
    default_public=""
  done

  export BROKER_URL BROKER_TOKEN PUBLIC_URL
  save_broker_env
}

configure_broker() {
  load_broker_env
  auto_public_url
  export BROKER_TOKEN="${BROKER_TOKEN:-${BROKER_SECRET:-}}"

  if [[ -n "${BROKER_URL:-}" && -n "${PUBLIC_URL:-}" && -n "${BROKER_TOKEN:-}" ]]; then
    BROKER_URL="${BROKER_URL%/}"
    PUBLIC_URL="${PUBLIC_URL%/}"
    BROKER_TOKEN="$(normalize_token "${BROKER_TOKEN}")"
    export BROKER_URL PUBLIC_URL BROKER_TOKEN
    if [[ "${#BROKER_TOKEN}" -ne 7 ]]; then
      echo "WARNING: BROKER_TOKEN must be 7 chars — broker registration disabled."
      unset BROKER_URL BROKER_TOKEN PUBLIC_URL || true
      return 0
    fi
    echo "==> Broker registration enabled"
    echo "    Broker URL : ${BROKER_URL}"
    echo "    Public URL : ${PUBLIC_URL}"
    return 0
  fi

  # Default ON so Salad/Vast users get the join-token prompt.
  # Set BROKER_PROMPT=0 to skip (headless / no discovery).
  if [[ "${BROKER_PROMPT:-1}" == "0" ]]; then
    echo "==> Broker: off (BROKER_PROMPT=0)"
    return 0
  fi

  prompt_broker_config
  if [[ -z "${BROKER_URL:-}" || -z "${PUBLIC_URL:-}" || -z "${BROKER_TOKEN:-}" ]]; then
    echo "==> Broker: off (prompt incomplete — set env vars and re-run)"
    return 0
  fi
  echo "==> Broker registration enabled"
  echo "    Broker URL : ${BROKER_URL}"
  echo "    Public URL : ${PUBLIC_URL}"
}

# --- main ---
ensure_python
maybe_apt_basics

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel setuptools --progress-bar on

install_torch
cuda_smoke_test

echo "==> Installing ${REQ_FILE}"
pip install -r "${REQ_FILE}" --progress-bar on

download_models
verify_models
pick_listen_port "${PORT}" "${HOST}"
configure_broker

echo "==> Starting API (server mode) on ${HOST}:${PORT}"
echo "    Health: http://127.0.0.1:${PORT}/api/health"
if [[ -n "${PUBLIC_URL:-}" ]]; then
  echo "    Public: ${PUBLIC_URL}/api/health"
fi
if [[ -n "${BROKER_URL:-}" && -n "${PUBLIC_URL:-}" && -n "${BROKER_TOKEN:-}" ]]; then
  echo "    Broker: ${BROKER_URL} (handshake + heartbeat every 2s)"
else
  echo "    Broker: off"
fi
exec python -m backend --ui none --server-mode --host "${HOST}" --port "${PORT}"
