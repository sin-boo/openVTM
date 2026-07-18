#!/usr/bin/env bash
# One-shot Linux GPU server setup + start (Vast / cloud, including RTX 5090).
# Zero-touch: venv → CUDA torch → server deps → HF models → API on 0.0.0.0:8765
# No interactive prompts unless BROKER_PROMPT=1.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")"

export HF_REPO="${HF_REPO:-sinBoo1/models-VT-prototype}"
HOST="${SDANIME_HOST:-0.0.0.0}"
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
echo "    bind: ${HOST}:${PORT}"

# --- helpers ---
have_cmd() { command -v "$1" >/dev/null 2>&1; }

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
  [[ -f data/models/AnythingV5V3_v5PrtRE.safetensors ]] \
    && [[ -f data/models/ip-adapter/models/ip-adapter_sd15.bin ]] \
    && [[ -f data/models/ip-adapter/models/ip-adapter-plus_sd15.bin ]] \
    && [[ -d data/models/ip-adapter/models/image_encoder ]] \
    && { [[ -f data/models/finetuned/checkpoints/pose_adapter_step_020000.pt ]] \
         || [[ -f data/models/finetuned/checkpoints/pose_adapter_latest.pt ]]; }
}

# One-shot HTTPS check — no retries (retries won't fix broken TLS on the host).
hf_https_probe() {
  local url="${1:-https://huggingface.co}"
  echo "==> Checking HTTPS to ${url}"
  if have_cmd curl; then
    if curl --http1.1 -fsS -o /dev/null --connect-timeout 15 --max-time 30 "${url}"; then
      echo "    OK (curl --http1.1)"
      return 0
    fi
    echo "    FAIL: curl cannot complete TLS to ${url}"
    return 1
  fi
  if have_cmd wget; then
    if wget --timeout=15 --tries=1 -q -O /dev/null "${url}"; then
      echo "    OK (wget)"
      return 0
    fi
    echo "    FAIL: wget cannot reach ${url}"
    return 1
  fi
  echo "    WARN: no curl/wget — skipping probe"
  return 0
}

# Refresh CA store once (common Vast image gap). Not a download retry.
maybe_fix_ca_certs() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]] || ! have_cmd apt-get; then
    return 0
  fi
  echo "==> Ensuring ca-certificates (apt, once)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq ca-certificates curl >/dev/null 2>&1 || true
  update-ca-certificates >/dev/null 2>&1 || true
}

# Download one HF resolve URL via curl HTTP/1.1 with a progress bar.
# Uses a different TLS stack than Python urllib — not a "retry" of the same call.
hf_curl_file() {
  local repo="$1"
  local rel="$2"
  local dest="$3"
  local token="${4:-}"
  local base="${HF_ENDPOINT:-https://huggingface.co}"
  base="${base%/}"
  local url="${base}/${repo}/resolve/main/${rel}"
  mkdir -p "$(dirname "${dest}")"
  if [[ -f "${dest}" ]] && [[ $(stat -c%s "${dest}" 2>/dev/null || echo 0) -gt 1000 ]]; then
    echo "    skip (exists) ${rel}"
    return 0
  fi
  echo "    ↓ ${rel}"
  local args=(--http1.1 -L --fail --connect-timeout 30 --retry 0 --progress-bar)
  if [[ -n "${token}" ]]; then
    args+=(-H "Authorization: Bearer ${token}")
  fi
  # -# progress bar to stderr; write to temp then mv
  local tmp="${dest}.partial"
  if ! curl "${args[@]}" -o "${tmp}" "${url}"; then
    rm -f "${tmp}"
    return 1
  fi
  mv -f "${tmp}" "${dest}"
}

download_models() {
  mkdir -p data/models data/refs outputs

  if models_complete; then
    echo "==> Models already present under data/models/ — skip download"
    return 0
  fi

  local token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  if [[ -n "${token}" ]]; then
    export HF_TOKEN="${token}"
    export HUGGING_FACE_HUB_TOKEN="${token}"
  fi

  maybe_fix_ca_certs

  local hf_base="${HF_ENDPOINT:-https://huggingface.co}"
  if ! hf_https_probe "${hf_base}"; then
    echo ""
    echo "ERROR: This machine cannot open a working HTTPS session to Hugging Face."
    echo "       That is a host/network/TLS problem — retrying the same download will not help."
    echo "       Fix one of:"
    echo "         • outbound HTTPS / firewall on this Vast image"
    echo "         • CA certs (we tried apt ca-certificates if root)"
    echo "         • set HF_ENDPOINT to a reachable mirror if you use one"
    echo "         • copy data/models/ onto the box some other way"
    exit 1
  fi

  echo "==> Downloading models (curl HTTP/1.1 + progress bar)"
  echo "    repo: ${HF_REPO}"
  echo "    dest: $(pwd)/data/models"
  if [[ -z "${token}" ]]; then
    echo "    note: no HF_TOKEN set — private repos will 401"
  fi

  # Prefer curl over Python huggingface_hub: Vast often breaks Python SSL
  # (SSLEOFError) while curl --http1.1 still works. No Python retry loop.
  local -a required=(
    "AnythingV5V3_v5PrtRE.safetensors"
    "ip-adapter/models/ip-adapter_sd15.bin"
    "ip-adapter/models/ip-adapter_sd15_light.bin"
    "ip-adapter/models/ip-adapter-plus_sd15.bin"
    "ip-adapter/models/image_encoder/config.json"
    "ip-adapter/models/image_encoder/preprocessor_config.json"
    "finetuned/param_stats.json"
    "finetuned/checkpoints/pose_adapter_step_020000.pt"
  )
  # image encoder weight — try safetensors first, then bin
  local -a encoder_weights=(
    "ip-adapter/models/image_encoder/model.safetensors"
    "ip-adapter/models/image_encoder/pytorch_model.bin"
  )

  local fail=0
  local rel
  for rel in "${required[@]}"; do
    if ! hf_curl_file "${HF_REPO}" "${rel}" "data/models/${rel}" "${token}"; then
      echo "    FAIL ${rel}"
      fail=1
    fi
  done

  local got_encoder=0
  for rel in "${encoder_weights[@]}"; do
    if [[ -f "data/models/${rel}" ]] && [[ $(stat -c%s "data/models/${rel}" 2>/dev/null || echo 0) -gt 1000000 ]]; then
      got_encoder=1
      break
    fi
    if hf_curl_file "${HF_REPO}" "${rel}" "data/models/${rel}" "${token}"; then
      got_encoder=1
      break
    fi
  done
  if [[ "${got_encoder}" -ne 1 ]]; then
    echo "    FAIL image_encoder weights"
    fail=1
  fi

  if [[ "${fail}" -ne 0 ]]; then
    echo ""
    echo "==> Bundle incomplete — trying public Anything V5 + IP-Adapter (still curl HTTP/1.1)"
    # Public fallbacks: different repos, same TLS path (not a blind retry of the private bundle).
    hf_curl_file "ckpt/anything-v5.0" "AnythingV5V3_v5PrtRE.safetensors" \
      "data/models/AnythingV5V3_v5PrtRE.safetensors" "" || true
    local pub
    for pub in \
      "models/ip-adapter_sd15.bin" \
      "models/ip-adapter_sd15_light.bin" \
      "models/ip-adapter-plus_sd15.bin" \
      "models/image_encoder/config.json" \
      "models/image_encoder/preprocessor_config.json" \
      "models/image_encoder/model.safetensors" \
      "models/image_encoder/pytorch_model.bin"
    do
      hf_curl_file "h94/IP-Adapter" "${pub}" "data/models/ip-adapter/${pub}" "" || true
    done
  fi

  if ! models_complete; then
    echo ""
    echo "ERROR: model download did not produce a complete set."
    echo "       Likely causes (not fixed by retrying):"
    echo "         • HF_TOKEN missing/wrong for private ${HF_REPO}"
    echo "         • TLS/network path to Hugging Face broken on this host"
    echo "         • repo layout missing expected files"
    echo "       Set HF_TOKEN if private, or copy weights into data/models/ manually."
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
  if [[ -d data/models/ip-adapter/models/image_encoder ]]; then
    echo "    OK  data/models/ip-adapter/models/image_encoder/"
  else
    echo "    MISSING data/models/ip-adapter/models/image_encoder/"
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
  # Only when BROKER_PROMPT=1 and a TTY is available.
  load_broker_env
  auto_public_url

  echo ""
  echo "==> Web terminal (discovery broker) — interactive"
  local default_url="${BROKER_URL:-https://webtermial.vercel.app}"
  local default_public="${PUBLIC_URL:-}"
  local default_token="${BROKER_TOKEN:-}"

  read -r -p "Broker URL [${default_url}]: " input_url
  BROKER_URL="${input_url:-$default_url}"
  BROKER_URL="${BROKER_URL%/}"

  while true; do
    if [[ -n "${default_token}" ]]; then
      read -r -p "Join token (7 chars) [saved — Enter to keep]: " input_token
      if [[ -z "${input_token}" ]]; then
        BROKER_TOKEN="${default_token}"
      else
        BROKER_TOKEN="$(normalize_token "${input_token}")"
      fi
    else
      read -r -p "Join token (7 letters/digits): " input_token
      BROKER_TOKEN="$(normalize_token "${input_token}")"
    fi
    if [[ "${#BROKER_TOKEN}" -eq 7 ]]; then
      break
    fi
    echo "    Token must be exactly 7 A–Z / 0–9 characters."
    default_token=""
  done

  while true; do
    read -r -p "Public URL [${default_public}]: " input_public
    PUBLIC_URL="${input_public:-$default_public}"
    PUBLIC_URL="${PUBLIC_URL%/}"
    if [[ "${PUBLIC_URL}" =~ ^https?:// ]]; then
      break
    fi
    echo "    Must start with http:// or https://"
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

  if [[ "${BROKER_PROMPT:-0}" == "1" ]] && [[ -t 0 ]]; then
    prompt_broker_config
    return 0
  fi

  echo "==> Broker: off (set BROKER_URL + BROKER_TOKEN + PUBLIC_URL, or BROKER_PROMPT=1)"
  echo "    Tip on Vast: open port ${PORT} so VAST_TCP_PORT_${PORT} is set; PUBLIC_URL auto-fills."
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
