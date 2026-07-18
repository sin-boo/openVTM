#!/usr/bin/env bash
# Find openVTM / SDAnime Pose install locations on this machine.
set -euo pipefail

echo "==> Looking for openVTM / SDAnime Pose"
echo "    hostname: $(hostname 2>/dev/null || echo unknown)"
echo "    user:     $(whoami 2>/dev/null || echo unknown)"
echo "    cwd:      $(pwd)"
echo ""

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Markers that identify this project.
is_openvtm_dir() {
  local d="$1"
  [[ -f "${d}/start-server-linux.sh" ]] \
    || [[ -f "${d}/backend/__main__.py" && -d "${d}/backend" ]] \
    || [[ -f "${d}/requirements.server.txt" ]]
}

print_hit() {
  local label="$1"
  local path="$2"
  echo "FOUND ${label}:"
  echo "  ${path}"
  if [[ -f "${path}/start-server-linux.sh" ]]; then
    echo "  start script: ${path}/start-server-linux.sh"
  fi
  if [[ -d "${path}/data/models" ]]; then
    echo "  models dir:   ${path}/data/models"
    if [[ -f "${path}/data/models/ip-adapter/models/image_encoder/model.safetensors" ]] \
      || [[ -f "${path}/data/models/ip-adapter/models/image_encoder/pytorch_model.bin" ]]; then
      echo "  image_encoder: OK (weights present)"
    elif [[ -d "${path}/data/models/ip-adapter/models/image_encoder" ]]; then
      echo "  image_encoder: folder exists (check weights)"
    else
      echo "  image_encoder: missing"
    fi
  fi
  if [[ -d "${path}/.venv" ]]; then
    echo "  venv:         ${path}/.venv"
  fi
  if [[ -f "${path}/.broker.env" ]]; then
    echo "  broker env:   ${path}/.broker.env"
  fi
  echo ""
}

declare -a hits=()

# 1) Common cloud paths
for d in \
  /openVTM \
  /root/openVTM \
  /workspace/openVTM \
  /home/*/openVTM \
  "$HOME/openVTM" \
  /real_stream_SDAnime \
  "$HOME/real_stream_SDAnime" \
  /workspace/real_stream_SDAnime
do
  # Expand globs safely
  for path in $d; do
    [[ -d "${path}" ]] || continue
    if is_openvtm_dir "${path}"; then
      hits+=("${path}")
    fi
  done
done

# 2) Search by start script name (limited depth / roots to stay fast)
echo "==> Searching filesystem (may take a few seconds)…"
search_roots=(/ /root /home /workspace /opt /var)
if have_cmd find; then
  while IFS= read -r f; do
    d="$(dirname "$f")"
    hits+=("$d")
  done < <(
    find "${search_roots[@]}" \
      -path '/proc' -prune -o \
      -path '/sys' -prune -o \
      -path '/dev' -prune -o \
      -path '*/.git/*' -prune -o \
      -name 'start-server-linux.sh' -type f -print 2>/dev/null \
      | head -n 20
  )
fi

# Deduplicate
declare -A seen=()
declare -a unique=()
for h in "${hits[@]+"${hits[@]}"}"; do
  [[ -n "${h}" ]] || continue
  # Resolve to absolute if possible
  if [[ -d "${h}" ]]; then
    abs="$(cd "${h}" && pwd)"
  else
    abs="${h}"
  fi
  if [[ -z "${seen[$abs]+x}" ]]; then
    seen[$abs]=1
    unique+=("$abs")
  fi
done

if [[ "${#unique[@]}" -eq 0 ]]; then
  echo "No install found."
  echo ""
  echo "Clone it with:"
  echo "  cd /"
  echo "  git clone https://github.com/sin-boo/openVTM.git"
  echo "  cd /openVTM"
  echo "  ./start-server-linux.sh"
  exit 1
fi

echo ""
echo "==> Install location(s)"
echo ""
for u in "${unique[@]}"; do
  print_hit "openVTM/SDAnime" "$u"
done

echo "==> What to run"
primary="${unique[0]}"
echo "  cd ${primary}"
echo "  ./start-server-linux.sh"
echo ""
echo "Or with broker (edit token):"
echo "  cd ${primary}"
echo "  export BROKER_URL=https://webtermial.vercel.app"
echo "  export BROKER_TOKEN=YOUR7CHR"
echo "  export PUBLIC_URL=https://YOUR-NAME.salad.cloud"
echo "  ./start-server-linux.sh"
