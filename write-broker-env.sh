#!/usr/bin/env bash
# Write .broker.env for start-server-linux.sh (local only — do not commit).
#
# Usage:
#   ./write-broker-env.sh <7-char-token> [public_url]
#
# Example:
#   ./write-broker-env.sh LEAF567 https://your-name.salad.cloud
set -euo pipefail
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")")"

BROKER_ENV_FILE="${BROKER_ENV_FILE:-.broker.env}"
BROKER_URL="${BROKER_URL:-https://webtermial.vercel.app}"

TOKEN_RAW="${1:-}"
PUBLIC_URL_RAW="${2:-${PUBLIC_URL:-}}"

if [[ -z "${TOKEN_RAW}" ]]; then
  echo "Usage: $0 <7-char-token> [public_url]"
  echo "  public_url example: https://your-name.salad.cloud"
  exit 1
fi

TOKEN="$(printf '%s' "${TOKEN_RAW}" | tr '[:lower:]' '[:upper:]' | tr -cd 'A-Z0-9' | cut -c1-7)"
if [[ "${#TOKEN}" -ne 7 ]]; then
  echo "ERROR: token must be exactly 7 A–Z / 0–9 characters (got '${TOKEN}')."
  exit 1
fi

PUBLIC_URL="$(printf '%s' "${PUBLIC_URL_RAW}" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
PUBLIC_URL="${PUBLIC_URL%/}"

if [[ -z "${PUBLIC_URL}" ]]; then
  echo "ERROR: public URL required (arg 2 or PUBLIC_URL env)."
  echo "  Example: $0 ${TOKEN} https://your-name.salad.cloud"
  exit 1
fi
if [[ ! "${PUBLIC_URL}" =~ ^https?:// ]]; then
  echo "ERROR: public URL must start with http:// or https://"
  exit 1
fi

BROKER_URL="${BROKER_URL%/}"
umask 077
cat > "${BROKER_ENV_FILE}" <<EOF
# Local only — do not commit. Used by start-server-linux.sh
BROKER_URL=${BROKER_URL}
BROKER_TOKEN=${TOKEN}
PUBLIC_URL=${PUBLIC_URL}
EOF

echo "==> Wrote ${BROKER_ENV_FILE}"
echo "    BROKER_URL  = ${BROKER_URL}"
echo "    BROKER_TOKEN= ${TOKEN}"
echo "    PUBLIC_URL  = ${PUBLIC_URL}"
echo ""
echo "Start with:"
echo "  ./start-server-linux.sh"
echo "(broker loads automatically from ${BROKER_ENV_FILE}; no prompt needed)"
