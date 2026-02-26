#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${LEAD_EMAIL:-}" ]]; then
  echo "LEAD_EMAIL is required (example: LEAD_EMAIL=sales@yourdomain.com)." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_FILE="$ROOT_DIR/config.js"

cat > "$TARGET_FILE" <<EOF
window.LEAD_EMAIL = "${LEAD_EMAIL}";
EOF

echo "Wrote $TARGET_FILE"
