#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

echo "[1/3] Starting services..."
docker compose up -d --build

echo "[2/3] Waiting for router health..."
for i in {1..60}; do
  if curl -sf "http://localhost:8024/health" >/dev/null; then
    break
  fi
  if [[ "$i" -eq 60 ]]; then
    echo "Router did not become healthy after 60 seconds." >&2
    exit 1
  fi
  sleep 1
done

echo "[3/3] Running sample conversion simulation..."
python3 scripts/demo.py "$@"
