#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${API_HOST:-127.0.0.1}"
API_PORT_VALUE="${API_PORT:-8000}"
STREAMLIT_PORT_VALUE="${STREAMLIT_PORT:-8501}"

cleanup() {
  kill "${API_PID:-}" "${UI_PID:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Starting FastAPI on http://${HOST}:${API_PORT_VALUE} ..."
python -m uvicorn api:app --host "$HOST" --port "$API_PORT_VALUE" &
API_PID=$!

echo "Starting Streamlit on http://${HOST}:${STREAMLIT_PORT_VALUE} ..."
python -m streamlit run streamlit_app.py \
  --server.address "$HOST" \
  --server.port "$STREAMLIT_PORT_VALUE" \
  --server.headless true &
UI_PID=$!

wait "$API_PID" "$UI_PID"
