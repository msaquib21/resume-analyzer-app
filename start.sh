#!/bin/sh
set -e

echo "Starting FastAPI backend on 127.0.0.1:8000..."
uvicorn backend.server:app --host 127.0.0.1 --port 8000 &

# Allow backend process to bind
sleep 2

PORT_TO_USE="${PORT:-10000}"
echo "Starting Streamlit frontend on 0.0.0.0:$PORT_TO_USE..."
exec streamlit run frontend/app.py \
  --server.port "$PORT_TO_USE" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
