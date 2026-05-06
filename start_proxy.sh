#!/usr/bin/env bash
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://10.10.111.43:8005}"
PORT="${PORT:-6003}"

cd /root

/usr/bin/python3 -m vllm_tool_proxy.server \
  --backend-url "${BACKEND_URL}" \
  --port "${PORT}" \
  --tool-parser auto \
  --native-template
