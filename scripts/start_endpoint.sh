#!/usr/bin/env bash
# start_endpoint.sh — Boot vLLM + Safe Simplification Endpoint
# Usage: bash scripts/start_endpoint.sh
# Requires: NEBIUS_API_KEY, HF_TOKEN env vars

set -euo pipefail

MODEL="chambul/MediSimplifier-OpenBioLLM-merged"
VLLM_PORT=8001
API_PORT=8000

echo "==> Starting vLLM on :${VLLM_PORT}..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${VLLM_PORT}" \
    --host 127.0.0.1 \
    --dtype float16 \
    --max-model-len 4096 &
VLLM_PID=$!

echo "==> Waiting for vLLM to load (~10-15 min)..."
until curl -s http://127.0.0.1:${VLLM_PORT}/health | grep -q "ok"; do
    sleep 10
done
echo "==> vLLM ready."

echo "==> Starting Safe Endpoint API on :${API_PORT}..."
cd /app
python3 -m uvicorn safe_endpoint:app --host 0.0.0.0 --port ${API_PORT}

wait ${VLLM_PID}
