#!/usr/bin/env bash
# MediSimplifier — Nebius reproduction script
# Requires: nebius CLI v0.12.229+, env vars NEBIUS_PROJECT_ID, NEBIUS_SUBNET_ID, HF_TOKEN
#
# Quick start (no training required):
#   export NEBIUS_PROJECT_ID=... NEBIUS_SUBNET_ID=... HF_TOKEN=...
#   bash scripts/reproduce.sh eval_only
#
# Modes: full | eval_only | ablation | serve

set -euo pipefail

: "${NEBIUS_PROJECT_ID:?Set NEBIUS_PROJECT_ID}"
: "${NEBIUS_SUBNET_ID:?Set NEBIUS_SUBNET_ID}"
: "${HF_TOKEN:?Set HF_TOKEN}"

MODE="${1:-full}"

IMAGE="chambul/medisimplifier:train-v25"
BUCKET="medisimplifier-adapters"
PLATFORM="gpu-h100-sxm"
PRESET="1gpu-16vcpu-200gb"
DISK="250Gi"

# ── Step 0: Create output bucket (idempotent) ─────────────────────────────────
if [[ "$MODE" == "full" || "$MODE" == "ablation" ]]; then
  echo "==> Creating storage bucket (skip if exists)..."
  nebius storage bucket create \
    --name "$BUCKET" \
    --parent-id "${NEBIUS_PROJECT_ID}" 2>/dev/null || true
fi

# ── Step 1: Full training (3 epochs, winner config) ───────────────────────────
if [[ "$MODE" == "full" ]]; then
  echo "==> Submitting full training job (~70 min, ~$9)..."
  nebius ai job create \
    --name medisimplifier-full-train \
    --parent-id "${NEBIUS_PROJECT_ID}" \
    --image "$IMAGE" \
    --container-command python \
    --args "train.py --model openbio --epochs 3 --rank 32 --modules all_attn --seed 42" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HF_HOME=/tmp/hf_cache \
    --platform "$PLATFORM" \
    --preset "$PRESET" \
    --disk-size "$DISK" \
    --subnet-id "${NEBIUS_SUBNET_ID}" \
    --volume "${BUCKET}:/output:rw" \
    --timeout 5h
fi

# ── Step 2: Ablation study (9 parallel jobs, ~$20 total) ─────────────────────
if [[ "$MODE" == "full" || "$MODE" == "ablation" ]]; then
  echo "==> Submitting ablation jobs (Phase 1 — rank, modules=q+v, data=8K)..."
  for RANK in 8 16 32; do
    nebius ai job create \
      --name "medisimplifier-ablation-r${RANK}" \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --image "$IMAGE" \
      --container-command python \
      --args "train.py --model openbio --epochs 1 --rank ${RANK} --modules q_v --data-size 7999 --seed 42" \
      --env HF_TOKEN="${HF_TOKEN}" \
      --platform "$PLATFORM" \
      --preset "$PRESET" \
      --disk-size "$DISK" \
      --volume "${BUCKET}:/output:rw" \
      --subnet-id "${NEBIUS_SUBNET_ID}" \
      --timeout 2h
  done

  echo "==> Submitting ablation jobs (Phase 2 — modules, r=32, data=8K)..."
  for MODULES in q_only q_v all_attn; do
    nebius ai job create \
      --name "medisimplifier-ablation-${MODULES}" \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --image "$IMAGE" \
      --container-command python \
      --args "train.py --model openbio --epochs 1 --rank 32 --modules ${MODULES} --data-size 7999 --seed 42" \
      --env HF_TOKEN="${HF_TOKEN}" \
      --platform "$PLATFORM" \
      --preset "$PRESET" \
      --disk-size "$DISK" \
      --volume "${BUCKET}:/output:rw" \
      --subnet-id "${NEBIUS_SUBNET_ID}" \
      --timeout 2h
  done

  echo "==> Submitting ablation jobs (Phase 3 — data size, r=32, all_attn)..."
  for DATA in 2000 4000 8000; do
    nebius ai job create \
      --name "medisimplifier-ablation-data${DATA}" \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --image "$IMAGE" \
      --container-command python \
      --args "train.py --model openbio --epochs 1 --rank 32 --modules all_attn --data-size ${DATA} --seed 42" \
      --env HF_TOKEN="${HF_TOKEN}" \
      --platform "$PLATFORM" \
      --preset "$PRESET" \
      --disk-size "$DISK" \
      --volume "${BUCKET}:/output:rw" \
      --subnet-id "${NEBIUS_SUBNET_ID}" \
      --timeout 2h
  done
  echo "9 ablation jobs submitted (~20 min each)."
fi

# ── Step 3: Evaluate (uses public HF adapters — no prior training needed) ─────
if [[ "$MODE" == "eval_only" || "$MODE" == "full" ]]; then
  echo "==> Submitting evaluation job (HF adapters, expected ROUGE-L 0.6638)..."
  nebius ai job create \
    --name medisimplifier-evaluate \
    --parent-id "${NEBIUS_PROJECT_ID}" \
    --image "$IMAGE" \
    --container-command python \
    --args "evaluate.py --model openbio --adapter-hf-repo chambul/MediSimplifier-LoRA-Adapter-Nebius --split test --output-dir /output/eval_results" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HF_HOME=/tmp/hf_cache \
    --platform "$PLATFORM" \
    --preset "$PRESET" \
    --disk-size "$DISK" \
    --subnet-id "${NEBIUS_SUBNET_ID}" \
    --volume "${BUCKET}:/output:rw" \
    --timeout 5h
fi

# ── Step 4: Deploy vLLM endpoint (merged model from HuggingFace) ──────────────
if [[ "$MODE" == "serve" || "$MODE" == "full" ]]; then
  echo "==> Creating vLLM endpoint (~10 min to load)..."
  nebius ai endpoint create \
    --name medisimplifier-vllm \
    --parent-id "${NEBIUS_PROJECT_ID}" \
    --image vllm/vllm-openai:latest \
    --args "-m vllm.entrypoints.openai.api_server --model chambul/MediSimplifier-OpenBioLLM-merged --port 8000 --dtype float16 --max-model-len 4096" \
    --container-port 8000 \
    --platform "$PLATFORM" \
    --preset "$PRESET" \
    --subnet-id "${NEBIUS_SUBNET_ID}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --public
  echo "After model loads: curl http://<endpoint-ip>:8000/v1/models"
fi

echo ""
echo "==> Monitor: nebius ai job list --parent-id \${NEBIUS_PROJECT_ID}"
echo "==> Pre-trained artifacts: https://huggingface.co/chambul/MediSimplifier-LoRA-Adapter-Nebius"
