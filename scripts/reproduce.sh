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

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "==> Preflight checks..."
if ! command -v nebius &>/dev/null; then
  echo "ERROR: nebius CLI not found. Install from https://console.nebius.com" && exit 1
fi
if ! nebius profile list &>/dev/null; then
  echo "ERROR: nebius CLI not authenticated. Run: nebius auth login" && exit 1
fi
if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami(token='${HF_TOKEN}')" &>/dev/null; then
  echo "ERROR: HF_TOKEN is invalid or expired. Get a new token at https://huggingface.co/settings/tokens" && exit 1
fi
echo "==> Preflight OK (nebius CLI authenticated, HF token valid)"

MODE="${1:-full}"
SMOKE="${2:-}"  # pass "smoke" as second arg for a quick 20-sample sanity check

IMAGE="chambul/medisimplifier:train-v27"
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
    --name medisimplifier-full-training \
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
      --args "train.py --model openbio --epochs 1 --rank ${RANK} --modules q_v --data-size 7999 --seed 42 --output-dir /output/ablation_r${RANK}_qv_8k" \
      --env HF_TOKEN="${HF_TOKEN}" \
      --platform "$PLATFORM" \
      --preset "$PRESET" \
      --disk-size "$DISK" \
      --volume "${BUCKET}:/output:rw" \
      --subnet-id "${NEBIUS_SUBNET_ID}" \
      --timeout 2h
  done

  echo "==> Submitting ablation jobs (Phase 2 — modules, r=32, data=8K)..."
  for MODULES in q_only all_attn; do
    nebius ai job create \
      --name "medisimplifier-ablation-${MODULES}" \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --image "$IMAGE" \
      --container-command python \
      --args "train.py --model openbio --epochs 1 --rank 32 --modules ${MODULES} --data-size 7999 --seed 42 --output-dir /output/ablation_r32_${MODULES}_8k" \
      --env HF_TOKEN="${HF_TOKEN}" \
      --platform "$PLATFORM" \
      --preset "$PRESET" \
      --disk-size "$DISK" \
      --volume "${BUCKET}:/output:rw" \
      --subnet-id "${NEBIUS_SUBNET_ID}" \
      --timeout 2h
  done

  echo "==> Submitting ablation jobs (Phase 3 — data size, r=32, all_attn)..."
  for DATA in 2000 4000; do
    nebius ai job create \
      --name "medisimplifier-ablation-data${DATA}" \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --image "$IMAGE" \
      --container-command python \
      --args "train.py --model openbio --epochs 1 --rank 32 --modules all_attn --data-size ${DATA} --seed 42 --output-dir /output/ablation_r32_all_attn_${DATA}" \
      --env HF_TOKEN="${HF_TOKEN}" \
      --platform "$PLATFORM" \
      --preset "$PRESET" \
      --disk-size "$DISK" \
      --volume "${BUCKET}:/output:rw" \
      --subnet-id "${NEBIUS_SUBNET_ID}" \
      --timeout 2h
  done
  echo "7 ablation jobs submitted in parallel (~20 min each): 3 rank × 1 modules + 2 modules + 2 data-size configs."
fi

# ── Step 3: Evaluate (uses public HF adapters — no prior training needed) ─────
if [[ "$MODE" == "eval_only" || "$MODE" == "full" ]]; then
  echo "==> Submitting evaluation job (HF adapters, expected ROUGE-L 0.6638)..."
  nebius ai job create \
    --name medisimplifier-evaluate \
    --parent-id "${NEBIUS_PROJECT_ID}" \
    --image "$IMAGE" \
    --container-command python \
    --args "evaluate.py --model openbio --adapter-hf-repo chambul/MediSimplifier-LoRA-Adapter-Nebius --split test --output-dir /output/eval_results${SMOKE:+ --limit 20 --fast}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HF_HOME=/tmp/hf_cache \
    --platform "$PLATFORM" \
    --preset "$PRESET" \
    --disk-size "$DISK" \
    --subnet-id "${NEBIUS_SUBNET_ID}" \
    --volume "${BUCKET}:/output:rw" \
    --timeout 5h
fi

# ── Auto-diff: verify ROUGE-L after eval_only ─────────────────────────────────
if [[ "$MODE" == "eval_only" && -z "$SMOKE" ]]; then
  echo "==> Waiting for eval job to complete before checking ROUGE-L..."
  echo "    (run manually after job completes:)"
  echo "    python3 -c \""
  echo "    import json; r=json.load(open('/tmp/eval_results/results.json'))"
  echo "    rouge=r['rouge_l']; expected=0.6638; tol=0.002"
  echo "    ok = abs(rouge-expected) <= tol"
  echo "    print(f'ROUGE-L: {rouge:.4f} (expected {expected} ± {tol}) — {\"OK\" if ok else \"MISMATCH\"}')"
  echo "    \""
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
