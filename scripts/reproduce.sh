#!/bin/bash
set -e

echo "MediSimplifier — Full Reproduction Pipeline"
echo "============================================"

# Prerequisites
echo "Step 0: Checking prerequisites..."
nebius profile list
echo "Nebius CLI configured ✓"

# Create bucket
echo "Step 1: Creating storage bucket..."
nebius storage bucket create \
  --name medisimplifier-adapters \
  --parent-id ${NEBIUS_PROJECT_ID} 2>/dev/null || echo "Bucket already exists"

# Submit training job
echo "Step 2: Submitting training job..."
nebius ai job create \
  --name medisimplifier-full-train \
  --parent-id ${NEBIUS_PROJECT_ID} \
  --image chambul/medisimplifier:train-v4 \
  --container-command python \
  --args "train.py --model openbio --epochs 3 --rank 32 --modules all_attn --data-size 8000 --seed 42" \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --disk-size 250Gi \
  --subnet-id ${NEBIUS_SUBNET_ID} \
  --volume medisimplifier-adapters:/output:rw \
  --env HF_TOKEN=${HF_TOKEN} \
  --timeout 2h

echo "Training job submitted. Wait for completion (~70 min), then run:"
echo "Step 3: nebius ai job create ... evaluate.py"
echo "Step 4: python src/merge_adapter.py + aws s3 sync"
echo "Step 5: Deploy endpoint_vllm.yaml via Nebius Console"
echo ""
echo "Or skip training: use public adapters at GuyDor007/MediSimplifier-LoRA-Adapters"
