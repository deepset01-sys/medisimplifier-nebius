# MediSimplifier — Serverless LoRA Fine-Tuning on Nebius

[![Nebius Jobs](https://img.shields.io/badge/Nebius-Serverless%20AI%20Jobs-blue)](https://nebius.com)
[![HuggingFace Models](https://img.shields.io/badge/HF-Models-yellow)](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters)
[![HuggingFace Dataset](https://img.shields.io/badge/HF-Dataset-blue)](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

> **Repository structure:**
> - **Research & results:** [gd007/MediSimplifier](https://github.com/gd007/MediSimplifier) —
>   original Technion course project with notebooks, IEEE paper, and all results
> - **Nebius pipeline:** [deepset01-sys/medisimplifier-nebius](https://github.com/deepset01-sys/medisimplifier-nebius) —
>   this repository: serverless Jobs + Endpoints on Nebius

> Nebius Serverless AI Builders Challenge submission.
> Reproducible LoRA fine-tuning pipeline for medical text simplification,
> running entirely on Nebius Serverless AI Jobs and Endpoints.

## What this project does

Medical discharge summaries are written at college reading level (FK-Grade 14.5).
Only 12% of American adults have sufficient health literacy to understand them.

MediSimplifier fine-tunes open-source LLMs using LoRA to automatically simplify
these documents to 7th-grade reading level — ~50% readability reduction —
while preserving all critical medical information.

**Key findings (challenging the conventional recommendation of r=4-8 (Hu et al. 2021)):**

| Finding | Result |
|---------|--------|
| Optimal LoRA rank | r=32 outperforms r=4-8 recommended by Hu et al. 2021 |
| Optimal modules | all_attn (Q+K+V+O) outperforms standard Q+V |
| Ranking reversal | Worst zero-shot model becomes best fine-tuned (+157%) |
| Readability | FK-Grade 14.5 → 6.91, all differences p<0.001 |
| Data efficiency | 4K samples achieves 97% of 8K performance |
| Baseline-improvement correlation | r ≈ -0.998 — lower zero-shot = higher gain |
| Total compute | 18 ablation runs + 3 full training runs, ~7.5 GPU hours |

## Results

| Model | ROUGE-L | SARI | BERTScore | FK-Grade | Improvement |
|-------|---------|------|-----------|----------|-------------|
| OpenBioLLM-8B | 0.6749 [0.6705–0.6793] | 74.64 | 0.9498 | 7.16 | +157.3% |
| Mistral-7B | 0.6491 [0.6445–0.6537] | 73.79 | 0.9464 | 6.91 | +65.9% |
| BioMistral-7B-DARE | 0.6318 [0.6272–0.6365] | 73.01 | 0.9439 | 6.95 | +53.3% |

95% CIs from bootstrap (n=10,000). All pairwise ROUGE-L differences significant at p<0.001.
All results use seed=42. Bootstrap CIs computed with n=10,000 resamples.

> **Note on FK-Grade target:** The original research target was FK ≤ 6.0.
> Best achieved: 6.91 (Mistral-7B). The gap of ~0.91 grade levels
> reflects the inherent tension between medical accuracy preservation
> and maximum simplification.

## Baseline vs Fine-Tuned Results

### Zero-Shot Baseline (no fine-tuning, 1,001 test samples)

| Model | ROUGE-L | SARI | BERTScore | FK-Grade |
|-------|---------|------|-----------|----------|
| OpenBioLLM-8B | 0.2623 | 36.98 | 0.637 | 12.53 |
| Mistral-7B | 0.3912 | 46.38 | 0.734 | 10.60 |
| BioMistral-7B-DARE | 0.4120 | 51.91 | 0.743 | 9.52 |

### After LoRA Fine-Tuning

| Model | ROUGE-L | SARI | BERTScore | FK-Grade | Improvement |
|-------|---------|------|-----------|----------|-------------|
| OpenBioLLM-8B | 0.6749 [0.6705–0.6793] | 74.64 | 0.9498 | 7.16 | +157.3% |
| Mistral-7B | 0.6491 [0.6445–0.6537] | 73.79 | 0.9464 | 6.91 | +65.9% |
| BioMistral-7B-DARE | 0.6318 [0.6272–0.6365] | 73.01 | 0.9439 | 6.95 | +53.3% |

**Key finding:** OpenBioLLM-8B had the *lowest* zero-shot score (0.2623)
but achieved the *highest* fine-tuned score (0.6749) — a full ranking
reversal. All pairwise differences significant at p<0.001 (bootstrap n=10,000).

### Nebius Reproduction Results

The full evaluation pipeline was run on Nebius Serverless Jobs
(H100 NVLink, June 2026, 1,001 test samples):

| Metric | Original Research | Nebius Reproduction | Delta |
|--------|-------------------|---------------------|-------|
| ROUGE-L | 0.6749 | 0.6638 | -1.6% |
| SARI | 74.64 | 73.49 | -1.5% |
| BERTScore | 0.9498 | 0.9460 | -0.4% |
| FK-Grade | 7.16 | 7.33 | +0.17 |

> All four metrics reproduce within 1.6% of the original research,
> confirming full pipeline reproducibility. Minor variance is explained
> by floating-point non-determinism across GPU hardware (original:
> H200 SXM, reproduction: H100 NVLink) and generation-time sampling.
> Evaluation Job: `medisimplifier-evaluation-full2` (1,001 samples,
> ROUGE-L + SARI + BERTScore + FK-Grade)

## Visualizations

Key figures from the research (available in the
[original repository](https://github.com/gd007/MediSimplifier/tree/main/results/figures)):

| Figure | Description |
|--------|-------------|
| `ranking_reversal.png` | Full ranking reversal — worst zero-shot → best fine-tuned |
| `ablation_study_summary.png` | All 3 ablation phases in one chart |
| `confidence_intervals.png` | 95% bootstrap CIs for all models |
| `baseline_vs_finetuned_metrics.png` | Before/after across all 4 metrics |
| `significance_matrix.png` | Pairwise statistical significance |
| `error_analysis_dashboard.png` | Qualitative error analysis |
| `model_performance_heatmap.png` | Full metrics heatmap |

## Ablation Study Results

All ablation runs: 1 epoch, OpenBioLLM-8B base, evaluated on held-out test set (1,001 samples).

**Phase 1 — LoRA Rank** (modules=q+v, data=8K)

| LoRA Rank | Modules | Data Size | ROUGE-L |
|-----------|---------|-----------|---------|
| r=8       | q+v     | 8K        | 0.6033  |
| r=16      | q+v     | 8K        | 0.6080  |
| **r=32**  | **q+v** | **8K**    | **0.6183** ← winner |

**Phase 2 — Target Modules** (rank=32, data=8K)

| LoRA Rank | Modules      | Data Size | ROUGE-L |
|-----------|--------------|-----------|---------|
| r=32      | q only       | 8K        | 0.6006  |
| r=32      | q+v          | 8K        | 0.6192  |
| **r=32**  | **all_attn** | **8K**    | **0.6357** ← winner |

**Phase 3 — Data Size** (rank=32, modules=all_attn)

| LoRA Rank | Modules  | Data Size | ROUGE-L |
|-----------|----------|-----------|---------|
| r=32      | all_attn | 2K        | 0.6014  |
| r=32      | all_attn | 4K        | 0.6198  |
| **r=32**  | **all_attn** | **8K** | **0.6345** ← winner |

Winner configuration: **r=32, all_attn, 8K** → used for full 3-epoch training → final ROUGE-L 0.6749.

## How it runs on Nebius

Nebius Serverless AI Jobs handle training and ablation.
Nebius Serverless AI Endpoints serve live inference.

Pipeline:

    Dataset (HuggingFace: GuyDor007/medisimplifier-dataset)
        |
        v
    Nebius Job: Ablation (9 parallel jobs — rank x modules x data size)
        |
        v
    Nebius Job: Full training (r=32, all_attn, 3 epochs, H100)
        |
        v
    Nebius Job: Evaluation (ROUGE-L, SARI, BERTScore, FK-Grade)
        |
        v
    Nebius Endpoint: POST /simplify -> simplified text

### Hardware and cost

| Step | GPU | Time | Cost |
|------|-----|------|------|
| Ablation x9 parallel | H100 | ~20 min each | ~$15 |
| Full training | H100 NVLink | ~70 min | ~$22 |
| Evaluation | H100 NVLink | ~45 min | ~$5 |
| Total | | | ~$42 |

> Original research hardware: RunPod H200 SXM (~90 min/model across 3 GPUs).
> Nebius reproduction uses H100 NVLink (~70 min, single GPU per job).

## Reproduce step by step

**Environment:** Python 3.11 · CUDA 12.1 · PyTorch 2.1.0

### 0. Install and authenticate Nebius CLI

Install:

    pip install nebius

Authenticate:

    nebius auth login

Verify:

    nebius profile list

### Prerequisites

```bash
export NEBIUS_PROJECT_ID=project-e00g1ev2pr00wjxv40r6ga
export NEBIUS_SUBNET_ID=vpcsubnet-e00jsdqfjrz04ygxc0
```

### 1. Clone and install

    git clone https://github.com/deepset01-sys/medisimplifier-nebius.git
    cd medisimplifier-nebius
    pip install -r requirements.txt

### 2. Submit training job (Nebius CLI)

```bash
nebius ai job create \
  --name medisimplifier-full-train \
  --parent-id ${NEBIUS_PROJECT_ID} \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  --container-command sh \
  --args "-c 'pip install transformers peft datasets trl accelerate bitsandbytes sentencepiece huggingface-hub rouge-score textstat --quiet && pip install git+https://github.com/feralvam/easse.git --quiet && git clone https://github.com/deepset01-sys/medisimplifier-nebius.git /workspace && python /workspace/src/train.py --model openbio --epochs 3 --rank 32 --modules all_attn'" \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --disk-size 250Gi \
  --subnet-id ${NEBIUS_SUBNET_ID} \
  --timeout 5h
```

> **Runtime setup:** Jobs use the official `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
> base image. Dependencies are installed at job startup via pip, and the pipeline
> code is cloned from this repository. This keeps the setup fully reproducible
> with no private image dependencies.

> Alternatively, submit via Nebius Console → AI Services → Jobs → Create job
> using the config in `jobs/job_train.yaml`.

Our training run:

- **Job name:** `medisimplifier-full-training`
- **GPU:** H100 SXM · 1 GPU · 3 epochs · ~70 min

### 3. Run ablation study (9 parallel jobs)

```bash
# Phase 1 — LoRA rank
for RANK in 8 16 32; do
  nebius ai job create \
    --name medisimplifier-ablation-r${RANK} \
    --parent-id ${NEBIUS_PROJECT_ID} \
    --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
    --container-command sh \
    --args "-c 'pip install transformers peft datasets trl accelerate bitsandbytes sentencepiece huggingface-hub rouge-score textstat --quiet && pip install git+https://github.com/feralvam/easse.git --quiet && git clone https://github.com/deepset01-sys/medisimplifier-nebius.git /workspace && python /workspace/src/train.py --model openbio --epochs 1 --rank ${RANK} --modules q_v --data-size 2000'" \
    --platform gpu-h100-sxm \
    --preset 1gpu-16vcpu-200gb \
    --disk-size 250Gi \
    --subnet-id ${NEBIUS_SUBNET_ID} \
    --timeout 2h
done
```

### 4. Evaluate

```bash
nebius ai job create \
  --name medisimplifier-evaluate \
  --parent-id ${NEBIUS_PROJECT_ID} \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  --container-command sh \
  --args "-c 'pip install transformers peft datasets trl accelerate bitsandbytes sentencepiece huggingface-hub rouge-score textstat --quiet && pip install git+https://github.com/feralvam/easse.git --quiet && git clone https://github.com/deepset01-sys/medisimplifier-nebius.git /workspace && python /workspace/src/evaluate.py --model openbio --adapter-path /mnt/adapters/full_training --split test --output-dir /mnt/adapters/eval_results'" \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --disk-size 250Gi \
  --subnet-id ${NEBIUS_SUBNET_ID} \
  --timeout 5h
```

Our evaluation run:

- **Job name:** `medisimplifier-evaluation-prompt`
- **GPU:** H100 SXM · 1 GPU · ~45 min

### 5. Deploy live endpoint

Via Nebius Console → AI Services → Endpoints → Create endpoint,
or using the config in `jobs/endpoint_serve.yaml`.

The endpoint exposes:

    POST http://<endpoint-ip>:8000/simplify
    {"text": "Patient presented with acute MI..."}
    → {"simplified": "The patient had a heart attack...",
       "model": "aaditya/Llama3-OpenBioLLM-8B",
       "adapter": "/mnt/adapters/full_training"}

### 6. Call the live endpoint

    curl -X POST http://<endpoint-ip>:8000/simplify \
      -H "Content-Type: application/json" \
      -d '{"text": "Patient presented with acute myocardial infarction..."}'

To redeploy: see `jobs/endpoint_serve.yaml` and step 5 above.

## Live Demo

The endpoint was live during development and returned:

    Input:  "Patient presented with acute myocardial infarction
             and was administered thrombolytic therapy."
    Output: "The patient came in with a heart attack and received
             medicine to break up blood clots."

To redeploy: see `jobs/endpoint_serve.yaml` and step 5 above.
Full response: `{"simplified": "...", "model": "aaditya/Llama3-OpenBioLLM-8B", "adapter": "/mnt/adapters/full_training"}`

## Qualitative Example

**Input (FK-Grade 16.2):**
> "The patient was admitted with acute decompensated heart failure,
> presenting with dyspnea, orthopnea, and bilateral lower extremity
> edema. Echocardiography revealed an ejection fraction of 25%."

**Output — OpenBioLLM-8B (FK-Grade 6.8):**
> "The patient came in with severe heart failure. They had trouble
> breathing, couldn't lie flat, and had swelling in both legs.
> A heart ultrasound showed the heart was only pumping at 25%
> of its normal strength."

**Preserved:** diagnosis, symptoms, test result, ejection fraction value
**Simplified:** all medical terms replaced with plain language

## Project structure

    src/
      train.py          LoRA training — runs as Nebius Job
      evaluate.py       Metrics: ROUGE-L, SARI, BERTScore, FK-Grade
      serve.py          FastAPI inference — runs as Nebius Endpoint
    docker/
      Dockerfile.train  Training image
      Dockerfile.serve  Serving image
    jobs/
      job_train.yaml       Full training job config
      job_ablation.yaml    Parametrized ablation job config
      job_evaluate.yaml    Evaluation job config
      endpoint_serve.yaml  Endpoint deployment config
    requirements.txt

## Key Configuration

All jobs use the `nebius ai job create` CLI. The training job parameters:

| Parameter | Value |
|-----------|-------|
| Image | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (deps installed at startup) |
| Platform | `gpu-h100-sxm` |
| Preset | `1gpu-16vcpu-200gb` |
| Disk | `250Gi` |
| Timeout | `5h` |
| LoRA rank | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj (all_attn) |
| Epochs | 3 |
| Batch size | 4 (grad_accum=4, effective=16) |
| Learning rate | 2e-4 (cosine, warmup 3%) |
| LoRA alpha (α) | 64 (= 2 × rank) |
| rsLoRA | True |
| Dropout | 0.05 |
| Trainable parameters | 27.3M (0.38% of total) |
| Random seed | 42 |

## Job & Endpoint Configs

### Training Job (jobs/job_train.yaml)

```yaml
name: medisimplifier-full-train
description: "MediSimplifier full LoRA training - OpenBioLLM-8B, r=32, all_attn, 3 epochs"

resources:
  gpu: H100
  gpu_count: 1

docker:
  image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

env:
  HF_TOKEN: "${HF_TOKEN}"
  PYTHONUNBUFFERED: "1"

command: >
  python train.py
  --model openbio
  --epochs 3
  --rank 32
  --modules all_attn
  --data-size 7999
  --output-dir /output/adapter

output:
  path: /output/adapter
```

### Evaluation Job (jobs/job_evaluate.yaml)

```yaml
name: medisimplifier-evaluate
description: "MediSimplifier evaluation — ROUGE-L, SARI, BERTScore, FK-Grade"

resources:
  gpu: H100
  gpu_count: 1

docker:
  image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

env:
  HF_TOKEN: "${HF_TOKEN}"
  PYTHONUNBUFFERED: "1"

command: >
  python evaluate.py
  --model ${MODEL:-openbio}
  --adapter-path ${ADAPTER_PATH:-/mnt/adapters/full_training}
  --split test
  --output-dir /output/eval

output:
  path: /output/eval
```

### Endpoint (jobs/endpoint_serve.yaml)

```yaml
name: medisimplifier-serve
description: "MediSimplifier inference endpoint — POST /simplify → simplified text"

resources:
  gpu: H100
  gpu_count: 1

docker:
  image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

env:
  HF_HOME: "/tmp/hf_cache"
  PYTHONUNBUFFERED: "1"

entrypoint: >
  sh -c "apt-get update -qq && apt-get install -y git -qq &&
  pip install transformers peft accelerate bitsandbytes
  sentencepiece huggingface-hub fastapi uvicorn --quiet &&
  git clone https://github.com/deepset01-sys/medisimplifier-nebius.git /workspace &&
  python /workspace/src/serve.py"

ports:
  - 8000

volumes:
  - bucket: medisimplifier-adapters
    mount: /mnt/adapters
```

## Dependencies

<details>
<summary>requirements.txt (click to expand)</summary>

```
# MediSimplifier Dependencies
# Tested on Python 3.9+ with CUDA 12.4

# Core ML
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.7.0
bitsandbytes>=0.41.0

# Evaluation Metrics
evaluate>=0.4.0
rouge-score>=0.1.2
bert-score>=0.3.13
textstat>=0.7.3
sacrebleu>=2.3.0

# SARI metric (install from GitHub)
# pip install git+https://github.com/feralvam/easse.git

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
typing_extensions>=4.10.0

# Jupyter
jupyter>=1.0.0
ipywidgets>=8.0.0
```

</details>

## Dataset and models

| Resource | Link |
|----------|------|
| Dataset | GuyDor007/medisimplifier-dataset — 10K samples, public |
| Models | GuyDor007/MediSimplifier-LoRA-Adapters — 3 adapters, public |

Dataset: Asclepius-Synthetic-Clinical-Notes (public, anonymized synthetic notes).
No real patient data. HCLS compliant.

## License

Apache 2.0
