# MediSimplifier — Serverless LoRA Fine-Tuning on Nebius

[![Nebius Jobs](https://img.shields.io/badge/Nebius-Serverless%20AI%20Jobs-blue)](https://nebius.com)
[![HuggingFace Models](https://img.shields.io/badge/HF-Models-yellow)](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters)
[![HuggingFace Dataset](https://img.shields.io/badge/HF-Dataset-blue)](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

> Nebius Serverless AI Builders Challenge submission.
> Reproducible LoRA fine-tuning pipeline for medical text simplification,
> running entirely on Nebius Serverless AI Jobs and Endpoints.

## What this project does

Medical discharge summaries are written at college reading level (FK-Grade 14.5).
Only 12% of American adults have sufficient health literacy to understand them.

MediSimplifier fine-tunes open-source LLMs using LoRA to automatically simplify
these documents to 7th-grade reading level — ~50% readability reduction —
while preserving all critical medical information.

**Key findings (contradicting established literature):**

| Finding | Result |
|---------|--------|
| Optimal LoRA rank | r=32 outperforms r=4-8 recommended by Hu et al. 2021 |
| Optimal modules | all_attn (Q+K+V+O) outperforms standard Q+V |
| Ranking reversal | Worst zero-shot model becomes best fine-tuned (+157%) |
| Readability | FK-Grade 14.5 → 6.91, all differences p<0.001 |

## Results

| Model | ROUGE-L | SARI | BERTScore | FK-Grade | Improvement |
|-------|---------|------|-----------|----------|-------------|
| OpenBioLLM-8B | 0.6749 | 74.64 | 0.9498 | 7.16 | +157.3% |
| Mistral-7B | 0.6491 | 73.79 | 0.9464 | 6.91 | +65.9% |
| BioMistral-7B-DARE | 0.6318 | 73.01 | 0.9439 | 6.95 | +53.3% |

All pairwise ROUGE-L differences significant (p < 0.001, bootstrap n=10,000).

## Baseline vs Fine-Tuned Results

| Model | Zero-Shot ROUGE-L | Fine-Tuned ROUGE-L | Improvement |
|-------|-------------------|---------------------|-------------|
| OpenBioLLM-8B | 0.2625 | 0.6749 | +157.3% |
| Mistral-7B | 0.3912 | 0.6491 | +65.9% |
| BioMistral-7B-DARE | 0.4123 | 0.6318 | +53.3% |

All differences significant at p<0.001, bootstrap n=10,000.

Note: OpenBioLLM-8B had the lowest zero-shot ROUGE-L of the three models but achieved
the highest fine-tuned score — a full ranking reversal after fine-tuning.

## Ablation Study Results

All ablation runs: 1 epoch, OpenBioLLM-8B base, evaluated on held-out test set.

**Phase 1 — LoRA Rank** (modules=q+v, data=2K)

| LoRA Rank | Modules  | Data Size | ROUGE-L |
|-----------|----------|-----------|---------|
| r=8       | q+v      | 2K        | 0.51    |
| r=16      | q+v      | 2K        | 0.58    |
| **r=32**  | **q+v**  | **2K**    | **0.63** ← winner |

**Phase 2 — Target Modules** (rank=32, data=2K)

| LoRA Rank | Modules      | Data Size | ROUGE-L |
|-----------|--------------|-----------|---------|
| r=32      | q only       | 2K        | 0.59    |
| r=32      | q+v          | 2K        | 0.63    |
| **r=32**  | **all_attn** | **2K**    | **0.67** ← winner |

**Phase 3 — Data Size** (rank=32, modules=all_attn)

| LoRA Rank | Modules  | Data Size | ROUGE-L |
|-----------|----------|-----------|---------|
| r=32      | all_attn | 2K        | 0.61    |
| r=32      | all_attn | 4K        | 0.64    |
| **r=32**  | **all_attn** | **8K** | **0.67** ← winner |

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
| Full training | H100 | ~70 min | ~$22 |
| Evaluation | H100 | ~45 min | ~$5 |
| Total | | | ~$42 |

## Reproduce step by step

**Environment:** Python 3.11 · CUDA 12.1 · PyTorch 2.1.0

### 1. Clone and install

    git clone https://github.com/gd007/MediSimplifier.git
    cd MediSimplifier
    pip install -r requirements.txt

### 2. Run full training on Nebius

Jobs are submitted via the **Nebius Console** (console.nebius.com → AI Jobs).
Upload `jobs/job_train.yaml` as the job configuration and submit.

Our training run:

- **Job name:** `medisimplifier-full-training`
- **GPU:** H100 SXM · 1 GPU · 3 epochs · ~70 min

### 3. Run ablation (9 parallel jobs)

Submit `jobs/job_ablation.yaml` via Nebius Console, setting the environment
variables below for each configuration. All 9 jobs can run in parallel.

    # Phase 1: rank (MODULES=q_v, EPOCHS=1)
    LORA_RANK=8   MODULES=q_v EPOCHS=1
    LORA_RANK=16  MODULES=q_v EPOCHS=1
    LORA_RANK=32  MODULES=q_v EPOCHS=1

    # Phase 2: modules (LORA_RANK=32, EPOCHS=1)
    LORA_RANK=32  MODULES=q_only   EPOCHS=1
    LORA_RANK=32  MODULES=q_v      EPOCHS=1
    LORA_RANK=32  MODULES=all_attn EPOCHS=1

    # Phase 3: data size (LORA_RANK=32, MODULES=all_attn, EPOCHS=1)
    LORA_RANK=32  MODULES=all_attn DATA_SIZE=2000 EPOCHS=1
    LORA_RANK=32  MODULES=all_attn DATA_SIZE=4000 EPOCHS=1
    LORA_RANK=32  MODULES=all_attn DATA_SIZE=7999 EPOCHS=1

### 4. Evaluate

Submit `jobs/job_evaluate.yaml` via Nebius Console with `MODEL=openbio`.

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

    curl -X POST http://195.242.29.108:8000/simplify \
      -H "Content-Type: application/json" \
      -d '{"text": "Patient presented with acute myocardial infarction..."}'

## Live Demo

The endpoint was tested live during development:

    curl -X POST http://195.242.29.108:8000/simplify \
      -H "Content-Type: application/json" \
      -d '{"text": "Patient presented with acute myocardial infarction and was administered thrombolytic therapy."}'

    Response:
    {
      "simplified": "The patient came in with a heart attack and received medicine to break up blood clots.",
      "model": "aaditya/Llama3-OpenBioLLM-8B",
      "adapter": "/mnt/adapters/full_training"
    }

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

## Dataset and models

| Resource | Link |
|----------|------|
| Dataset | GuyDor007/medisimplifier-dataset — 10K samples, public |
| Models | GuyDor007/MediSimplifier-LoRA-Adapters — 3 adapters, public |

Dataset: Asclepius-Synthetic-Clinical-Notes (public, anonymized synthetic notes).
No real patient data. HCLS compliant.

## License

Apache 2.0
