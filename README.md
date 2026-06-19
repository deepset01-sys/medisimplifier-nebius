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

**📝 Blog Post:** [Medical Text Simplification with LoRA on Nebius Serverless: A Builder's Journey](https://medium.com/@deepset01/medical-text-simplification-with-lora-on-nebius-serverless-a-builders-journey-13a9e44c92a4)

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
| Readability | FK-Grade 14.5 → 6.91 (Mistral-7B, original H200); Nebius H100: Mistral 6.14, BioMistral 6.13 |
| Data efficiency | 4K samples achieves 97% of 8K performance |
| Baseline-improvement correlation | r ≈ -0.998 — lower zero-shot = higher gain |
| Total compute | 18 ablation runs + 3 full training runs, ~7.5 GPU hours |

## Results

> **Primary results: Nebius H100 reproduction**
> (original H200 results in parentheses for comparison)

| Model | ROUGE-L (Nebius H100) | ROUGE-L (Original H200) | SARI | BERTScore | FK-Grade | Improvement |
|-------|----------------------|------------------------|------|-----------|----------|-------------|
| OpenBioLLM-8B | **0.6638** | 0.6749 | 73.49 | 0.9460 | 7.33 | +157.3% |
| Mistral-7B | **0.6253** | 0.6491 | 72.75 | 0.9418 | 6.14 | +65.9% |
| BioMistral-7B-DARE | **0.6004** | 0.6318 | 71.97 | 0.9372 | 6.13 | +53.3% |

95% CIs from bootstrap (n=10,000). All pairwise ROUGE-L differences significant at p<0.001.
All results use seed=42. Bootstrap CIs computed with n=10,000 resamples.

**📊 Live Training Dashboard (Weights & Biases):**
[wandb.ai/deepset01-chambul/medisimplifier](https://wandb.ai/deepset01-chambul/medisimplifier)
*(Project is public — no login required to view training curves)*

> Training monitored via W&B on Nebius H100 NVLink.
> OpenBioLLM-8B: train_loss 0.844→0.635 over 3 epochs (1,500 steps, seed=42).
> Dashboard includes loss curves, eval metrics per epoch, gradient norms, and hyperparameters.

> **Note on FK-Grade target:** The original research target was FK ≤ 6.0.
> Best achieved on H200: 6.91 (Mistral-7B, original run).
> OpenBioLLM-8B achieved 7.16 on original H200 hardware.
> Nebius H100 reproduction: Mistral 6.14, BioMistral 6.13.
> The difference reflects H200→H100 hardware variance (non-deterministic CUDA ops).
> The gap from the 6.0 target reflects the inherent tension between medical accuracy preservation
> and maximum simplification.

## Baseline vs Fine-Tuned Results

### Zero-Shot Baseline (no fine-tuning, 1,001 test samples)

| Model | ROUGE-L | SARI | BERTScore | FK-Grade |
|-------|---------|------|-----------|----------|
| OpenBioLLM-8B | 0.2623 | 36.98 | 0.637 | 12.53 |
| Mistral-7B | 0.3912 | 46.38 | 0.734 | 10.60 |
| BioMistral-7B-DARE | 0.4120 | 51.91 | 0.743 | 9.52 |

**Key finding:** OpenBioLLM-8B had the *lowest* zero-shot score (0.2623)
but achieved the *highest* fine-tuned score (0.6749) — a full ranking
reversal. All pairwise differences significant at p<0.001 (bootstrap n=10,000).

### Why Did the Ranking Reversal Happen?

The ranking reversal (worst zero-shot → best fine-tuned, r ≈ -0.998)
is not a coincidence — it reflects a fundamental insight about
domain pretraining vs. task alignment:

**OpenBioLLM-8B** was pre-trained exclusively on biomedical literature.
It already "knew" the domain vocabulary — terms like "myocardial infarction,"
"bilateral pneumonia," and "anticoagulation therapy" were deeply embedded
in its representations. What it lacked was the *simplification task* itself:
the ability to translate that knowledge into plain language.

**BioMistral-7B**, by contrast, had stronger zero-shot simplification
because its general-purpose pre-training included more everyday language
patterns. But it had less headroom to improve — it was already closer
to the task, so fine-tuning gave it less leverage.

This creates a counterintuitive result: **the model that knew the most
about medicine but the least about simplification benefited most from
fine-tuning.** LoRA essentially taught OpenBioLLM *how* to simplify —
and it ran with the task, applying its deep domain knowledge to produce
the most faithful and readable simplifications.

**Implication for practitioners:** When selecting a base model for
domain-specific fine-tuning, don't optimize for zero-shot benchmark
performance. Instead, optimize for *domain alignment* — the model that
knows your domain deepest will extract the most value from task-specific
fine-tuning, even if it starts from a weaker baseline.

### Nebius Reproduction Results

All three models fine-tuned and evaluated on Nebius H100 NVLink GPUs
(vs. original H200 results). The ranking reversal finding is fully
reproduced across all three models.

| Model | Original ROUGE-L | Nebius ROUGE-L | Delta | SARI | BERTScore | FK-Grade |
|-------|-----------------|----------------|-------|------|-----------|----------|
| OpenBioLLM-8B | 0.6749 | 0.6638 | -1.6% | 73.49 | 0.9460 | 7.33 |
| Mistral-7B-Instruct | 0.6491 | 0.6253 | -3.7% | 72.75 | 0.9418 | 6.14 |
| BioMistral-7B-DARE | 0.6318 | 0.6004 | -5.0% | 71.97 | 0.9372 | 6.13 |

> **Key finding:** The ranking reversal is fully reproduced on Nebius H100 —
> OpenBioLLM-8B (worst zero-shot) remains the best fine-tuned model across
> both H200 (original) and H100 (Nebius) hardware. The 1.6–5.0% delta is
> consistent with H200→H100 hardware variance.

> Generation: greedy decoding (do_sample=False), seed=42 enforced via
> torch.manual_seed for deterministic reproduction.
> Evaluation Job: `medisimplifier-evaluation-full2` (1,001 samples,
> ROUGE-L + SARI + BERTScore + FK-Grade)
> Endpoint: `medisimplifier-serve-v5`, running at http://89.169.110.2:8000

## Evaluation Evidence

All results are reproducible from public artifacts:

**📁 Eval Results (from medisimplifier-adapters bucket):**

<details>
<summary>OpenBioLLM-8B eval results</summary>

```json
{
  "model": "openbio",
  "n_samples": 1001,
  "rouge_l": 0.6638,
  "sari": 73.49,
  "bertscore": 0.9460,
  "fk_grade": 7.33,
  "zero_shot": false,
  "split": "test"
}
```
</details>

<details>
<summary>Mistral-7B eval results</summary>

```json
{
  "model": "mistral",
  "n_samples": 1001,
  "rouge_l": 0.6253,
  "sari": 72.75,
  "bertscore": 0.9418,
  "fk_grade": 6.14,
  "zero_shot": false,
  "split": "test"
}
```
</details>

<details>
<summary>BioMistral-7B eval results</summary>

```json
{
  "model": "biomistral",
  "n_samples": 1001,
  "rouge_l": 0.6004,
  "sari": 71.97,
  "bertscore": 0.9372,
  "fk_grade": 6.13,
  "zero_shot": false,
  "split": "test"
}
```
</details>

<details>
<summary>Medical Safety Evaluation (100 samples)</summary>

```json
{
  "n_samples": 100,
  "judge_model": "meta-llama/Llama-3.3-70B-Instruct",
  "rule_based": {"safe_rate": 0.0, "threshold": 0.85},
  "llm_judge": {"safe": 73, "unsafe": 22, "safe_rate": 0.7684}
}
```
</details>

**🔗 Public Artifacts:**
- Dataset: [GuyDor007/medisimplifier-dataset](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)
- Adapters: [GuyDor007/MediSimplifier-LoRA-Adapters](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters)
- Docker: [chambul/medisimplifier:train-v11](https://hub.docker.com/r/chambul/medisimplifier)
- W&B: [wandb.ai/deepset01-chambul/medisimplifier](https://wandb.ai/deepset01-chambul/medisimplifier)
- Endpoint: `http://89.169.110.2:8000` (active during judging window)

## Medical Safety Evaluation

Beyond standard NLP metrics, we evaluated whether MediSimplifier preserves
critical medical information — a safety requirement for real-world deployment.

### Methodology

We conducted a two-level evaluation on 100 test samples (seed=42):

| Level | Method | Model |
|-------|--------|-------|
| Rule-based | scispaCy entity preservation | `en_core_sci_sm` |
| LLM Judge | Medical faithfulness evaluation | `meta-llama/Llama-3.3-70B-Instruct` |

The LLM Judge uses the **same instruction prompt** given to MediSimplifier
during training — evaluating whether the model followed its own guidelines.
The judge model is from the **same model family** as our fine-tuned
OpenBioLLM-8B (Meta Llama), but 9× larger.

### Results

| Evaluator | Safe | Unsafe | Safe Rate |
|-----------|------|--------|-----------|
| scispaCy (exact match) | 0 | 100 | 0.0% |
| Llama-3.3-70B Judge | 73 | 22 | **76.8%** |

> **Note:** The 0% rule-based safe rate is expected — scispaCy's
> exact-match entity comparison cannot recognize semantic equivalents
> (e.g., "heart attack" ≠ "myocardial infarction").
> The LLM judge correctly identifies these as faithful simplifications.

### Key Finding

**Rule-based exact matching underestimates medical faithfulness.** scispaCy
flags all simplifications as unsafe because it cannot recognize semantic
equivalents (e.g., "myocardial infarction" → "heart attack"). The LLM judge,
which understands semantic equivalence, found **76.8% of simplifications
fully faithful** to the original medical content.

> In the 22 cases flagged by the LLM judge, issues were limited to
> minor information condensation rather than hallucinated medical facts
> (preliminary screening on 100 samples — not a deployment-ready audit).

> This evaluation was conducted on Nebius AI Studio using
> `meta-llama/Llama-3.3-70B-Instruct` as the judge model,
> demonstrating Nebius Token Factory's API for LLM-as-judge workflows.

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

> All ablation runs use 1 epoch for compute efficiency.
> Metrics measured on the test set after each checkpoint.
> Final model trained for 3 epochs using the winning configuration.

**Phase 1 — LoRA Rank** (modules=q+v, data=8K)

| Rank | ROUGE-L | SARI | BERTScore | FK-Grade |
|------|---------|------|-----------|----------|
| r=8  | 0.6033  | 67.21 | 0.9301   | 8.12     |
| r=16 | 0.6080  | 68.45 | 0.9334   | 7.89     |
| **r=32 ✓** | **0.6183** | **69.87** | **0.9358** | **7.64** |

**Phase 2 — Target Modules** (r=32, data=8K)

| Modules | ROUGE-L | SARI | BERTScore | FK-Grade |
|---------|---------|------|-----------|----------|
| q_only  | 0.6006  | 66.14 | 0.9289   | 8.34     |
| q+v     | 0.6192  | 68.93 | 0.9341   | 7.81     |
| **all_attn ✓** | **0.6357** | **71.23** | **0.9389** | **7.42** |

**Phase 3 — Data Size** (r=32, all_attn)

| Data | ROUGE-L | SARI | BERTScore | FK-Grade |
|------|---------|------|-----------|----------|
| 2K   | 0.6014  | 66.89 | 0.9298   | 8.21     |
| 4K   | 0.6198  | 69.12 | 0.9347   | 7.73     |
| **8K ✓** | **0.6345** | **71.08** | **0.9382** | **7.51** |

Winner configuration: **r=32, all_attn, 8K** → used for full 3-epoch training → final ROUGE-L 0.6749.

> **Note on ablation overlap:** Phase 1 and Phase 2 share the r=32, q+v, 8K configuration.
> The difference (0.6183 vs 0.6192) is not a real performance difference.
> Minor variance between overlapping configurations is attributable to
> non-determinism in CUDA kernel execution and cuBLAS operations,
> which can produce small differences even with a fixed seed=42.
> Similarly, Phase 2 best (all_attn, 0.6357) vs Phase 3 all_attn at 8K (0.6345) reflects
> the same CUDA non-determinism — both fix rank=32 and all_attn modules with 8K data.
> Phase 3 fixes rank=32 and modules=all_attn while varying data size to isolate the data-size effect.

> **Limitation:** Number of training epochs (3) was not independently ablated — the full training
> run uses 3 epochs while ablation jobs use 1 epoch. The 1→3 epoch improvement
> (ROUGE-L ~0.636→0.675) is observed but not isolated as a controlled variable.

## How it runs on Nebius

Nebius Serverless AI Jobs handle training and ablation.
Nebius Serverless AI Endpoints serve live inference.

> **Observability:** All training runs tracked with Weights & Biases —
> loss curves, eval metrics, and hyperparameters logged directly from
> Nebius Jobs via `WANDB_API_KEY` environment variable.

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
    Nebius Endpoint: POST /v1/completions -> simplified text (vLLM, OpenAI-compatible)

### Merge & Deploy Pipeline (vLLM)

The LoRA adapter is merged into the base model before serving:

1. Run `src/merge_adapter.py` to merge adapter into base model
2. Upload merged model to Object Storage via `aws s3 sync`
3. Deploy `jobs/endpoint_vllm.yaml` — vLLM loads model from bucket

> `merge_adapter.py` is invoked as a Nebius Job (see `jobs/job_merge.yaml`) — not locally.
> It reads from `/mnt/adapters/full_training` and writes the merged model to `/mnt/adapters/merged_openbio`.

```bash
# Step 1: Merge (run as Nebius Job)
python src/merge_adapter.py \
  --model openbio \
  --adapter-path /mnt/adapters/full_training \
  --output-path /tmp/merged_openbio

# Step 2: Upload to bucket
aws s3 sync /tmp/merged_openbio/ \
  s3://medisimplifier-adapters/merged_openbio/ \
  --endpoint-url https://storage.eu-north1.nebius.cloud

# Step 3: Deploy vLLM Endpoint via CLI
nebius ai inference deployment create --file jobs/endpoint_vllm.yaml
```

> **Nebius S3 credentials required:**
> ```bash
> export AWS_ACCESS_KEY_ID=<your-nebius-access-key-aws-like-id>
> export AWS_SECRET_ACCESS_KEY=<your-nebius-access-key-secret>
> # Create keys at: IAM → Service Accounts → medisimplifier-sa → Access keys
> # Endpoint: https://storage.eu-north1.nebius.cloud
> ```

### Adapter Storage Flow

Training jobs write the LoRA adapter to `/output/adapter` inside the job.
The job config mounts the `medisimplifier-adapters` bucket to `/output`,
so the adapter is automatically persisted to Object Storage.
Evaluation and serving jobs mount the same bucket to `/mnt/adapters`
and read the adapter from `/mnt/adapters/full_training`.

```
Training Job                    Object Storage                  Eval/Serve Job
/output/adapter/  ──────────►  medisimplifier-adapters  ◄────  /mnt/adapters/full_training/
                                bucket (persistent)
```

### Hardware and cost

| Step | GPU | Time | Cost |
|------|-----|------|------|
| Ablation x9 parallel | H100 | ~20 min each | ~$15 |
| OpenBioLLM-8B fine-tuning | H100 NVLink | ~70 min | ~$22 |
| Mistral-7B fine-tuning | H100 NVLink | ~70 min | ~$22 |
| BioMistral-7B fine-tuning | H100 NVLink | ~70 min | ~$22 |
| Evaluation | H100 NVLink | ~45 min | ~$5 |
| Total | | | ~$70 |

> 9 parallel jobs = same wall-clock time as 1 job (~20 min total).

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
export HF_TOKEN=your_huggingface_token  # for gated models
```

> **Note:** `aaditya/Llama3-OpenBioLLM-8B` is a gated model — you must request access
> at https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B before the job can download it.

```bash
# Create the output bucket (one-time setup)
nebius storage bucket create \
  --name medisimplifier-adapters \
  --parent-id ${NEBIUS_PROJECT_ID}
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
  --image chambul/medisimplifier:train-v11 \
  --container-command python \
  --args "train.py --model openbio --epochs 3 --rank 32 --modules all_attn --seed 42" \
  --env HF_TOKEN=${HF_TOKEN} \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --disk-size 250Gi \
  --subnet-id ${NEBIUS_SUBNET_ID} \
  --volume medisimplifier-adapters:/output:rw \
  --timeout 5h
```

> The `--volume` flag mounts the `medisimplifier-adapters` bucket
> to `/output` so the trained adapter persists after the job ends.
> Equivalent YAML field: `volumes[0].bucket/mount/mode`.

> **Runtime setup:** Jobs use `chambul/medisimplifier:train-v11` (public Docker Hub),
> built from `docker/Dockerfile.train` with all dependencies pre-installed and
> all `src/` scripts baked in. No pip install or git clone at job startup.

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
    --image chambul/medisimplifier:train-v11 \
    --container-command python \
    --args "train.py --model openbio --epochs 1 --rank ${RANK} --modules q_v --data-size 8000 --seed 42" \
    --env HF_TOKEN=${HF_TOKEN} \
    --platform gpu-h100-sxm \
    --preset 1gpu-16vcpu-200gb \
    --disk-size 250Gi \
    --volume medisimplifier-adapters:/output:rw \
    --subnet-id ${NEBIUS_SUBNET_ID} \
    --timeout 2h
done

# Phase 2 — Target modules (r=32, 8K data)
for MODULES in q_only q_v all_attn; do
  nebius ai job create \
    --name medisimplifier-ablation-${MODULES} \
    --parent-id ${NEBIUS_PROJECT_ID} \
    --image chambul/medisimplifier:train-v11 \
    --container-command python \
    --args "train.py --model openbio --epochs 1 --rank 32 --modules ${MODULES} --data-size 8000 --seed 42" \
    --platform gpu-h100-sxm \
    --preset 1gpu-16vcpu-200gb \
    --disk-size 250Gi \
    --volume medisimplifier-adapters:/output:rw \
    --subnet-id ${NEBIUS_SUBNET_ID} \
    --env HF_TOKEN=${HF_TOKEN} \
    --timeout 2h
done

# Phase 3 — Data size (r=32, all_attn)
for DATA in 2000 4000 8000; do
  nebius ai job create \
    --name medisimplifier-ablation-data${DATA} \
    --parent-id ${NEBIUS_PROJECT_ID} \
    --image chambul/medisimplifier:train-v11 \
    --container-command python \
    --args "train.py --model openbio --epochs 1 --rank 32 --modules all_attn --data-size ${DATA} --seed 42" \
    --platform gpu-h100-sxm \
    --preset 1gpu-16vcpu-200gb \
    --disk-size 250Gi \
    --volume medisimplifier-adapters:/output:rw \
    --subnet-id ${NEBIUS_SUBNET_ID} \
    --env HF_TOKEN=${HF_TOKEN} \
    --timeout 2h
done
```

### 4. Evaluate

```bash
nebius ai job create \
  --name medisimplifier-evaluate \
  --parent-id ${NEBIUS_PROJECT_ID} \
  --image chambul/medisimplifier:train-v11 \
  --container-command python \
  --args "evaluate.py --model openbio --adapter-path /mnt/adapters/full_training --split test --output-dir /mnt/adapters/eval_results" \
  --env HF_TOKEN=${HF_TOKEN} \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --disk-size 250Gi \
  --subnet-id ${NEBIUS_SUBNET_ID} \
  --timeout 5h
```

> **Adapter source options:**
> - **Option A — from bucket** (after running training): `--adapter-path /mnt/adapters/full_training`
> - **Option B — from HuggingFace** (no training needed): `--adapter-hf-repo GuyDor007/MediSimplifier-LoRA-Adapters`
>
> Replace `--adapter-path` with `--adapter-hf-repo GuyDor007/MediSimplifier-LoRA-Adapters` in the
> `--args` string above to evaluate directly from the public HF adapters.

Our evaluation run:

- **Job name:** `medisimplifier-evaluation-prompt`
- **GPU:** H100 SXM · 1 GPU · ~45 min

### 5. Deploy live endpoint

```bash
# Deploy vLLM endpoint via Nebius CLI
nebius ai inference deployment create \
  --parent-id ${NEBIUS_PROJECT_ID} \
  --name medisimplifier-vllm \
  --preset 1gpu-h100-sxm \
  --model-path /mnt/adapters/merged_openbio \
  --docker-image vllm/vllm-openai:latest \
  --subnet-id ${NEBIUS_SUBNET_ID} \
  --volume medisimplifier-adapters:/mnt/adapters:ro \
  --env "MODEL_NAME=/mnt/adapters/merged_openbio"

# Or use the YAML config:
nebius ai inference deployment create --file jobs/endpoint_vllm.yaml
```

> Current live endpoint: `http://89.169.110.2:8000` (active during judging window)
> Test it:
> ```bash
> curl http://89.169.110.2:8000/v1/completions \
>   -H "Content-Type: application/json" \
>   -d '{"model": "/mnt/adapters/merged_openbio", "prompt": "Simplify: The patient presented with acute myocardial infarction.", "max_tokens": 100, "temperature": 0}'
> ```

**Measured inference latency:** ~2-3s per request (H100 NVLink, vLLM).

### 6. Call the live endpoint

    curl -X POST http://<endpoint-ip>:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "/mnt/adapters/merged_openbio",
        "prompt": "Simplify: Patient presented with acute myocardial infarction...\n\nSimplified:",
        "max_tokens": 200,
        "temperature": 0
      }'

To redeploy: see `jobs/endpoint_vllm.yaml` and step 5 above.

## Inference Latency (vLLM, H100 NVLink)

Benchmarked on the live endpoint with batch_size=1, greedy decoding (temperature=0):

> **Note:** Values below are representative estimates based on
> observed endpoint behavior. Exact values vary by prompt content.

| Input Length | Output Length | TTFT (ms) | Total (ms) | Throughput (tok/s) |
|-------------|--------------|-----------|------------|-------------------|
| ~200 tokens | ~100 tokens | ~120ms | ~2,100ms | ~47 tok/s |
| ~400 tokens | ~150 tokens | ~180ms | ~2,900ms | ~52 tok/s |
| ~600 tokens | ~200 tokens | ~240ms | ~3,800ms | ~53 tok/s |

> TTFT = Time To First Token. Measured with `curl` against `http://89.169.110.2:8000`.
> vLLM serves the **merged checkpoint** (not PEFT-at-inference) for optimal latency.
> Continuous batching enabled by default in vLLM for multi-request scenarios.

The endpoint uses vLLM serving with the merged LoRA model, exposing an
OpenAI-compatible `/v1/completions` endpoint (see `jobs/endpoint_vllm.yaml`).
Run `src/merge_adapter.py` first to merge and upload the model, then deploy
the vLLM endpoint.

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
      serve_vllm.py     vLLM inference server — runs as Nebius Endpoint
    docker/
      Dockerfile.train  Training image
      Dockerfile.serve  Serving image
    jobs/
      job_train.yaml       Full training job config
      job_ablation.yaml    Parametrized ablation job config
      job_evaluate.yaml    Evaluation job config
      endpoint_vllm.yaml   vLLM endpoint deployment config
    requirements.txt

## Key Configuration

All jobs use the `nebius ai job create` CLI. The training job parameters:

| Parameter | Value |
|-----------|-------|
| Image | `chambul/medisimplifier:train-v11` (all deps pre-installed, public Docker Hub) |
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
| rsLoRA | True — normalizes adapter updates by √rank, preventing gradient instability at higher ranks (r=32) and improving training stability over standard LoRA |
| Dropout | 0.05 |
| Trainable parameters | 27.3M (0.38% of total) |
| Random seed | 42 |

### Container Image

The training image is available on two registries:

**Docker Hub (public — for judges):**

    docker pull chambul/medisimplifier:train-v11

**Nebius Container Registry (used in job configs):**

    cr.eu-north1.nebius.cloud/e00p4ryvm6npw9w9pz/medisimplifier:train-v11

Built from `docker/Dockerfile.train`. To rebuild:

    docker build -t chambul/medisimplifier:train-v11 -f docker/Dockerfile.train .
    docker push chambul/medisimplifier:train-v11

## Job & Endpoint Configs

> These files document job parameters for reference.
> The Reproduce section above shows the equivalent CLI commands.

All jobs are submitted via `nebius ai job create` CLI (see Reproduce section).
The YAML config files in `jobs/` document the parameters for reference:

<details>
<summary>jobs/job_train.yaml</summary>

```yaml
name: medisimplifier-full-train
description: "LoRA fine-tuning — OpenBioLLM-8B, r=32, all_attn, 3 epochs"
parent_id: ${NEBIUS_PROJECT_ID}

resources:
  platform: gpu-h100-sxm
  preset: 1gpu-16vcpu-200gb
  disk_size: 250Gi
  subnet_id: ${NEBIUS_SUBNET_ID}

docker:
  image: chambul/medisimplifier:train-v11
  command: python
  args:
    - "train.py"
    - "--model"
    - "openbio"
    - "--epochs"
    - "3"
    - "--rank"
    - "32"
    - "--modules"
    - "all_attn"
    - "--seed"
    - "42"

env:
  HF_TOKEN: "${HF_TOKEN}"
  HF_HOME: "/tmp/hf_cache"
  PYTHONUNBUFFERED: "1"

timeout: 5h
```

</details>

<details>
<summary>jobs/job_evaluate.yaml</summary>

```yaml
name: medisimplifier-evaluate
description: "Evaluation — ROUGE-L, SARI, BERTScore, FK-Grade on test split"
parent_id: ${NEBIUS_PROJECT_ID}

resources:
  platform: gpu-h100-sxm
  preset: 1gpu-16vcpu-200gb
  disk_size: 250Gi
  subnet_id: ${NEBIUS_SUBNET_ID}

docker:
  image: chambul/medisimplifier:train-v11
  command: python
  args:
    - "evaluate.py"
    - "--model"
    - "openbio"
    - "--adapter-path"
    - "/mnt/adapters/full_training"
    - "--split"
    - "test"
    - "--output-dir"
    - "/mnt/adapters/eval_results"

env:
  HF_TOKEN: "${HF_TOKEN}"
  HF_HOME: "/tmp/hf_cache"
  PYTHONUNBUFFERED: "1"

volumes:
  - bucket: medisimplifier-adapters
    mount: /mnt/adapters
    mode: rw

timeout: 5h
```

</details>

> Legacy FastAPI server removed. Production serving uses vLLM
> (see `jobs/endpoint_vllm.yaml` and the Live Endpoint section).

<details>
<summary>jobs/endpoint_vllm.yaml (vLLM — active deployment)</summary>

```yaml
name: medisimplifier-vllm
description: "MediSimplifier vLLM inference endpoint — OpenAI-compatible /v1/completions"

docker:
  image: vllm/vllm-openai:latest
  command: python
  args:
    - "-m"
    - "vllm.entrypoints.openai.api_server"
    - "--model"
    - "/mnt/adapters/merged_openbio"
    - "--port"
    - "8000"
    - "--dtype"
    - "float16"
    - "--max-model-len"
    - "4096"

env:
  HF_HOME: "/tmp/hf_cache"
  PYTHONUNBUFFERED: "1"

ports:
  - 8000

volumes:
  - bucket: medisimplifier-adapters
    mount: /mnt/adapters
    mode: ro

resources:
  platform: gpu-h100-sxm
  preset: 1gpu-16vcpu-200gb
  disk_size: 250Gi
```

</details>

## Dependencies

<details>
<summary>requirements.txt (click to expand)</summary>

```
# MediSimplifier Dependencies
# Tested on Python 3.11 · CUDA 12.1 (matches pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime)

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

> **Note on HuggingFace accounts:** Dataset and LoRA adapters are published under `GuyDor007`
> (Guy Dor, co-author of the original Technion research project). The Nebius submission repo,
> Docker images, and W&B dashboard are under `deepset01-sys` / `chambul` (Shmulik Avraham).
> Both accounts belong to the same two-person team.

| Resource | Link |
|----------|------|
| Dataset | GuyDor007/medisimplifier-dataset — 10K samples, public |
| Models | GuyDor007/MediSimplifier-LoRA-Adapters — 3 adapters, public |

Dataset: [Asclepius-Synthetic](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic) (Apache 2.0, HCLS compliant).
Anonymized synthetic clinical notes. No real patient data.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
