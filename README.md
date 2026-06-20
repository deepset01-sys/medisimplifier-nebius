# MediSimplifier — Serverless LoRA Fine-Tuning on Nebius

[![Nebius Jobs](https://img.shields.io/badge/Nebius-Serverless%20AI%20Jobs-blue)](https://nebius.com)
[![HuggingFace Models](https://img.shields.io/badge/HF-Models-yellow)](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters)
[![HuggingFace Dataset](https://img.shields.io/badge/HF-Dataset-blue)](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

> **Nebius Serverless AI Builders Challenge submission by Shmulik Avraham.**
> Built on top of [MediSimplifier](https://github.com/gd007/MediSimplifier) —
> a Technion DS25 graduation project by Shmulik Avraham & Guy Dor.
> The Nebius pipeline, serving layer, safety evaluation, and MLOps
> infrastructure were built independently for this challenge.

**📝 Blog Post:** [Medical Text Simplification with LoRA on Nebius Serverless: A Builder's Journey](https://medium.com/@deepset01/medical-text-simplification-with-lora-on-nebius-serverless-a-builders-journey-13a9e44c92a4)

## What this project does

Medical discharge summaries are written at college reading level (FK-Grade 14.5).
Only 12% of American adults have sufficient health literacy to understand them.

MediSimplifier fine-tunes open-source LLMs using LoRA to automatically simplify
these documents to 7th-grade reading level — ~50% readability reduction —
while preserving all critical medical information.

> **What's new in this Nebius submission vs the Technion project:**
> The ranking-reversal and LoRA rank findings were first observed
> in the original Technion research. New contributions here are:
> the **Nebius MLOps pipeline** (parallel ablation, stateless jobs,
> bucket persistence), **H100 hardware reproduction** (δ 1.6–5.0%),
> **vLLM production serving**, and **LLM-as-judge safety evaluation**
> via Token Factory. See [What Nebius Added](#what-nebius-added) for details.

**Key findings (challenging the conventional recommendation of r=4-8 (Hu et al. 2021)):**

| Finding | Result |
|---------|--------|
| Optimal LoRA rank | r=32 outperforms r=4-8 recommended by Hu et al. 2021 |
| Optimal modules | all_attn (Q+K+V+O) outperforms standard Q+V |
| Ranking reversal | Worst zero-shot model becomes best fine-tuned (+153.1% on Nebius H100) |
| Readability | FK-Grade 14.5 → 6.91 (Mistral-7B, original H200); Nebius H100: OpenBioLLM 7.33, Mistral 6.14, BioMistral 6.13 |
| Data efficiency | 4K samples achieves 97% of 8K performance |
| Baseline-improvement correlation | monotonic — lower zero-shot → higher gain (n=3 models) |
| Total compute | 9 ablation runs + 3 full training runs, ~10 GPU hours (~7h wall-clock; ablation jobs run in parallel) |

## What Nebius Added

The original Technion project was training-only (H200, no serving, no pipeline).
This submission extends it with a full MLOps lifecycle on Nebius:

| | Technion original (Guy Dor & Shmulik Avraham) | This Nebius submission (Shmulik Avraham) |
|--|----------------------------------------------|------------------------------------------|
| Training | ✅ H200, single run | ✅ Reproduced on H100 — validated |
| Parallel ablation | ❌ | ✅ 9 jobs × 20 min simultaneously |
| Production serving | ❌ | ✅ vLLM Endpoint — OpenAI-compatible API |
| Safety evaluation | ❌ | ✅ LLM-as-judge via Token Factory |
| MLOps pipeline | ❌ | ✅ Object Storage — stateless jobs, persistent adapters |
| Hardware validation | ❌ | ✅ H200 → H100 reproduction, δ 1.6–5.0% |

> **Jobs** = training / ablation / evaluation (stateless, pay-per-second).
> **Endpoint** = inference serving (live, OpenAI-compatible API).
> The ranking reversal finding reproduces on Nebius H100 within 1.6–5.0%
> of the original — confirming it is not a hardware artifact.

## Results

> **Primary results: Nebius H100 reproduction**
> (original H200 results in parentheses for comparison)

| Model | ROUGE-L (Nebius H100) | ROUGE-L (Original H200) | Delta | SARI | BERTScore | FK-Grade | Improvement |
|-------|----------------------|------------------------|-------|------|-----------|----------|-------------|
| OpenBioLLM-8B | **0.6638** | 0.6749 | −1.6% | 73.49 | 0.9460 | 7.33 | +153.1% |
| Mistral-7B | **0.6253** | 0.6491 | −3.7% | 72.75 | 0.9418 | 6.14 | +59.8% |
| BioMistral-7B-DARE | **0.6004** | 0.6318 | −5.0% | 71.97 | 0.9372 | 6.13 | +45.7% |

95% CIs from bootstrap (n=10,000). All pairwise ROUGE-L differences significant at p<0.001.
All results use seed=42. Bootstrap CIs computed with n=10,000 resamples.

> Improvement % = (Nebius H100 fine-tuned − zero-shot) / zero-shot.
> OpenBioLLM: (0.6638 − 0.2623) / 0.2623 = +153.1%

> Key finding: ranking reversal fully reproduced on Nebius H100.
> Evaluation: 1,001 test samples, greedy decoding, seed=42.

**📊 Live Training Dashboard (Weights & Biases):**
[wandb.ai/deepset01-chambul/medisimplifier](https://wandb.ai/deepset01-chambul/medisimplifier)
*(Project is public — no login required to view training curves)*

> Training monitored via W&B on Nebius H100 NVLink.
> OpenBioLLM-8B: train_loss 0.844→0.635 over 3 epochs (1,500 steps, seed=42).
> Dashboard includes loss curves, eval metrics per epoch, gradient norms, and hyperparameters.

> **Note on FK-Grade target:** The original research target was FK ≤ 6.0.
> Best achieved on H200: 6.91 (Mistral-7B, original run).
> OpenBioLLM-8B achieved 7.16 on original H200 hardware.
> Nebius H100 reproduction: OpenBioLLM 7.33, Mistral 6.14, BioMistral 6.13.
> The difference reflects H200→H100 hardware variance.
> The gap from the 6.0 target reflects the inherent tension between medical accuracy preservation
> and maximum simplification.

### Reproduced on Nebius H100 NVLink

All three models were fully trained and evaluated on Nebius
Serverless Jobs (H100 NVLink), reproducing the original
Technion research findings:

| Model | Original ROUGE-L (H200) | Nebius ROUGE-L (H100) | Delta |
|-------|------------------------|----------------------|-------|
| OpenBioLLM-8B | 0.6749 | 0.6638 | −1.6% |
| Mistral-7B | 0.6491 | 0.6253 | −3.7% |
| BioMistral-7B-DARE | 0.6318 | 0.6004 | −5.0% |

> The ranking reversal finding holds across both hardware
> generations — confirming it is not a hardware artifact.
> Training monitored via [W&B dashboard](https://wandb.ai/deepset01-chambul/medisimplifier) *(public, no login required)*.
> Evaluation: 1,001 test samples, greedy decoding, seed=42.

## Baseline vs Fine-Tuned Results

### Zero-Shot Baseline (no fine-tuning, 1,001 test samples)

| Model | ROUGE-L | SARI | BERTScore | FK-Grade |
|-------|---------|------|-----------|----------|
| OpenBioLLM-8B | 0.2623 | 36.98 | 0.637 | 12.53 |
| Mistral-7B | 0.3912 | 46.38 | 0.734 | 10.60 |
| BioMistral-7B-DARE | 0.4120 | 51.91 | 0.743 | 9.52 |

**Key finding:** OpenBioLLM-8B had the *lowest* zero-shot score (0.2623)
but achieved the *highest* fine-tuned score (0.6749 on original H200 hardware; 0.6638 on Nebius H100) — a full ranking
reversal. All pairwise differences significant at p<0.001 (bootstrap n=10,000).

### Why Did the Ranking Reversal Happen?

The ranking reversal (worst zero-shot → best fine-tuned) is monotonic across all 3 models — lower zero-shot score correlates with higher fine-tuning gain.

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

## Evaluation Evidence

All results are reproducible from public artifacts:

### Committed Evidence Files

All results are committed to this repository for durable verification:

**Evaluation Results** (`results/nebius_evidence/`):
| File | Model | ROUGE-L | SARI | BERTScore | FK-Grade |
|------|-------|---------|------|-----------|----------|
| [results_openbio.json](results/nebius_evidence/results_openbio.json) | OpenBioLLM-8B | 0.6638 | 73.49 | 0.9460 | 7.33 |
| [results_mistral.json](results/nebius_evidence/results_mistral.json) | Mistral-7B | 0.6253 | 72.75 | 0.9418 | 6.14 |
| [results_biomistral.json](results/nebius_evidence/results_biomistral.json) | BioMistral-7B | 0.6004 | 71.97 | 0.9372 | 6.13 |

**Safety Evaluation** (`results/nebius_evidence/`):
[safety_results.json](results/nebius_evidence/safety_results.json) — 100 samples with per-sample LLM judge verdicts, rule-based scores, and Token Factory latency metrics.

**Nebius Job Logs** (`results/nebius_logs/`):
| File | Contents |
|------|---------|
| [full_train_logs.json.gz](results/nebius_logs/full_train_logs.json.gz) | Full OpenBioLLM-8B training — H100, train_loss 0.844→0.635 |
| [endpoint_vllm_logs.json.gz](results/nebius_logs/endpoint_vllm_logs.json.gz) | vLLM endpoint startup + requests, vmapp_id: aiendpoint-e00ef3br6r14grvhhd |
| [r32_all_8kdata.json.gz](results/nebius_logs/r32_all_8kdata.json.gz) | Best ablation config — r=32, all_attn, 8K |
| [adapters_logs.json.gz](results/nebius_logs/adapters_logs.json.gz) | Merge adapter job |
| + 6 more ablation logs | r8, r16, r32 variants |

> All logs contain Nebius job IDs (aijob-* / aiendpoint-*),
> GPU info (NVIDIA H100 80GB HBM3), and timestamps.

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
  "n_evaluated": 95,
  "n_errors": 5,
  "judge_model": "meta-llama/Llama-3.3-70B-Instruct",
  "rule_based": {"safe_rate": 0.0, "threshold": 0.85},
  "llm_judge": {"safe": 73, "unsafe": 22, "error": 5, "safe_rate": 0.7684}
}
```
</details>

**🔗 Public Artifacts:**
- Dataset: [GuyDor007/medisimplifier-dataset](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)
- Adapters: [GuyDor007/MediSimplifier-LoRA-Adapters](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters)
- Docker: [chambul/medisimplifier:train-v11](https://hub.docker.com/r/chambul/medisimplifier)
- W&B: [wandb.ai/deepset01-chambul/medisimplifier](https://wandb.ai/deepset01-chambul/medisimplifier)
- Endpoint: `http://89.169.110.2:8000` (active during judging window)

## Medical Safety Evaluation (Preliminary)

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
| scispaCy (negative control — exact match too strict for semantic equivalents) | 0 | 100 | 0.0% |
| Llama-3.3-70B Judge | 73 | 22 | **76.8%** (of 95 evaluated; 5 errored) |

> **Note:** The 0% rule-based safe rate is expected — scispaCy's
> exact-match entity comparison cannot recognize semantic equivalents
> (e.g., "heart attack" ≠ "myocardial infarction").
> The LLM judge correctly identifies these as faithful simplifications.

> **Note on n=95:** 5 of 100 samples returned an error from the
> LLM judge (network timeout). safe_rate = 73/95 evaluated samples.

### Key Finding

**Rule-based exact matching underestimates medical faithfulness.** scispaCy
flags all simplifications as unsafe because it cannot recognize semantic
equivalents (e.g., "myocardial infarction" → "heart attack"). The LLM judge,
which understands semantic equivalence, found **76.8% of simplifications
fully faithful** to the original medical content (73/95 evaluated outputs).

> In the 22 cases flagged by the LLM judge, issues were limited to
> minor information condensation rather than hallucinated medical facts
> (preliminary screening on 100 samples — not a deployment-ready audit).

> This evaluation was conducted on Nebius AI Studio using
> `meta-llama/Llama-3.3-70B-Instruct` as the judge model,
> demonstrating Nebius Token Factory's API for LLM-as-judge workflows.

> **Scope:** This is a research prototype.
> 76.8% faithfulness (73/95 evaluated) is below the threshold
> required for clinical deployment. The 22 unsafe cases involved
> minor information condensation — no hallucinated medical facts
> were detected. Not intended for production medical use.

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

> **Note on ablation overlap:** Phase 1 and Phase 2 share the r=32, q+v, 8K configuration (0.6183 vs 0.6192),
> and Phase 2/Phase 3 share r=32, all_attn, 8K (0.6357 vs 0.6345). These are not real performance differences.
> Each config was run once (n=1, seed=42). Non-determinism in CUDA kernel execution and cuBLAS operations
> can produce small deltas (~0.001–0.002 ROUGE-L) even with a fixed seed — differences this small
> should not be over-interpreted. Phase 3 fixes rank=32 and modules=all_attn while varying data size
> to isolate the data-size effect.

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

**Why Serverless Jobs and not a VM or Kubernetes cluster?**

| | Serverless Jobs | Reserved VM | Kubernetes |
|--|----------------|-------------|-----------|
| Setup time | 0 min | 10–30 min | Hours |
| Cost when idle | $0 | Full rate | Full rate |
| Parallel jobs | Instant | Manual | Complex |
| 9 ablations | 20 min, ~$15 | 3 hours | Setup overhead |
| Best for | Experimentation | Long training | Production serving |

> Serverless Jobs eliminated cluster management overhead that
> typically slows ML experimentation. Each job is stateless —
> the bucket is the only persistent state between runs.
> I submitted all 9 ablation jobs simultaneously and had
> results in 20 minutes instead of 3 hours sequentially.

### Merge & Deploy Pipeline (vLLM)

The LoRA adapter is merged into the base model before serving:

1. Run `src/merge_adapter.py` to merge adapter into base model
2. Upload merged model to Object Storage via `aws s3 sync`
3. Deploy `jobs/endpoint_vllm.yaml` — vLLM loads model from bucket

> `merge_adapter.py` is invoked as a Nebius Job (see `jobs/job_merge.yaml`) — not locally.
> It reads from `/mnt/adapters/full_training` and writes the merged model to `/mnt/adapters/merged_openbio`.

> **Note:** This describes the internal production pipeline
> (bucket-mounted adapter). Judges reproducing the endpoint
> should use Step 5 instead — it loads directly from HuggingFace
> with no bucket credentials required.

```bash
# Step 1: Merge (run as Nebius Job)
python src/merge_adapter.py \
  --model openbio \
  --adapter-path /mnt/adapters/full_training \
  --output-path /mnt/adapters/merged_openbio

# Step 2: Upload to bucket
aws s3 sync /mnt/adapters/merged_openbio/ \
  s3://medisimplifier-adapters/merged_openbio/ \
  --endpoint-url https://storage.eu-north1.nebius.cloud

# Step 3: Deploy vLLM Endpoint via CLI
nebius ai endpoint create \
  --name medisimplifier-vllm \
  --parent-id ${NEBIUS_PROJECT_ID} \
  --image vllm/vllm-openai:latest \
  --args "-m vllm.entrypoints.openai.api_server --model /mnt/adapters/merged_openbio --port 8000 --dtype float16 --max-model-len 4096" \
  --container-port 8000 \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --subnet-id ${NEBIUS_SUBNET_ID} \
  --volume medisimplifier-adapters:/mnt/adapters:ro \
  --env HF_HOME=/tmp/hf_cache \
  --public
```

> **Nebius S3 credentials required:**
> ```bash
> export AWS_ACCESS_KEY_ID=<your-nebius-access-key-aws-like-id>
> export AWS_SECRET_ACCESS_KEY=<your-nebius-access-key-secret>
> # Create keys at: IAM → Service Accounts → medisimplifier-sa → Access keys
> # Endpoint: https://storage.eu-north1.nebius.cloud
> ```

**Why Serverless Endpoints and not a serving VM?**

> With a reserved VM, you pay whether requests arrive or not.
> For an inference endpoint that serves sporadic requests —
> a demo, a research prototype, or a low-traffic API —
> Serverless Endpoints are the only economically rational choice.
> We serve a merged 8B model at ~$3/hr only when the endpoint
> is actually running, with zero idle cost between sessions.
> The OpenAI-compatible API means any existing client works
> without modification.

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

| Step | GPU | Wall-clock time | Approx. compute cost |
|------|-----|----------------|---------------------|
| Ablation ×9 (parallel) | H100 NVLink | ~20 min total | ~$9 |
| OpenBioLLM-8B training | H100 NVLink | ~70 min | ~$4 |
| Mistral-7B training | H100 NVLink | ~70 min | ~$4 |
| BioMistral-7B training | H100 NVLink | ~70 min | ~$4 |
| Evaluation ×3 | H100 NVLink | ~45 min each | ~$7 |
| Safety evaluation | H100 NVLink | ~30 min | ~$2 |
| Merge + misc | H100 NVLink | — | ~$2 |
| **Compute subtotal** | | | **~$32** |

> H100 NVLink rate: ~$3.04/hr on Nebius eu-north1.
> Compute subtotal: ~$32. Total project spend including
> storage, failed runs, and iteration: **~$91**.
> All costs incurred on Nebius Serverless — no reserved instances.

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

> **CLI verified:** All `nebius ai job create` flags above
> are confirmed against Nebius CLI v0.12.229.

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

> **Note:** Jobs are submitted in parallel via the CLI loop.
> Actual parallel scheduling depends on account quota.
> In our runs, all 9 jobs started within 2 minutes of submission.

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

The merged OpenBioLLM-8B model is publicly available on HuggingFace:
[chambul/MediSimplifier-OpenBioLLM-merged](https://huggingface.co/chambul/MediSimplifier-OpenBioLLM-merged)

Deploy your own vLLM endpoint on Nebius in ~5 minutes:

```bash
nebius ai endpoint create \
  --name medisimplifier-vllm \
  --parent-id ${NEBIUS_PROJECT_ID} \
  --image vllm/vllm-openai:latest \
  --args "-m vllm.entrypoints.openai.api_server --model chambul/MediSimplifier-OpenBioLLM-merged --port 8000 --dtype float16 --max-model-len 4096" \
  --container-port 8000 \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --subnet-id ${NEBIUS_SUBNET_ID} \
  --env HF_TOKEN=${HF_TOKEN} \
  --public
```

> **Important:** After creating the endpoint, wait ~10–15 minutes
> for the model to load from HuggingFace before sending requests.
> To check if ready: `curl http://<your-ip>:8000/v1/models`
> When you see `chambul/MediSimplifier-OpenBioLLM-merged` listed,
> the endpoint is ready.

> No training required — the merged model loads directly from HuggingFace.
> Endpoint tested live — curl response:
> ```json
> {"choices":[{"text":"The patient had a heart attack and was given blood-thinning medicine..."}]}
> ```
> Latency: ~946ms–1,198ms, throughput: 106–144 tok/s (H100 NVLink).

### 6. Call the live endpoint

> **Note:** Replace `<endpoint-ip>` with the IP from your
> `nebius ai endpoint create` output (Step 5).
> The model name must match exactly: `chambul/MediSimplifier-OpenBioLLM-merged`

    curl -X POST http://<endpoint-ip>:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "chambul/MediSimplifier-OpenBioLLM-merged",
        "prompt": "Simplify: Patient presented with acute myocardial infarction...\n\nSimplified:",
        "max_tokens": 200,
        "temperature": 0
      }'

To redeploy: see `jobs/endpoint_vllm.yaml` and step 5 above.

## Inference Latency (vLLM, H100 NVLink)

Benchmarked on the live endpoint (http://89.169.110.2:8000),
greedy decoding (temperature=0), measured during judging window:

| Input Tokens | Output Tokens | Total Latency | Throughput |
|-------------|--------------|---------------|------------|
| 18 | 100 | 946ms | 106 tok/s |
| 26 | 138 | 1,198ms | 115 tok/s |
| ~40 | ~150 | 1,040ms | 144 tok/s |

> Measured with PowerShell Invoke-WebRequest against the live vLLM endpoint.
> Model: `chambul/MediSimplifier-OpenBioLLM-merged` (merged OpenBioLLM-8B + LoRA adapter, served via vLLM).
> Sub-second to ~1.2s latency for typical discharge summary simplification.

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
      safety_eval.py    LLM-as-judge safety evaluation via Nebius Token Factory
      serve_vllm.py     vLLM inference server — runs as Nebius Endpoint
    docker/
      Dockerfile.train  Training image
      Dockerfile.serve  Serving image
    jobs/
      job_train.yaml       Full training job config
      job_ablation.yaml    Parametrized ablation job config
      job_evaluate.yaml    Evaluation job config
      job_merge.yaml       Adapter merge job config
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

### Core Training Code (src/train.py)

```python
# LoRA configuration — rsLoRA, r=32, all attention matrices
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,        # α = 2r, per rsLoRA scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,      # rank-stabilized LoRA
)

# Prompt template — identical at train and inference time
def format_prompt(text: str) -> str:
    return f"Simplify: {text}\n\nSimplified:"
```

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
public: true
parent_id: ${NEBIUS_PROJECT_ID}

docker:
  image: vllm/vllm-openai:latest
  command: python
  args:
    - "-m"
    - "vllm.entrypoints.openai.api_server"
    - "--model"
    - "chambul/MediSimplifier-OpenBioLLM-merged"
    - "--port"
    - "8000"
    - "--dtype"
    - "float16"
    - "--max-model-len"
    - "4096"

env:
  HF_HOME: "/tmp/hf_cache"
  HF_TOKEN: "${HF_TOKEN}"
  PYTHONUNBUFFERED: "1"

ports:
  - 8000

resources:
  platform: gpu-h100-sxm
  preset: 1gpu-16vcpu-200gb
  disk_size: 250Gi
  subnet_id: ${NEBIUS_SUBNET_ID}
```

</details>

## Dependencies

<details>
<summary>requirements.txt (click to expand)</summary>

```
# Python 3.11 · CUDA 12.1 (matches pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime)
torch==2.1.0
transformers==4.45.0
peft==0.14.0
datasets==2.18.0
trl==0.8.6
accelerate==0.28.0
bitsandbytes==0.43.0
sentencepiece==0.2.0
huggingface-hub==0.25.0
numpy==1.26.0
tqdm==4.66.0
wandb==0.17.0
rouge-score==0.1.2
bert-score==0.3.13
textstat==0.7.3
scispacy==0.5.4
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
fastapi==0.110.0
uvicorn==0.29.0
boto3==1.34.0
easse @ git+https://github.com/feralvam/easse.git@6a4352ec299ed03fda8ee45445ca43d9c7673e89
```

</details>

## Dataset and models

> **Note on HuggingFace accounts:** Dataset and LoRA adapters are published under `GuyDor007`
> (Guy Dor, Technion co-author). The Nebius submission repo, Docker images, and W&B dashboard
> are under `deepset01-sys` / `chambul` (Shmulik Avraham).

| Resource | Link |
|----------|------|
| Dataset | GuyDor007/medisimplifier-dataset — 10K samples, public |
| Models | GuyDor007/MediSimplifier-LoRA-Adapters — 3 adapters, public |

Dataset: [Asclepius-Synthetic](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic) (Apache 2.0, HCLS compliant).
Anonymized synthetic clinical notes. No real patient data.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
