# MediSimplifier — End-to-End Serverless ML Pipeline on Nebius

[![Nebius Jobs](https://img.shields.io/badge/Nebius-Serverless%20AI%20Jobs-blue)](https://nebius.com)
[![HuggingFace Models](https://img.shields.io/badge/HF-Models-yellow)](https://huggingface.co/chambul/MediSimplifier-LoRA-Adapter-Nebius)
[![HuggingFace Dataset](https://img.shields.io/badge/HF-Dataset-blue)](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

> **Nebius Serverless AI Builders Challenge submission by Shmulik Avraham.**
> Built on top of [MediSimplifier](https://github.com/gd007/MediSimplifier) —
> a Technion DS25 graduation project by Shmulik Avraham & Guy Dor.
> The Nebius pipeline, serving layer, safety evaluation, and MLOps
> infrastructure were built independently for this challenge.

**📝 Blog Post:** [Building an End-to-End Serverless ML Pipeline on Nebius: A Builder's Journey](https://medium.com/@deepset01/medical-text-simplification-with-lora-on-nebius-serverless-a-builders-journey-13a9e44c92a4)

## What this project does

Medical discharge summaries are written at college reading level (FK-Grade 14.5).
Only 12% of American adults have sufficient health literacy to understand them.

MediSimplifier is an end-to-end serverless ML pipeline on Nebius — covering fine-tuning, parallel ablation, production serving, safety evaluation, and MLOps — applied to the task of simplifying medical discharge summaries from college reading level (FK-Grade 14.5) to 7th-grade level (~7 grade levels) while preserving all critical medical information.

> **What's new in this Nebius submission vs the Technion project:**
> The ranking-reversal and LoRA rank findings were first observed
> in the Technion project. New contributions here are:
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
| Readability | FK-Grade 14.5 → 6.91 (Mistral-7B, Technion H200); Nebius H100: OpenBioLLM 7.33, Mistral 6.14, BioMistral 6.13 |
| Data efficiency | 4K samples achieves 97% of 8K performance |
| Baseline-improvement correlation | Pearson r = −0.998 across 3 models (descriptive statistic, not a significance claim) |
| Total compute | 9 ablation runs + 3 full training runs, ~10 GPU hours (~7h wall-clock; ablation jobs run in parallel) |

## What Nebius Added

The Technion project was training-only (H200, no serving, no pipeline).
This submission extends it with a full MLOps lifecycle on Nebius:

| | Technion project (Guy Dor & Shmulik Avraham) | This Nebius submission (Shmulik Avraham) |
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
> (Technion H200 results in parentheses for comparison)

| Model | ROUGE-L (Nebius H100) | ROUGE-L (Technion H200) | Delta | SARI | BERTScore | FK-Grade | Improvement |
|-------|----------------------|------------------------|-------|------|-----------|----------|-------------|
| OpenBioLLM-8B | **0.6638** | 0.6749 | −1.6% | 73.49 | 0.9460 | 7.33 | +153.1% |
| Mistral-7B | **0.6253** | 0.6491 | −3.7% | 72.75 | 0.9418 | 6.14 | +59.8% |
| BioMistral-7B-DARE | **0.6004** | 0.6318 | −5.0% | 71.97 | 0.9372 | 6.13 | +45.7% |

All results use seed=42. Multi-seed validation (seeds 42 and 2) confirms ROUGE-L variance of 0.0013 (0.6638 vs 0.6651) — within the expected ~0.001–0.002 CUDA non-determinism range. See [eval_seed2.json](results/nebius_evidence/eval_seed2.json).

> **Note on OOM fallback:** `evaluate.py` includes a per-sample OOM retry path added as a robustness measure. Log inspection across all 17 committed Nebius job logs (including all eval and safety eval runs) confirms zero OOM events — the fallback was never triggered, and all committed ROUGE-L scores reflect the standard batch path.

> **Note on data size:** The training split contains 7,999 samples (not 8,000). All references to "8K" in this README mean 7,999 — the full training set. `--data-size 7999` is the correct CLI value. In the Phase 3 data-size ablation loop, `--data-size 8000` exceeds the dataset size — the `if args.data_size < len(dataset)` guard in `train.py` leaves the full 7,999-sample dataset unchanged.

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

**📊 Evaluation Tracking (Nebius Managed MLflow):**
All evaluation results logged to Nebius Managed MLflow during development — 4 runs across 3 models and 2 seeds, with full params, metrics, and git commit traceability.
Export: [mlflow_runs.csv](results/nebius_evidence/mlflow_runs.csv) | Logging script: [send_to_mlflow.py](send_to_mlflow.py)

To restore the live experiment: create a Nebius Managed MLflow cluster, set `MLFLOW_TRACKING_URI`, and run `python send_to_mlflow.py` — all 4 runs restore in ~5 minutes.

> Training observability (W&B) + Evaluation observability (Nebius MLflow) = two-layer MLOps visibility, both on Nebius infrastructure.

> **Note on adapter provenance:** The merged model (`chambul/MediSimplifier-OpenBioLLM-merged`) was created by merging the LoRA adapter trained on Nebius H100 NVLink (`medisimplifier-adapters/adapter/`, job: `medisimplifier-full-training`, r=32, all_attn, 3 epochs, best_epoch=2) with the base model. This adapter produced ROUGE-L 0.6638 as documented in [`results_openbio.json`](results/nebius_evidence/results_openbio.json) and [`full_training_results.json`](results/nebius_evidence/full_training_results.json). The adapter lives in the Nebius bucket and is also publicly available at [`chambul/MediSimplifier-LoRA-Adapter-Nebius`](https://huggingface.co/chambul/MediSimplifier-LoRA-Adapter-Nebius) — the `GuyDor007` repo contains the Technion-era adapters.

> **Note on FK-Grade target:** The Technion project target was FK ≤ 6.0.
> Best achieved on H200: 6.91 (Mistral-7B, Technion run).
> OpenBioLLM-8B achieved 7.16 on Technion H200 hardware.
> Nebius H100 reproduction: OpenBioLLM 7.33, Mistral 6.14, BioMistral 6.13.
> The difference reflects H200→H100 hardware variance.
> The gap from the 6.0 target reflects the inherent tension between medical accuracy preservation
> and maximum simplification.

> The ranking reversal holds across both hardware generations —
> see Delta column above (H200→H100 variance: 1.6–5.0%).

## Baseline vs Fine-Tuned Results

### Zero-Shot Baseline (no fine-tuning, 1,001 test samples)

| Model | ROUGE-L | SARI | BERTScore | FK-Grade |
|-------|---------|------|-----------|----------|
| OpenBioLLM-8B | 0.2623 | 36.98 | 0.637 | 12.53 |
| Mistral-7B | 0.3912 | 46.38 | 0.734 | 10.60 |
| BioMistral-7B-DARE | 0.4120 | 51.91 | 0.743 | 9.52 |

**Key finding:** OpenBioLLM-8B had the *lowest* zero-shot score (0.2623)
but achieved the *highest* fine-tuned score (0.6749 on Technion H200 hardware; 0.6638 on Nebius H100) — a full ranking
reversal. All pairwise differences statistically significant: 95% CIs do not overlap across all 3 models (bootstrap n=10,000, n=1,001 test samples each).

### Why Did the Ranking Reversal Happen?

The ranking reversal (worst zero-shot → best fine-tuned) is consistent across all 3 models: Pearson r = −0.998 between zero-shot ROUGE-L and fine-tuning gain (descriptive statistic, n=3 — not a significance claim).

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
| File | Model | ROUGE-L | 95% CI | SARI | BERTScore | FK-Grade |
|------|-------|---------|--------|------|-----------|----------|
| [results_openbio.json](results/nebius_evidence/results_openbio.json) | OpenBioLLM-8B | 0.6638 | 0.660–0.668 | 73.49 | 0.9460 | 7.33 |
| [results_mistral.json](results/nebius_evidence/results_mistral.json) | Mistral-7B | 0.6253 | 0.620–0.630 | 72.75 | 0.9418 | 6.14 |
| [results_biomistral.json](results/nebius_evidence/results_biomistral.json) | BioMistral-7B | 0.6004 | 0.595–0.605 | 71.97 | 0.9372 | 6.13 |
| [bootstrap_input.json](results/nebius_evidence/bootstrap_input.json) | OpenBioLLM-8B | Per-sample ROUGE-L scores (n=1,001) — source for 95% CI bootstrap (n=10,000 resamples) | | | |
| [eval_persamples_mistral.json](results/nebius_evidence/eval_persamples_mistral.json) | Mistral-7B | Per-sample ROUGE-L scores (n=1,001) — source for 95% CI bootstrap (n=10,000 resamples) | | | |
| [eval_persamples_biomistral.json](results/nebius_evidence/eval_persamples_biomistral.json) | BioMistral-7B | Per-sample ROUGE-L scores (n=1,001) — source for 95% CI bootstrap (n=10,000 resamples) | | | |

**Safety Evaluation** (`results/nebius_evidence/`):
- [safety_results_v2.json](results/nebius_evidence/safety_results_v2.json) — 1,001 samples, dual judge (Llama + Qwen), simple prompt
- [safety_results_v3.json](results/nebius_evidence/safety_results_v3.json) — 1,001 samples, dual judge (Llama + Qwen), 4-step CoT prompt
- [safety_results.json](results/nebius_evidence/safety_results.json) — v1 preliminary (100 samples, single Llama judge)

**Nebius Job Logs** (`results/nebius_logs/`):
| File | Contents |
|------|---------|
| [full_train_logs.json.gz](results/nebius_logs/full_train_logs.json.gz) | Full OpenBioLLM-8B training — H100, train_loss 0.844→0.635 |
| [endpoint_vllm_logs.json.gz](results/nebius_logs/endpoint_vllm_logs.json.gz) | vLLM endpoint startup + requests, vmapp_id: aiendpoint-e00ef3br6r14grvhhd |
| [adapters_logs.json.gz](results/nebius_logs/adapters_logs.json.gz) | Merge adapter job |
| [r32_all_8kdata.json.gz](results/nebius_logs/r32_all_8kdata.json.gz) | Ablation: r=32, all_attn, 8K data — ROUGE-L 0.6345 |
| [r32_all_4kdata_logs.json.gz](results/nebius_logs/r32_all_4kdata_logs.json.gz) | Ablation: r=32, all_attn, 4K data — ROUGE-L 0.6198 |
| [r32_all_attention_logs.json.gz](results/nebius_logs/r32_all_attention_logs.json.gz) | Ablation: r=32, all_attn modules |
| [r32_qv_logs.json.gz](results/nebius_logs/r32_qv_logs.json.gz) | Ablation: r=32, q+v modules — ROUGE-L 0.6192 |
| [r32_qonly_logs.json.gz](results/nebius_logs/r32_qonly_logs.json.gz) | Ablation: r=32, q only — ROUGE-L 0.6006 |
| [r16_qv_logs.json.gz](results/nebius_logs/r16_qv_logs.json.gz) | Ablation: r=16, q+v — ROUGE-L 0.6080 |
| [r8_qv_logs.json.gz](results/nebius_logs/r8_qv_logs.json.gz) | Ablation: r=8, q+v — ROUGE-L 0.6033 |
| [eval-persamples.json.gz](results/nebius_logs/eval-persamples.json.gz) | OpenBioLLM-8B evaluation job — 1,001 samples, ROUGE-L 0.6638 |
| [eval-persamples-mistral.json.gz](results/nebius_logs/eval-persamples-mistral.json.gz) | Mistral-7B evaluation job — 1,001 samples, ROUGE-L 0.6253 |
| [eval-persamples-biomistral.json.gz](results/nebius_logs/eval-persamples-biomistral.json.gz) | BioMistral-7B evaluation job — 1,001 samples, ROUGE-L 0.6004 |
| [eval-seed2.json.gz](results/nebius_logs/eval-seed2.json.gz) | OpenBioLLM-8B eval — seed=2 (multi-seed validation), ROUGE-L 0.6651 |
| [seed2.json.gz](results/nebius_logs/seed2.json.gz) | OpenBioLLM-8B training — seed=2, multi-seed validation run |
| [safety-eval-v2.json.gz](results/nebius_logs/safety-eval-v2.json.gz) | Safety eval v2 — dual judge, simple prompt, n=1,001 |
| [safety-eval-v3.json.gz](results/nebius_logs/safety-eval-v3.json.gz) | Safety eval v3 — dual judge, 4-step CoT prompt, n=1,001 |

> All logs contain Nebius job IDs (aijob-* / aiendpoint-*),
> GPU info (NVIDIA H100 80GB HBM3), and timestamps.
> Proving the full pipeline ran on Nebius infrastructure.

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
<summary>Medical Safety Evaluation v2 — Dual Judge, Simple Prompt (1,001 samples)</summary>

```json
{
  "n_samples": 1001,
  "n_evaluated_llama": 1001,
  "n_errors_llama": 0,
  "n_evaluated_qwen": 1000,
  "n_errors_qwen": 1,
  "llama_safe_rate": 0.3247,
  "qwen_safe_rate": 0.888,
  "cohen_kappa": 0.1114,
  "rouge_faithfulness_pearson_r": 0.2128,
  "judges": {"llama": "meta-llama/Llama-3.3-70B-Instruct", "qwen": "Qwen/Qwen3-32B"}
}
```
</details>

**🔗 Public Artifacts:**
- Merged model: [chambul/MediSimplifier-OpenBioLLM-merged](https://huggingface.co/chambul/MediSimplifier-OpenBioLLM-merged)
- Docker: [chambul/medisimplifier:train-v24](https://hub.docker.com/r/chambul/medisimplifier)
- W&B: [wandb.ai/deepset01-chambul/medisimplifier](https://wandb.ai/deepset01-chambul/medisimplifier)
- Endpoint: Deploy your own in ~5 minutes — see [Step 5](#5-deploy-live-endpoint) above
- MLflow: [Experiment export](results/nebius_evidence/mlflow_runs.csv) + [send_to_mlflow.py](send_to_mlflow.py) — restore live experiment in ~5 min
- LoRA Adapter (Nebius): [chambul/MediSimplifier-LoRA-Adapter-Nebius](https://huggingface.co/chambul/MediSimplifier-LoRA-Adapter-Nebius) — r=32, all_attn, 3 epochs, ROUGE-L 0.6638
- Dataset: [GuyDor007/medisimplifier-dataset](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)
- Adapters (Technion-era): [GuyDor007/MediSimplifier-LoRA-Adapters](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters)

## Medical Safety Evaluation

Beyond standard NLP metrics, I conducted a three-round investigation into
whether MediSimplifier preserves critical medical information — each round
designed to answer a question raised by the previous round's findings.

### Research Design

| Round | Research Question | Judge Design | Key Finding |
|-------|------------------|-------------|-------------|
| v1 (n=100) | What is the baseline faithfulness rate? | Single judge — Llama-3.3-70B (same-family) | 76.8% — but same-family judge may be biased |
| v2 (n=1,001) | Is evaluation judge-family dependent? | Two judges — Llama-70B + Qwen-32B: differ in **both family and scale** (intentional — to probe sensitivity across both dimensions simultaneously) | Llama 32.5% vs Qwen 88.8%, κ=0.11 |
| v3 (n=1,001) | Does structured reasoning reduce bias? | Same two judges + 4-step CoT with anti-sycophancy warning (inspired by my LLM evaluation coursework — Nebius Academy AI Performance Engineering) | CoT amplifies bias — κ drops further to 0.04 |

Each experiment used the full Nebius Token Factory pipeline — 4,004 total judge calls across all rounds.

### Methodology

Three-level evaluation per round:

| Level | Method | Model |
|-------|--------|-------|
| Rule-based | scispaCy entity preservation | `en_core_sci_sm` |
| LLM Judge A | Same-family judge | `meta-llama/Llama-3.3-70B-Instruct` |
| LLM Judge B | Cross-family judge | `Qwen/Qwen3-32B` |

Both LLM judges evaluate medical faithfulness using the same prompt.
Judge A (Llama) is from the **same model family** as OpenBioLLM-8B.
Judge B (Qwen) is from a **different model family** — enabling cross-family agreement analysis.
Both judges run via **Nebius Token Factory** serverless inference.

### Results

**Experiment 1 — Simple prompt (v2):**

| Evaluator | Safe | Unsafe | Safe Rate | n evaluated |
|-----------|------|--------|-----------|-------------|
| scispaCy (negative control) | 0 | 1,001 | 0.0% | 1,001 |
| Llama-3.3-70B (same-family) | 325 | 676 | **32.5%** | 1,001 (0 errors) |
| Qwen3-32B (cross-family) | 888 | 112 | **88.8%** | 1,000 (1 error) |

Inter-judge agreement: Cohen's κ = 0.1114 | ROUGE↔faithfulness: r = 0.2128

**Experiment 2 — 4-step CoT prompt with anti-sycophancy warning (v3):**

| Evaluator | Safe | Unsafe | Safe Rate | n evaluated |
|-----------|------|--------|-----------|-------------|
| Llama-3.3-70B (same-family) | 87 | 909 | **8.7%** | 996 (5 errors) |
| Qwen3-32B (cross-family) | 803 | 198 | **80.3%** | 1,001 (0 errors) |

Inter-judge agreement: Cohen's κ = 0.0431 | ROUGE↔faithfulness: r = 0.0725

**Key observation:** CoT prompting made Llama ~4× stricter (32.5% → 8.7%) while Qwen shifted only moderately (88.8% → 80.3%). Agreement *decreased* with CoT (κ: 0.11 → 0.04) — chain-of-thought amplified judge family bias rather than reducing it.

> **Note on scispaCy 0%:** Expected — exact-match entity comparison cannot
> recognize semantic equivalents (e.g., "myocardial infarction" → "heart attack").
> This confirms rule-based metrics are inappropriate for medical simplification.

### Novel Finding: Judge Family Bias in Medical Faithfulness Evaluation

**The most striking result is not the safe rate itself — it is the disagreement between judges.**

Cohen's κ = 0.11 indicates near-random agreement between Llama and Qwen on
whether a simplification is medically faithful. Llama-3.3-70B (same family as
OpenBioLLM) flags 67.5% of outputs as unsafe; Qwen3-32B flags only 11.2%.
When CoT prompting was added (v3), the bias intensified rather than diminished:
κ dropped further to 0.04, with Llama flagging 91.3% unsafe vs Qwen 19.7% —
suggesting that structured reasoning amplifies family-specific tendencies.

**This reveals a significant judge family bias:** LLM-as-judge medical
faithfulness evaluation is highly model-dependent. A same-family judge may
penalize outputs that share stylistic patterns with the evaluated model,
inflating the unsafe rate. Cross-family evaluation yields a more lenient
but potentially less biased assessment.

**Implication:** Medical AI evaluation requires multi-judge protocols with
cross-family diversity. Single-judge evaluation (as in most published work)
may systematically over- or under-estimate faithfulness depending on judge
selection.

> **ROUGE-L ↔ Faithfulness:** Pearson r = 0.21 — weak positive correlation,
> confirming that ROUGE-L alone is an insufficient proxy for medical faithfulness.
> High ROUGE-L does not guarantee faithful simplification.

> Evidence:
> - [safety_results_v2.json](results/nebius_evidence/safety_results_v2.json) — simple prompt, n=1,001
> - [safety_results_v3.json](results/nebius_evidence/safety_results_v3.json) — CoT prompt, n=1,001
> - [safety_results.json](results/nebius_evidence/safety_results.json) — v1 preliminary (100 samples, single judge)

> **Limitation and future work:** Without human-anchored ground truth, I cannot determine which judge is closer to clinical accuracy — only that they disagree strongly. LLM evaluation research consistently shows that judges tend to rate fluent, plausible-sounding text as factually accurate even when it contains errors (sycophancy toward fluency). A controlled annotation study with clinical reviewers would be needed to establish which judge family and which prompt structure best approximates human medical judgment — and would convert this from an observational finding into a validated one.

> Both experiments conducted via Nebius Token Factory API — demonstrating
> serverless LLM-as-judge evaluation at scale (1,001 samples × 2 judges × 2 prompt variants = 4,004 judge calls).

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

Winner configuration: **r=32, all_attn, 8K** → used for full 3-epoch training → final ROUGE-L **0.6638** (Nebius H100; 0.6749 on Technion H200).

> **Note on ablation overlap:** Phase 1 and Phase 2 share the r=32, q+v, 8K configuration (0.6183 vs 0.6192),
> and Phase 2/Phase 3 share r=32, all_attn, 8K (0.6357 vs 0.6345). These are not real performance differences.
> Each config was run once (n=1, seed=42) as a separate Nebius Job. Multi-seed validation on the winning config (seeds 42 and 2) measured variance of 0.0013 ROUGE-L — confirming differences this small
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
> Nebius Jobs. Set `WANDB_API_KEY` to log online; omit to run offline
> (`WANDB_MODE=offline` fallback — no key required).

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
| Parallel jobs | Native | Manual | Native |
| Job dependencies | Manual | Manual | Native ✅ |
| 9 ablations | 20 min, ~$9 | 3 hours | Setup overhead |
| Best for | Experimentation | Long stable training | Production pipelines |

> Serverless Jobs eliminated cluster management overhead that
> typically slows ML experimentation. Each job is stateless —
> the bucket is the only persistent state between runs.
> I submitted all 9 ablation jobs simultaneously and had
> results in 20 minutes instead of 3 hours sequentially.
> Kubernetes would have added native job dependencies (train → merge → eval → safety eval
> required manual monitoring between stages) — a tradeoff worth noting.
> For pure stateless experimentation, Serverless Jobs won on every dimension.

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
> I serve a merged 8B model at ~$3.85/hr only when the endpoint
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
| Ablation ×9 (parallel) | H100 NVLink | ~20 min total | ~$12 |
| OpenBioLLM-8B training | H100 NVLink | ~70 min | ~$4.50 |
| Mistral-7B training | H100 NVLink | ~70 min | ~$4.50 |
| BioMistral-7B training | H100 NVLink | ~70 min | ~$4.50 |
| Evaluation ×3 | H100 NVLink | ~45 min each | ~$9 |
| Safety evaluation | H100 NVLink | ~30 min | ~$2 |
| Merge + misc | H100 NVLink | — | ~$2 |
| Bootstrap CI eval ×3 (parallel) | H100 NVLink | ~45 min total | ~$9 |
| **Core pipeline subtotal** | | | **~$43** |

**Cost breakdown (actual Nebius billing):**

| Resource | Usage | Cost |
|----------|-------|------|
| NVIDIA H100 NVLink | 48.41 GPU hours | $186.39 |
| CPU (AMD EPYC Genoa) | 166.24 vCPU hours | $1.99 |
| RAM | 664.95 GiB hours | $2.13 |
| Network SSD disk | 50,087.83 GiB hours | $4.87 |
| Object Storage | medisimplifier-adapters bucket | $0.08 |
| **Total Nebius spend** | | **$195.47** |

> H100 NVLink rate: ~$3.85/hr on Nebius eu-north1.
> Core pipeline compute (ablation + training + eval + safety + merge + bootstrap CI): ~$43.
> Remaining ~$152 covers failed runs during debugging, endpoint serving,
> Build VM, and iteration — all on Nebius Serverless, no reserved instances.

> 9 parallel jobs = same wall-clock time as 1 job (~20 min total).

> Technion hardware: RunPod H200 SXM (~90 min/model across 3 GPUs).
> Nebius reproduction uses H100 NVLink (~70 min, single GPU per job).

## Reproduce step by step

> **TL;DR:** `bash scripts/reproduce.sh eval_only` evaluates from public HF adapters with no training required.
> `bash scripts/reproduce.sh full` runs the complete pipeline (training + ablations + eval + endpoint).
> Requires `NEBIUS_PROJECT_ID`, `NEBIUS_SUBNET_ID`, `HF_TOKEN` env vars and Nebius CLI v0.12.229+.

<details>
<summary>scripts/reproduce.sh — invocation modes (train-v24)</summary>

```bash
bash scripts/reproduce.sh eval_only    # evaluate only (no training, uses HF adapters)
bash scripts/reproduce.sh full         # full pipeline: ablation + train + eval + serve
bash scripts/reproduce.sh ablation     # ablation study only
bash scripts/reproduce.sh serve        # deploy vLLM endpoint only
```

See [`scripts/reproduce.sh`](scripts/reproduce.sh) for the complete implementation.
Expected output (eval_only): `rouge_l: 0.6638` (±0.002 CUDA variance)
</details>

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
  --image chambul/medisimplifier:train-v24 \
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

> **Runtime setup:** Jobs use `chambul/medisimplifier:train-v24` (public Docker Hub),
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
    --image chambul/medisimplifier:train-v24 \
    --container-command python \
    --args "train.py --model openbio --epochs 1 --rank ${RANK} --modules q_v --data-size 7999 --seed 42" \
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
    --image chambul/medisimplifier:train-v24 \
    --container-command python \
    --args "train.py --model openbio --epochs 1 --rank 32 --modules ${MODULES} --data-size 7999 --seed 42" \
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
    --image chambul/medisimplifier:train-v24 \
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
  --image chambul/medisimplifier:train-v24 \
  --container-command python \
  --args "evaluate.py --model openbio --adapter-path /mnt/adapters/full_training --split test --output-dir /mnt/adapters/eval_results" \
  --env HF_TOKEN=${HF_TOKEN} \
  --platform gpu-h100-sxm \
  --preset 1gpu-16vcpu-200gb \
  --disk-size 250Gi \
  --subnet-id ${NEBIUS_SUBNET_ID} \
  --volume medisimplifier-adapters:/mnt/adapters:rw \
  --timeout 5h
```

> **Adapter source options:**
> - **Option A — from bucket** (after running training): `--adapter-path /mnt/adapters/full_training`
> - **Option B — from HuggingFace** (no training needed): `--adapter-hf-repo chambul/MediSimplifier-LoRA-Adapter-Nebius`
>
> Replace `--adapter-path` with `--adapter-hf-repo chambul/MediSimplifier-LoRA-Adapter-Nebius` in the
> `--args` string above to evaluate directly from the Nebius-trained public adapter (produces ROUGE-L 0.6638).

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
> See [Inference Latency](#inference-latency-vllm-h100-nlink) below for benchmarks.

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

Benchmarked on a live Nebius H100 vLLM endpoint during development,
greedy decoding (temperature=0):

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
The merged model is publicly available on HuggingFace — see Step 5 above. For custom adapter deployment, see the Merge & Deploy Pipeline section.

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
      train.py           LoRA training — runs as Nebius Job
      evaluate.py        Metrics: ROUGE-L, SARI, BERTScore, FK-Grade
      safety_eval.py     LLM-as-judge safety evaluation v1 (single judge, n=100)
      safety_eval_v2.py  LLM-as-judge safety evaluation v2/v3 (dual judge, n=1,001)
      merge_adapter.py   Merge LoRA adapter into base model for vLLM serving
      save_adapter.py    Save adapter utility
      serve_vllm.py      vLLM inference server — runs as Nebius Endpoint
    docker/
      Dockerfile.train   Training image
      Dockerfile.serve   Serving image
    jobs/
      job_train.yaml         Full training job config
      job_ablation.yaml      Parametrized ablation job config
      job_evaluate.yaml      Evaluation job config
      job_safety_eval.yaml   Safety evaluation job config
      job_ablation_run.sh    Ablation submission shell script
      endpoint_vllm.yaml     vLLM endpoint deployment config
    scripts/
      reproduce.sh       One-command reproducibility script
    tests/
      test_metrics.py    Unit tests for ROUGE-L, FK-Grade, prompt builder
    send_to_mlflow.py    Nebius Managed MLflow experiment logging
    requirements.txt

## Key Configuration

All jobs use the `nebius ai job create` CLI. The training job parameters:

| Parameter | Value |
|-----------|-------|
| Image | `chambul/medisimplifier:train-v24` (all deps pre-installed, public Docker Hub) |
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
| Training quantization | 4-bit NF4 (QLoRA via bitsandbytes) |
| Evaluation quantization | fp16 (no quantization — higher precision for metrics) |

> **Train/eval quantization note:** Training uses 4-bit NF4 QLoRA for memory efficiency; evaluation loads the base model in fp16 for higher precision. This is a deliberate tradeoff: eval at full precision ensures metrics reflect true model quality rather than quantization artifacts. The H100→H200 delta (1.6–5.0%) captures hardware differences but not quantization precision differences.

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

# Prompt format: see src/train.py format_sample() for full CHATML/Mistral template
```

### Container Image

The training image is available on two registries:

**Docker Hub (public — for judges):**

    docker pull chambul/medisimplifier:train-v24

**Nebius Container Registry (used in job configs):**

    cr.eu-north1.nebius.cloud/e00p4ryvm6npw9w9pz/medisimplifier:train-v24

Built from `docker/Dockerfile.train`. To rebuild:

    docker build -t chambul/medisimplifier:train-v24 -f docker/Dockerfile.train .
    docker push chambul/medisimplifier:train-v24

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
  image: chambul/medisimplifier:train-v24
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

volumes:
  - bucket: medisimplifier-adapters
    mount: /output
    mode: rw

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
  image: chambul/medisimplifier:train-v24
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
torchaudio==2.1.0
pytest>=7.0.0
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
# FastAPI/uvicorn retained for Dockerfile base compatibility
# Production serving uses vLLM (see endpoint_vllm.yaml)
fastapi==0.110.0
uvicorn==0.29.0
boto3==1.34.0
mlflow>=2.0.0
requests>=2.31.0
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
| LoRA Adapter (Nebius) | [chambul/MediSimplifier-LoRA-Adapter-Nebius](https://huggingface.co/chambul/MediSimplifier-LoRA-Adapter-Nebius) — r=32, all_attn, 3 epochs, ROUGE-L 0.6638 |
| Merged Model | [chambul/MediSimplifier-OpenBioLLM-merged](https://huggingface.co/chambul/MediSimplifier-OpenBioLLM-merged) — OpenBioLLM-8B + adapter, ready for vLLM serving |
| Adapters (Technion-era) | [GuyDor007/MediSimplifier-LoRA-Adapters](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters) — 3 adapters from Technion project |

Dataset: [Asclepius-Synthetic](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic) (Apache 2.0, HCLS compliant).
Anonymized synthetic clinical notes. No real patient data.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
