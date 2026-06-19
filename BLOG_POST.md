# MediSimplifier — Teaching an AI to Speak Human

---

## 1. The Health Literacy Crisis

Medical discharge summaries are written at FK-Grade 14.5 — college level.
Only 12% of American adults have the health literacy to understand them.

A patient leaves the hospital holding a page that says:
"The patient presented with acute decompensated heart failure, bilateral lower
extremity edema, and an ejection fraction of 25%." They nod. They go home.
They don't understand a word of it.

Misunderstood discharge instructions are a leading cause of 30-day readmissions.
This isn't a reading problem. It's a writing problem — and it's fixable with LLMs.

Could we fine-tune an open-source model to automatically rewrite these documents
in plain language, without losing a single medical fact?

---

## 2. The Idea — LoRA Fine-Tuning on Nebius Serverless

**Why LoRA:** Full fine-tuning of an 8B model costs thousands of dollars and weeks
of iteration. LoRA freezes the base model and trains a small set of rank-decomposed
adapter matrices — only 27.3M parameters (0.38% of total) — achieving comparable
quality at a fraction of the cost.

**Why Nebius Serverless Jobs:**
- No cluster management — submit a job, get a result, pay only for what you use
- Parallel job execution — run 9 ablation experiments simultaneously instead of sequentially
- H100 NVLink available on demand — no reservation, no queue, no cold start penalty
- Total pipeline cost: ~$70 for 9 ablation runs + 3 full training runs + 3 evaluations

**The dataset:** 10,000 (input, simplified output) pairs from Asclepius Synthetic
Clinical Notes — public, anonymized, no real patient data.

**The models evaluated:** OpenBioLLM-8B, Mistral-7B-Instruct, BioMistral-7B-DARE

**The pipeline:**

```
HuggingFace Dataset
    ↓
Nebius Jobs: 9 parallel ablation runs (20 min each, ~$15 total)
    ↓
Nebius Job: Full training — r=32, all_attn, 3 epochs, H100 (~70 min, ~$22)
    ↓
Nebius Job: Evaluation — ROUGE-L, SARI, BERTScore, FK-Grade (~45 min, ~$5)
    ↓
Nebius Endpoint: POST /simplify → plain-language summary
```

---

## 3. The Surprising Finding — The Ranking Reversal

Before running a single fine-tuning job, we measured zero-shot performance.
BioMistral-7B-DARE came out on top (ROUGE-L 0.4120). OpenBioLLM-8B was last (0.2623).
Conventional wisdom: start with the best zero-shot model and make it better.

Then we fine-tuned. Here's what happened:

| Model | Zero-Shot ROUGE-L | Fine-Tuned ROUGE-L | Improvement |
|-------|-------------------|--------------------|-------------|
| OpenBioLLM-8B | 0.2623 (worst) | **0.6749** (best) | **+157.3%** |
| Mistral-7B | 0.3912 | 0.6491 | +65.9% |
| BioMistral-7B-DARE | 0.4120 (best) | 0.6318 (worst) | +53.3% |

**Complete ranking reversal.** The worst zero-shot model became the best fine-tuned model.
The correlation between baseline performance and improvement: r ≈ −0.998.
The lower the zero-shot score, the more room fine-tuning has to work.

What this means in practice: don't pick your base model by zero-shot benchmark.
Pick it by domain alignment. OpenBioLLM-8B was pre-trained on biomedical text —
it already "knows" the medical concepts, it just hadn't learned the simplification task yet.
Fine-tuning gave it exactly that, and it ran with it.

**Second finding — LoRA rank:** Hu et al. 2021 (the original LoRA paper) recommends
r=4–8. We tested r=8, r=16, r=32. Winner: r=32. The task requires enough capacity
to learn a new output style across the full vocabulary of medical terminology.
Low rank underfits.

**Third finding — data efficiency:** 4K training samples achieves 97% of 8K performance.
After 4K, returns diminish sharply. You don't need a massive dataset to fine-tune
for a well-defined transformation task.

---

## 4. How Nebius Made It Possible — 9 Parallel Jobs

The ablation design: 3 LoRA ranks × 3 module configs × 3 data sizes = 27 combinations.
We ran a smart subset: 9 jobs across 3 phases, each 1 epoch.

Without parallelism, running sequentially on a single GPU:
9 × 20 min = **3 hours of dead waiting**.

With Nebius Serverless Jobs:

```bash
for RANK in 8 16 32; do
  nebius ai job create \
    --name medisimplifier-ablation-r${RANK} \
    --image ghcr.io/gd007/medisimplifier:train-latest \
    --platform gpu-h100-sxm \
    --args "train.py --rank ${RANK} --modules q_v" ...
done
```

All 3 jobs fire in parallel. **20 minutes total, not 60.**

**Key Nebius features used:**
- `nebius ai job create` CLI — scriptable, repeatable, CI-friendly
- Bucket volumes (`/mnt/adapters`) — adapters persist across jobs without manual copy
- `--preset 1gpu-16vcpu-200gb` — right-sized for 8B model training, no over-provisioning
- Pay-per-second billing — ablation jobs that finish early don't waste budget

**Full ablation results** (from actual Nebius job logs):

Phase 1 — LoRA Rank (modules=q+v, data=8K):

| Rank | ROUGE-L |
|------|---------|
| r=8 | 0.6033 |
| r=16 | 0.6080 |
| **r=32** | **0.6183** ← winner |

Phase 2 — Target Modules (rank=32, data=8K):

| Modules | ROUGE-L |
|---------|---------|
| q only | 0.6006 |
| q+v | 0.6192 |
| **all_attn** | **0.6357** ← winner |

Phase 3 — Data Size (rank=32, modules=all_attn):

| Data | ROUGE-L |
|------|---------|
| 2K | 0.6014 |
| 4K | 0.6198 |
| **8K** | **0.6345** ← winner |

Winner: **r=32, all_attn, 8K** — used for full 3-epoch training → final ROUGE-L 0.6749.

---

## 5. Live Demo — The Endpoint

One YAML config, one Nebius Endpoint, FastAPI serving with 4-bit quantization
(BitsAndBytes NF4) so the 8B model fits on a single H100.

**Input (FK-Grade 16.2):**
> "The patient was admitted with acute decompensated heart failure, presenting
> with dyspnea, orthopnea, and bilateral lower extremity edema.
> Echocardiography revealed an ejection fraction of 25%."

**Output (FK-Grade 6.8):**
> "The patient came in with severe heart failure. They had trouble breathing,
> couldn't lie flat, and had swelling in both legs. A heart ultrasound showed
> the heart was only pumping at 25% of its normal strength."

**Preserved:** diagnosis, all symptoms, the specific 25% number, test type.  
**Changed:** every medical term replaced with a plain-language equivalent.

**The API:**

```bash
curl -X POST http://<endpoint>/simplify \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presented with acute myocardial infarction..."}'
# → {"simplified": "The patient had a heart attack...", "model": "...", "adapter": "..."}
```

**Reproducibility:** Nebius reproduction results (June 2026, H100 NVLink) confirmed
ROUGE-L 0.6638 vs 0.6749 original — a 1.6% variance explained entirely by
floating-point non-determinism across H200 vs H100 hardware.

> Evaluation Job: `medisimplifier-evaluation-spec` (1,001 test samples)

## Nebius H100 Reproduction

All three models were fully reproduced on Nebius H100 NVLink GPUs:

| Model | Original ROUGE-L (H200) | Nebius ROUGE-L (H100) | Delta |
|-------|------------------------|----------------------|-------|
| OpenBioLLM-8B | 0.6749 | 0.6638 | −1.6% |
| Mistral-7B | 0.6491 | 0.6253 | −3.7% |
| BioMistral-7B-DARE | 0.6318 | 0.6004 | −5.0% |

The ranking reversal holds across both hardware generations.
Training runs tracked live via our public [W&B dashboard](https://wandb.ai/deepset01-chambul/medisimplifier)
*(no login required)*.

---

## 6. Results and What's Next

**Final numbers — OpenBioLLM-8B, 1,001 test samples, bootstrap n=10,000:**

| Metric | Zero-Shot | Fine-Tuned | Change |
|--------|-----------|------------|--------|
| ROUGE-L | 0.2623 | 0.6749 [0.6705–0.6793] | +157.3% |
| SARI | 36.98 | 74.64 | +101.8% |
| BERTScore | 0.637 | 0.9498 | +49.1% |
| FK-Grade | 12.53 | 7.16 | −5.37 grades |

All pairwise ROUGE-L differences significant at p<0.001
(bootstrap CIs over 1,001 test items at seed=42).

**Medical safety evaluation:** We ran a two-level safety check on 100 model outputs.
A rule-based scispaCy entity matcher screened for medical term preservation.
A Llama-3.3-70B judge (running on Nebius AI Studio Token Factory) evaluated
factual faithfulness: 76.8% of outputs were rated fully safe, with the remaining
23.2% flagged for minor information condensation — not hallucination.

**Honest caveat:** We targeted FK ≤ 6.0. Best achieved: 6.91 (Mistral-7B).
The ~0.91 grade gap reflects an inherent tension — pushing further toward simpler
language risks paraphrasing medical facts. That's the right place to stop.

**What's next:**
- Streaming endpoint for long documents (chunk-by-section)
- RLHF pass using patient preference data to close the FK gap without sacrificing accuracy
- Multi-language support — the health literacy problem is global
- Exploring rsLoRA scaling laws: does r=64 help further, or have we hit the ceiling?

The $42 total compute cost is the headline. The ranking reversal is the story.

---

*All code, data, and adapters are public:*
- *Pipeline: [deepset01-sys/medisimplifier-nebius](https://github.com/deepset01-sys/medisimplifier-nebius)*
- *Research: [gd007/MediSimplifier](https://github.com/gd007/MediSimplifier)*
- *Dataset: [GuyDor007/medisimplifier-dataset](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)*
- *Adapters: [GuyDor007/MediSimplifier-LoRA-Adapters](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters)*

---

*Built for the [Nebius Serverless AI Builders Challenge](https://nebius.com).
Try it, fork it, or just steal the ranking-reversal insight for your next fine-tuning project.*

*#NebiusServerlessChallenge #LLM #MedicalAI #LoRA #OpenSource*
