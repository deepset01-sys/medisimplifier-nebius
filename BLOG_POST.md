# Medical Text Simplification with LoRA on Nebius Serverless: A Builder's Journey

---

## 1. The Problem Worth Solving

Medical discharge summaries are written at FK-Grade 14.5 — college level.
Only 12% of American adults have the health literacy to understand them.

A patient leaves the hospital holding a page that says:
> "The patient presented with acute decompensated heart failure, bilateral lower
> extremity edema, and an ejection fraction of 25%."

They nod. They go home. They don't understand a word of it.

Misunderstood discharge instructions are a leading cause of 30-day readmissions.
This isn't a reading problem. It's a writing problem — and it's fixable with LLMs.

Could we fine-tune an open-source model to automatically rewrite these documents
in plain language, without losing a single medical fact?

---

## 2. What We Built — and How Nebius Made It Possible

**The stack:** Three 7–8B models, LoRA fine-tuning, Nebius Serverless Jobs and Endpoints,
Object Storage for adapter persistence, and Token Factory for LLM-as-judge safety evaluation.

**Why LoRA:** Full fine-tuning of an 8B model costs thousands of dollars and weeks
of iteration. LoRA freezes the base model and trains a small set of rank-decomposed
adapter matrices — only 27.3M parameters (0.38% of total) — achieving comparable
quality at a fraction of the cost.

**Why Nebius Serverless Jobs specifically:**

We needed to run 9 ablation experiments (3 ranks × 3 module configs × 3 data sizes)
before committing to a full training run. On a single GPU, that's 3 hours of sequential
waiting. With Nebius Serverless Jobs, we submitted all 9 in parallel:

```bash
for RANK in 8 16 32; do
  nebius ai job create \
    --name medisimplifier-ablation-r${RANK} \
    --image chambul/medisimplifier:train-v11 \
    --platform gpu-h100-sxm \
    --preset 1gpu-16vcpu-200gb \
    --volume medisimplifier-adapters:/output:rw \
    --env HF_TOKEN=${HF_TOKEN} \
    --timeout 30m
done
```

20 minutes total, not 180. That's the difference between iterating in a day
versus a week.

**Object Storage as the backbone:**
Every adapter, checkpoint, and eval result flows through one bucket:
Training Job → /output/adapter → medisimplifier-adapters bucket
↓
Eval Job ← /mnt/adapters/full_training ←─┘
No manual copying. No lost checkpoints. Jobs are stateless; the bucket is the state.

**The pipeline in full:**
HuggingFace Dataset (10K samples, Asclepius Synthetic)
↓
Nebius Jobs: 9 parallel ablation runs (20 min each, ~$15 total)
↓
Nebius Job: Full training — r=32, all_attn, 3 epochs, H100 (~70 min, ~$22 each × 3 models)
↓
Nebius Job: Evaluation — ROUGE-L, SARI, BERTScore, FK-Grade (~45 min each)
↓
Nebius Job: Safety eval — scispaCy + Llama-3.3-70B via Token Factory
↓
Nebius Endpoint: vLLM serving merged adapter, OpenAI-compatible API

Total cost: ~$70. Total wall-clock time from first job to live endpoint: ~2 days.

---

## 3. What Didn't Work (And What We Learned)

This is the part that doesn't make it into papers.

**Bug 1: The safetensors permission wall**

Every training run — all three models — completed 100% of training steps and then
crashed at the final save:
SafetensorError: I/O error: Operation not permitted (os error 1)

The H100 job writes to `/output`, which is bucket-mounted read-write. But
`safetensors` format tries to create temporary files in a way the FUSE mount
doesn't allow. The fix was two lines:

```python
# In TrainingArguments:
save_safetensors=False

# In the final save:
model.save_pretrained(output_path, safe_serialization=False)
```

We lost about 4 hours to this across the three models. The lesson: test your
save path on a 1-step job before a 70-minute training run.

**Bug 2: The PEFT/TRL version dance**
TypeError: SFTTrainer.init() got an unexpected keyword argument 'processing_class'

The `trl` library renamed `tokenizer` to `processing_class` in one version, then
back again. We ended up pinning every dependency to exact versions in requirements.txt
— `peft==0.14.0`, `transformers==4.45.0`, `trl==0.8.6` — and building that into
the Docker image. Once the image was stable, every job was reproducible.

**Bug 3: The easse install that wasn't**

SARI score depends on `easse`, which had a broken `@main` branch. The fix:
pin to a specific commit SHA:
easse @ git+https://github.com/feralvam/easse.git@6a4352ec299ed03fda8ee45445ca43d9c7673e89

The broader lesson: in a Serverless Jobs environment where every job starts
from scratch, a flaky `pip install` is a 20-minute debugging session in disguise.
Pin everything.

**Decision: Build a Docker image first**

After the first few failed jobs, we made a decision that changed everything:
stop pip-installing at job start time, and bake all dependencies into a Docker
image pushed to both Docker Hub and Nebius Container Registry.

```bash
docker build -t chambul/medisimplifier:train-v11 \
             -t cr.eu-north1.nebius.cloud/.../medisimplifier:train-v11 \
             -f docker/Dockerfile.train .
docker push chambul/medisimplifier:train-v11
```

Cold start went from ~8 minutes (pip install on every job) to ~45 seconds
(image pull from registry). That's the real Nebius MLOps workflow.

**The moment the first job worked**

After three days of debugging, we submitted `medisimplifier-full-training` and
watched the logs stream in Nebius Console:
Train: 7999 | Val: 999 | Test: 1001
Starting training...
{'loss': 1.1339, 'epoch': 0.1}
{'loss': 0.8735, 'epoch': 0.2}
...
100%|██████████| 1500/1500 [1:43:13]
train_loss: 0.756 → 0.635

Then silence. Then the safetensors crash. We'd been here before.

But this time we had `save_adapter.py` ready — a recovery script that loads
from the last checkpoint and saves with `safe_serialization=False`.
Loading from: /mnt/adapters/full_training/checkpoint-1500
Saving adapter to: /mnt/adapters/full_training
Done!

That "Done!" after three days was genuinely satisfying.

---

## 4. The Finding We Didn't Expect

Before running a single fine-tuning job, we measured zero-shot performance.
BioMistral-7B-DARE came out on top (ROUGE-L 0.4120). OpenBioLLM-8B was last (0.2623).
Conventional wisdom: start with the best zero-shot model and make it better.

Then we fine-tuned. Here's what happened:

| Model | Zero-Shot ROUGE-L | Fine-Tuned ROUGE-L | Improvement |
|-------|-------------------|--------------------|-------------|
| OpenBioLLM-8B | 0.2623 (worst) | **0.6638** (best) | **+153%** |
| Mistral-7B | 0.3912 | 0.6253 | +60% |
| BioMistral-7B-DARE | 0.4120 (best) | 0.6004 (worst) | +46% |

**Complete ranking reversal.** The worst zero-shot model became the best fine-tuned model.
The correlation between baseline performance and improvement: r ≈ −0.998.

What this means in practice: **don't pick your base model by zero-shot benchmark.**
Pick it by domain alignment. OpenBioLLM-8B was pre-trained on biomedical text —
it already "knows" the medical concepts, it just hadn't learned the simplification
task yet. Fine-tuning gave it exactly that.

**The H100 reproduction held:**
We ran all three models on Nebius H100 NVLink. The ranking reversal reproduced
exactly — confirming this isn't a hardware artifact.

| Model | Original ROUGE-L (H200) | Nebius ROUGE-L (H100) | Delta |
|-------|------------------------|----------------------|-------|
| OpenBioLLM-8B | 0.6749 | 0.6638 | −1.6% |
| Mistral-7B | 0.6491 | 0.6253 | −3.7% |
| BioMistral-7B-DARE | 0.6318 | 0.6004 | −5.0% |

Training runs tracked live on our public
[W&B dashboard](https://wandb.ai/deepset01-chambul/medisimplifier) *(no login required)*.

**Three more findings from the ablation:**
- r=32 > r=16 > r=8 (contrary to Hu et al. 2021 recommendation of r=4–8)
- `all_attn` (q+k+v+o) > `q+v` > `q only`
- 4K training samples achieves 97% of 8K performance — diminishing returns kick in fast

---

## 5. Live Demo

One YAML config, one Nebius Endpoint, vLLM serving the merged adapter.

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

```bash
curl http://89.169.110.2:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/adapters/merged_openbio",
    "prompt": "Simplify: The patient presented with acute myocardial infarction.\n\nSimplified:",
    "max_tokens": 200,
    "temperature": 0
  }'
```

*(Endpoint active June 29–30, 2026 for judging window)*

**Medical safety evaluation:**
We ran a two-level check on 100 outputs using Nebius Token Factory:
a scispaCy rule-based screener and a Llama-3.3-70B judge.
Result: 76.8% rated fully safe. The 23.2% flagged had minor condensation,
not hallucination — the right failure mode for a simplification task.

---

## 6. What We Learned

**On Nebius:**
- Serverless Jobs are genuinely serverless. No cluster to manage, no quota to reserve.
  The H100 was available immediately every time we submitted.
- Object Storage bucket mounts work well for adapter persistence — with one caveat:
  safetensors FUSE writes need `safe_serialization=False`.
- Token Factory (LLM-as-judge) is powerful for domain-specific eval. We used
  Llama-3.3-70B as a medical safety evaluator for ~$2 total.
- Build your Docker image first. Every hour spent on the image saves 10 hours
  of debugging flaky pip installs inside jobs.

**On LoRA fine-tuning:**
- Zero-shot rank ≠ fine-tuning potential. Domain pre-training matters more.
- r=32 is not too large for an 8B model on a well-defined task. The original
  LoRA paper's r=4–8 recommendation comes from NLU tasks, not generation.
- 4K samples is a good starting point. Double the data for 3% more performance
  is a poor trade.
- Pin every dependency. In a containerized job environment, version drift is silent death.

**The $70 question:**
The full pipeline — 9 ablation jobs + 3 full training runs + 3 evals + safety eval
+ endpoint deployment — cost ~$70 on Nebius H100. That's less than one hour
of a machine learning engineer's time. The compute is not the bottleneck anymore.

---

*All code, data, and adapters are public:*
- *Pipeline: [deepset01-sys/medisimplifier-nebius](https://github.com/deepset01-sys/medisimplifier-nebius)*
- *Research: [gd007/MediSimplifier](https://github.com/gd007/MediSimplifier)*
- *Dataset: [GuyDor007/medisimplifier-dataset](https://huggingface.co/datasets/GuyDor007/medisimplifier-dataset)*
- *Adapters: [GuyDor007/MediSimplifier-LoRA-Adapters](https://huggingface.co/GuyDor007/MediSimplifier-LoRA-Adapters)*
- *W&B: [wandb.ai/deepset01-chambul/medisimplifier](https://wandb.ai/deepset01-chambul/medisimplifier)*

---

*Built for the [Nebius Serverless AI Builders Challenge](https://nebius.com).
Try it, fork it, or just steal the ranking-reversal insight for your next fine-tuning project.*

*#NebiusServerlessChallenge #LLM #MedicalAI #LoRA #OpenSource*
