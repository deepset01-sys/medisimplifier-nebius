# MediSimplifier — קונטקסט מלא לסשנים הבאים
**עודכן לאחרונה:** יוני 2026  
**ציון נוכחי:** 49/60  
**Deadline:** 30 יוני 2026

---

## 👥 הצוות

- **שמוליק** — סטודנט ב-Nebius Academy AI Performance Engineering IL
- **גיא דור** — שותף מפרויקט הגמר המקורי (Technion DS25)
- **Claude** — שותף פיתוח לאורך כל הדרך

---

## 🎯 מה אנחנו עושים

**האתגר:** Nebius Serverless AI Builders Challenge  
**הפרויקט:** MediSimplifier — fine-tuning של LLMs לפישוט מסמכים רפואיים  
**הבסיס:** פרויקט גמר של שמוליק וגיא מ-Technion DS25 Deep Learning

---

## 📁 ריפואים

| ריפו | תפקיד |
|------|--------|
| `deepset01-sys/medisimplifier-nebius` | הריפו הראשי לאתגר ניביוס |
| `gd007/MediSimplifier` | פרויקט הגמר המקורי עם notebooks, IEEE paper, כל התוצאות |
| `GuyDor007/medisimplifier-dataset` | Dataset ב-HuggingFace (10K samples, public) |
| `GuyDor007/MediSimplifier-LoRA-Adapters` | LoRA adapters ב-HuggingFace (3 מודלים, public) |

---

## 📊 תוצאות המחקר — הנתונים האמיתיים

### Zero-Shot Baseline
| Model | ROUGE-L | SARI | BERTScore | FK-Grade |
|-------|---------|------|-----------|----------|
| OpenBioLLM-8B | 0.2623 | 36.98 | 0.637 | 12.53 |
| Mistral-7B | 0.3912 | 46.38 | 0.734 | 10.60 |
| BioMistral-7B-DARE | 0.4120 | 51.91 | 0.743 | 9.52 |

### After Fine-Tuning (LoRA r=32, all_attn)
| Model | ROUGE-L | SARI | BERTScore | FK-Grade | Improvement |
|-------|---------|------|-----------|----------|-------------|
| OpenBioLLM-8B | 0.6749 [0.6705–0.6793] | 74.64 | 0.9498 | 7.16 | **+157.3%** |
| Mistral-7B | 0.6491 [0.6445–0.6537] | 73.79 | 0.9464 | 6.91 | +65.9% |
| BioMistral-7B-DARE | 0.6318 [0.6272–0.6365] | 73.01 | 0.9439 | 6.95 | +53.3% |

### Nebius Reproduction (H100 NVLink, 13 June 2026)
| Metric | Original | Nebius | Delta |
|--------|----------|--------|-------|
| ROUGE-L | 0.6749 | 0.6638 | -1.6% |
| SARI | 74.64 | 73.49 | -1.5% |
| BERTScore | 0.9498 | 0.9460 | -0.4% |
| FK-Grade | 7.16 | 7.33 | +0.17 |

### Ablation Results (אמיתיים מ-JSON logs)
**Phase 1 — Rank (q+v, 8K):** r=8→0.6033, r=16→0.6080, **r=32→0.6183** ✓  
**Phase 2 — Modules (r=32, 8K):** q_only→0.6006, q_v→0.6192, **all_attn→0.6357** ✓  
**Phase 3 — Data (r=32, all_attn):** 2K→0.6014, 4K→0.6198, **8K→0.6345** ✓

---

## 🔑 הממצאים המדעיים המרכזיים

1. **Ranking Reversal** — OpenBioLLM היה הכי גרוע zero-shot (0.2623) → הכי טוב fine-tuned (0.6749)
2. **Correlation r≈-0.998** — ככל שהמודל גרוע יותר zero-shot, כך מרוויח יותר מ-fine-tuning
3. **r=32 > r=4-8** — מנצח את המלצת Hu et al. 2021 ב-domain-specific task
4. **4K = 97% של 8K** — diminishing returns אחרי 4K samples
5. **FK target miss** — יעד היה ≤6.0, הושג 6.91 — מתועד בכנות

---

## 🏗️ ארכיטקטורה הנוכחית

### Pipeline
```
HuggingFace Dataset (10K)
    ↓
Nebius Jobs: 9 ablations במקביל (H100, ~20 min each, ~$15)
    ↓
Nebius Job: Full training (H100, ~70 min, ~$22)
    ↓
Nebius Job: Evaluation (H100, ~45 min, ~$5)
    ↓
Nebius Endpoint: FastAPI /simplify → plain text
Total cost: ~$42
```

### Adapter Storage Flow
```
Training Job → /output/adapter → medisimplifier-adapters bucket
                                          ↓
Eval/Serve Jobs ← /mnt/adapters/full_training
```

### Docker Image
```
cr.eu-north1.nebius.cloud/e00p4ryvm6npw9w9pz/medisimplifier:train-v1
```
בנוי מ-`docker/Dockerfile.train`, נדחף ל-Nebius Container Registry.

### LoRA Hyperparameters
| פרמטר | ערך |
|--------|-----|
| rank (r) | 32 |
| alpha (α) | 64 (= 2×r) |
| modules | q, k, v, o (all_attn) |
| rsLoRA | True |
| dropout | 0.05 |
| trainable params | 27.3M (0.38%) |
| seed | 42 |
| LR | 2e-4 (cosine, 3% warmup) |
| batch | 4 × grad_accum=4 = effective 16 |
| precision | BF16 |
| max_seq | 2048 |

---

## 🌐 Nebius Infrastructure

### IDs חשובים
| משאב | ID |
|------|-----|
| Tenant | tenant-e00ryr4vttfg4hrnxm (chambul-y83) |
| Project | project-e00g1ev2pr00wjxv40r6ga (default-project-eu-north1) |
| Subnet | vpcsubnet-e00jsdqfjrz04ygxc0 |
| Container Registry | registry-e00p4ryvm6npw9w9pz |
| Registry URL | cr.eu-north1.nebius.cloud/e00p4ryvm6npw9w9pz |
| Service Account | serviceaccount-e00vez9f3d2mv2p7ar (medisimplifier-sa) |
| Public Key ID | publickey-e00qxxdba17g1g1eag |
| Object Storage | medisimplifier-adapters bucket |

### Build VM
- **Name:** medisimplifier-build-vm
- **SSH:** `ssh -i C:\Users\User\.ssh\nebius_vm ubuntu@{IP}`
- **IP:** dynamic — בדוק בקונסולה לפני חיבור
- **Private key:** `/tmp/private.pem` (על ה-VM)

### Docker Login (לכל סשן build חדש)
```bash
nebius iam get-access-token | sudo docker login \
  cr.eu-north1.nebius.cloud --username iam --password-stdin
```

---

## 📋 מצב הריפו

### Commit נוכחי
```
4fd2483 — feat: use pre-built Nebius Container Registry image in all jobs
```

### קבצים עיקריים
```
src/
  train.py      — LoRA training עם seed=42
  evaluate.py   — 4 metrics + torchaudio stub + --adapter-hf-repo flag
  serve.py      — FastAPI + torchaudio stub + /health + validation
docker/
  Dockerfile.train  — מובנה ל-cr.eu-north1.nebius.cloud
  Dockerfile.serve
jobs/
  job_train.yaml    — כולל volumes mount לbucket
  job_evaluate.yaml
  job_ablation.yaml
  endpoint_serve.yaml
requirements.txt    — כל deps pinned ל-==
LICENSE             — Apache 2.0
BLOG_POST.md        — טיוטת הבלוג (עדיין לא פורסם)
```

---

## 🚨 בעיות ידועות שנפתרו

| בעיה | פתרון |
|------|--------|
| torchaudio OSError | stub בראש evaluate.py ו-serve.py |
| SARI refs_sents shape | שינוי `[[r] for r in refs]` ל-`[refs]` |
| easse@main לא קיים | שינוי ל-`@master` |
| docker login permission | `sudo cp ~/.nebius/config.yaml /root/.nebius/` |
| easse pip install נכשל | הסרת `bert-score` מpip + lazy import |
| evaluation loop איטי | החלפה ל-`generate_predictions` עם batch_size=4 |

---

## 🎯 משימות לסשן הבא — לפי סדר עדיפויות

### 1. vLLM Endpoint (הכי חשוב — ישנה ציון)
**הבעיה:** ה-Endpoint הנוכחי הוא FastAPI על raw IP — לא Nebius Serverless Endpoint אמיתי.  
**הפתרון:** Merge LoRA + vLLM serving

**שלב A — Merge Job:**
```python
# merge_adapter.py
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("aaditya/Llama3-OpenBioLLM-8B")
model = PeftModel.from_pretrained(model, "/mnt/adapters/full_training")
merged = model.merge_and_unload()
merged.save_pretrained("/mnt/adapters/merged_openbio")
tokenizer.save_pretrained("/mnt/adapters/merged_openbio")
```

**שלב B — vLLM Endpoint:**
```yaml
image: vllm/vllm-openai:latest
command: python -m vllm.entrypoints.openai.api_server
  --model /mnt/adapters/merged_openbio
  --port 8000
volumes:
  - bucket: medisimplifier-adapters
    mount: /mnt/adapters
    mode: ro
```

**שלב C — עדכון serve.py לOpenAI format:**
```python
# POST /v1/chat/completions
# OpenAI-compatible
```

### 2. הסרת pip install מ-`--args` (קל, +1 נקודה)
כל ה-CLI commands עדיין מכילים `pip install ... &&` — סותר את ה-Docker image.  
פשוט להסיר — ה-image כבר מכיל הכל.

### 3. Seed=42 explanation תיקון (קל)
הסבר שonlytorch/numpy seeded, לא HF datasets shuffle.

### 4. Blog Post פרסום (חובה להגשה)
פרסם ב-Medium/LinkedIn עם `#NebiusServerlessChallenge`  
יש מסמך תשתית מוכן: `BLOG_INFRASTRUCTURE.md`

### 5. Video Walkthrough (מומלץ)
3-10 דקות ב-Loom/YouTube  
מה בנינו, למה ניביוס, ranking reversal finding

### 6. הגשה סופית
`https://checker.academy.nebius.com`

---

## 📈 היסטוריית ציונים

| גרסה | ציון | שינוי עיקרי |
|------|------|-------------|
| v1 | 37/60 | מקור |
| v2 | 41/60 | baseline table + job names |
| v4 | 43/60 | ablation table + training time |
| v6 | 42/60 | CLI כprimary |
| v7 | 44/60 | ablation+eval CLI, live demo |
| v9 | 43/60 | YAML + Docker org |
| v10 | 49/60 | כל תיקוני README |
| v10b | 49/60 | bucket creation + HF note |
| v12 | 49/60 | Docker image אמיתי |

**טווח רעש סטטיסטי:** 45-50  
**יעד עם vLLM:** 54+/60

---

## 🔍 מה reviewer אומר בכל review (סיגנל חזק)

| בעיה | פעמים | פתרון |
|------|-------|--------|
| Dockerfiles dead code | ✅ נפתר עם train-v1 | Docker image |
| pip install ב-args | 5+ | להסיר |
| Endpoint = raw IP | 6+ | vLLM |
| YAML schema לא מאומת | 7+ | נשאר open |

---

## 📚 הקשר לקורס Nebius Academy

שמוליק לומד AI Performance Engineering IL בניביוס.  
ציוני הקורס עד כה:
- LLM Evaluation: 93/100
- RAG: 109.5/110
- Building Agent: **120/120** 🏆
- LLM Architecture HW1: 98/100
- LLM Architecture HW2: 92/100
- LLM Architecture HW3: 96/100
- MLOps HW1: בהמתנה
- MLOps HW2: הגשה 20.6
- Performance Engineering HW1: בקרוב

**הנקודה לבלוג:** הפרויקט מביא לידי ביטוי את כל מה שנלמד בקורס.

---

## 🏆 Job History בניביוס

Jobs שרצו בהצלחה:
- `medisimplifier-full-training` — training OpenBioLLM-8B ✅
- `medisimplifier-ablation-r8/r16/r32` — ablation phase 1 ✅
- `medisimplifier-evaluation-spec` — evaluation עם fast mode ✅
- `medisimplifier-evaluation-full2` — evaluation מלא 4 מטריקות ✅

Endpoints שרצו:
- `medisimplifier-serve-v5` — FastAPI, IP: 89.169.123.166:8000 ✅ (נעצר)

---

## 💡 הערות חשובות

1. **ה-private key** `/tmp/private.pem` נמצא על ה-Build VM — אם ה-VM נמחק, צריך לחזור על תהליך יצירת ה-key
2. **easse** — חייב `@master` לא `@main` ב-requirements.txt
3. **torchaudio stub** — חייב להיות בראש כל קובץ Python שמייבא transformers
4. **Docker login** — token פג כל ~12 שעות, יש לחדש לפני כל push
5. **VM IP** — dynamic, בדוק בקונסולה לפני SSH
6. **גיא** — יכול לפתוח את ה-GHCR image אם יהיה צורך (עדיין לא ענה)
