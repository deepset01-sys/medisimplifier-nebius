import sys
import types

# torchaudio is imported by some HuggingFace internals but not needed
# for text evaluation. This stub prevents ImportError on environments
# where torchaudio is not installed (e.g., Nebius Jobs with torch-only image).
_torchaudio = types.ModuleType('torchaudio')
_torchaudio.__version__ = '0.0.0'
_torchaudio.__spec__ = types.SimpleNamespace(
    name='torchaudio',
    origin=None,
    submodule_search_locations=[]
)
sys.modules['torchaudio'] = _torchaudio

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer
from easse.sari import corpus_sari
import textstat

MODELS = {
    "openbio": {
        "hf_path": "aaditya/Llama3-OpenBioLLM-8B",
        "format": "chatml",
    },
    "biomistral": {
        "hf_path": "BioMistral/BioMistral-7B-DARE",
        "format": "mistral",
    },
    "mistral": {
        "hf_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "format": "mistral",
    },
}

TASK_INSTRUCTION = """Simplify the following medical discharge summary in plain language for patients with no medical background.
Guidelines:
- Replace medical jargon with everyday words (e.g., "hypertension" → "high blood pressure")
- Keep all important information (diagnoses, medications, follow-up instructions)
- Use short, clear sentences (aim for 15-20 words per sentence)
- Aim for a 6th-grade reading level
- Maintain the same structure as the original
- Do not add or omit information
- Keep the same patient reference style
- Output plain text only (no markdown, no bold, no headers, no bullet points)
- Do not include empty lines or separator characters like ---"""

SYSTEM_MESSAGE = "You are a helpful medical assistant that simplifies complex medical text for patients."

CHATML_INFERENCE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n"
MISTRAL_INFERENCE = "[INST] <<SYS>>\n{system}\n<</SYS>>\n{instruction}\n\n{input} [/INST]"


def build_prompt(sample, model_format):
    template = CHATML_INFERENCE if model_format == "chatml" else MISTRAL_INFERENCE
    return template.format(
        system=SYSTEM_MESSAGE,
        instruction=TASK_INSTRUCTION,
        input=sample["input"],
    )


def load_model(hf_path, adapter_path):
    print(f"Loading tokenizer: {hf_path}")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model (no quantization)...")
    base = AutoModelForCausalLM.from_pretrained(
        hf_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base

    model.eval()
    print("Model ready!")
    return model, tokenizer


def generate_predictions(model, tokenizer, dataset, model_format, batch_size=4):
    predictions = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        prompts = [build_prompt(sample, model_format) for sample in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        print(f"Generating sample {i+1}/{len(dataset)}...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # With left-padding, all rows in the batch share the same padded length.
        # Use shape[1] (the shared dimension) rather than per-row shape[0] to make
        # the slice robust regardless of how the batch loop is structured.
        prompt_len = inputs["input_ids"].shape[1]
        for output in outputs:
            generated = output[prompt_len:]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            # Sanity-check: decoded generation must not start with the prompt text
            assert not pred.startswith(prompts[0][:30]), (
                "Generation slice leaked prompt text — check padding_side='left'"
            )
            predictions.append(pred)
        if (i + batch_size) % 100 == 0 or i == 0:
            print(f"  Generated {min(i+batch_size, len(dataset))}/{len(dataset)}")
    return predictions


def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(scores))


def compute_bertscore(predictions, references):
    from bert_score import score as bert_score
    P, R, F1 = bert_score(
        predictions,
        references,
        lang="en",
        model_type="roberta-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    return float(F1.mean())


def compute_sari(sources, predictions, references):
    return corpus_sari(
        orig_sents=sources,
        sys_sents=predictions,
        refs_sents=[references],
    )


def compute_fk_grade(texts):
    scores = []
    for t in texts:
        try:
            scores.append(textstat.flesch_kincaid_grade(t))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           default="openbio")
    parser.add_argument("--adapter-path",    default="/mnt/adapters/full_training")
    parser.add_argument("--adapter-hf-repo", default=None,
                        help="HuggingFace repo ID for adapter (skips bucket mount)")
    parser.add_argument("--split",           default="test")
    parser.add_argument("--output-dir",      default="/output/eval")
    parser.add_argument("--zero-shot",       action="store_true")
    parser.add_argument("--fast",            action="store_true",
                        help="Skip BERTScore and SARI")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────
    print("Loading dataset...")
    from datasets import load_dataset
    ds = load_dataset("GuyDor007/medisimplifier-dataset", split=args.split)
    sources     = [ex["input"]    for ex in ds]
    references  = [ex["output"]   for ex in ds]

    # ── Load model ─────────────────────────────────────────────
    print("Loading model...")
    if args.zero_shot:
        adapter_path = None
    elif args.adapter_hf_repo:
        adapter_path = args.adapter_hf_repo
    else:
        adapter_path = args.adapter_path
    model, tokenizer = load_model(
        MODELS[args.model]["hf_path"],
        adapter_path=adapter_path,
    )

    # ── Generate predictions ───────────────────────────────────
    print(f"Generating on {len(sources)} samples...")
    dataset_for_gen = [{"input": s, "output": r}
                       for s, r in zip(sources, references)]
    predictions = generate_predictions(
        model, tokenizer,
        dataset_for_gen,
        MODELS[args.model]["format"],
        batch_size=4,
    )

    # ── Metrics ────────────────────────────────────────────────
    print("Computing ROUGE-L...")
    rouge = compute_rouge_l(predictions, references)

    print("Computing FK-Grade...")
    fk = compute_fk_grade(predictions)

    if args.fast:
        bertscore_val, sari_val = 0.0, 0.0
        print("Skipping BERTScore + SARI (--fast mode)")
    else:
        print("Computing BERTScore...")
        bertscore_val = compute_bertscore(predictions, references)
        print("Computing SARI...")
        sari_val = compute_sari(sources, predictions, references)

    # ── Save results ───────────────────────────────────────────
    results = {
        "model":      args.model,
        "split":      args.split,
        "n_samples":  len(predictions),
        "zero_shot":  args.zero_shot,
        "rouge_l":    rouge,
        "bertscore":  bertscore_val,
        "sari":       sari_val,
        "fk_grade":   fk,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n── Results ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {out / 'results.json'}")


if __name__ == "__main__":
    main()
