import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer
import textstat
from typing import Optional

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
    "merged": {
        "hf_path": "chambul/MediSimplifier-OpenBioLLM-merged",
        "format": "chatml",
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


def build_prompt(sample: dict, model_format: str, tokenizer=None, use_native_template: bool = False) -> str:
    if use_native_template and tokenizer is not None:
        # Llama-3 native format for OpenBioLLM-8B (aaditya/Llama3-OpenBioLLM-8B)
        # Mistral native format for Mistral-7B-Instruct-v0.2 and BioMistral-7B-DARE
        if model_format == "chatml":
            return (
                f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_MESSAGE}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{TASK_INSTRUCTION}\n\n{sample['input']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            return f"[INST] {SYSTEM_MESSAGE}\n\n{TASK_INSTRUCTION}\n\n{sample['input']} [/INST]"
    template = CHATML_INFERENCE if model_format == "chatml" else MISTRAL_INFERENCE
    return template.format(
        system=SYSTEM_MESSAGE,
        instruction=TASK_INSTRUCTION,
        input=sample["input"],
    )


def load_model(hf_path: str, adapter_path: Optional[str]):
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


def generate_predictions(model, tokenizer, dataset: list, model_format: str, batch_size: int = 4, use_native_template: bool = False) -> tuple[list[str], dict]:
    predictions = []
    n_truncated = 0
    n_prompt_leaks = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        prompts = [build_prompt(sample, model_format, tokenizer=tokenizer, use_native_template=use_native_template) for sample in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, max_length=None)
        n_truncated += sum(1 for ids in encoded["input_ids"] if len(ids) > 2048)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        print(f"Generating samples {i+1}-{min(i+batch_size, len(dataset))}/{len(dataset)}...")
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM on batch {i}–{i+batch_size}, retrying sample-by-sample...")
                torch.cuda.empty_cache()
                outputs = []
                for idx, single_input in enumerate(inputs["input_ids"]):
                    single = {
                        "input_ids": single_input.unsqueeze(0),
                        "attention_mask": inputs["attention_mask"][idx:idx+1],
                    }
                    with torch.no_grad():
                        out = model.generate(
                            **single,
                            max_new_tokens=512,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    outputs.append(out[0])
            else:
                raise
        # With left-padding, all rows in the batch share the same padded length.
        # Use shape[1] (the shared dimension) rather than per-row shape[0] to make
        # the slice robust regardless of how the batch loop is structured.
        prompt_len = inputs["input_ids"].shape[1]
        for j, output in enumerate(outputs):
            generated = output[prompt_len:]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            # Sanity-check: each decoded generation must not start with its own prompt text
            if pred.startswith(prompts[j][:30]):
                print(f"  Warning: generation slice leaked prompt for batch {i}, item {j} — appending empty prediction")
                n_prompt_leaks += 1
                predictions.append("")
                continue
            predictions.append(pred)
        if (i + batch_size) % 100 == 0 or i == 0:
            print(f"  Generated {min(i+batch_size, len(dataset))}/{len(dataset)}")
    return predictions, {"n_truncated": n_truncated, "n_prompt_leaks": n_prompt_leaks}


def compute_rouge_l(predictions: list[str], references: list[str]) -> tuple[float, list[float]]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(scores)), [float(s) for s in scores]


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
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


def compute_sari(sources: list[str], predictions: list[str], references: list[str]) -> float:
    from easse.sari import corpus_sari  # lazy import — easse is git-based and only needed for SARI
    return corpus_sari(
        orig_sents=sources,
        sys_sents=predictions,
        refs_sents=[references],
    )


def compute_fk_grade(texts: list[str]) -> float:
    scores = []
    for t in texts:
        try:
            scores.append(textstat.flesch_kincaid_grade(t))
        except Exception as e:
            print(f"Warning: FK-Grade failed for a sample: {e}")
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
    parser.add_argument("--native-template", action="store_true",
                        help="Use each model's native chat template for zero-shot baseline (do not use with adapter)")
    parser.add_argument("--batch-size",      type=int, default=4,
                        help="Batch size for generation (default: 4)")
    parser.add_argument("--limit",           type=int, default=None,
                        help="Limit evaluation to first N samples (smoke test)")
    parser.add_argument("--fast",            action="store_true",
                        help="Skip BERTScore and SARI")
    args = parser.parse_args()

    # Guard: --native-template without --zero-shot means adapter + wrong template = silent wrong results
    if args.native_template and not args.zero_shot:
        parser.error("--native-template requires --zero-shot (native templates are for zero-shot baselines only). Fine-tuned models must use the training-time template.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────
    print("Loading dataset...")
    ds = load_dataset("GuyDor007/medisimplifier-dataset", split=args.split)
    sources     = [ex["input"]    for ex in ds]
    references  = [ex["output"]   for ex in ds]
    if args.limit:
        sources     = sources[:args.limit]
        references  = references[:args.limit]
        print(f"Smoke mode: limiting to {args.limit} samples")

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
    predictions, gen_stats = generate_predictions(
        model, tokenizer,
        dataset_for_gen,
        MODELS[args.model]["format"],
        batch_size=args.batch_size,
        use_native_template=args.native_template,
    )
    if gen_stats["n_truncated"] > 0:
        print(f"  Warning: {gen_stats['n_truncated']} prompts were truncated at max_length=2048")
    if gen_stats["n_prompt_leaks"] > 0:
        print(f"  Warning: {gen_stats['n_prompt_leaks']} prompt leaks detected (empty predictions)")

    # ── Metrics ────────────────────────────────────────────────
    print("Computing ROUGE-L...")
    rouge, rouge_per_sample = compute_rouge_l(predictions, references)

    print("Computing FK-Grade...")
    fk = compute_fk_grade(predictions)

    if args.fast:
        bertscore_val, sari_val = None, None
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
        "rouge_l_per_sample": rouge_per_sample,
        "bertscore":  bertscore_val,
        "sari":       sari_val,
        "fk_grade":   fk,
        "n_truncated_prompts": gen_stats["n_truncated"],
        "n_prompt_leaks":      gen_stats["n_prompt_leaks"],
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n── Results ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {out / 'results.json'}")


if __name__ == "__main__":
    main()
