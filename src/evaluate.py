import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
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
- Replace medical jargon with everyday words (e.g., "hypertension" -> "high blood pressure")
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
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        hf_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if adapter_path:
        print(f"  Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    return model, tokenizer


def generate_predictions(model, tokenizer, dataset, model_format, batch_size=1):
    predictions = []
    for i, sample in enumerate(dataset):
        prompt = build_prompt(sample, model_format)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append(pred)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{len(dataset)}")
    return predictions


def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(scores))


def compute_fk_grade(texts):
    scores = [textstat.flesch_kincaid_grade(t) for t in texts if t.strip()]
    return float(np.mean(scores))


def compute_bertscore(predictions, references):
    try:
        from bert_score import score as bs_score
        P, R, F = bs_score(predictions, references, lang="en", verbose=False)
        return float(F.mean())
    except Exception as e:
        print(f"  BERTScore failed: {e}")
        return 0.0


def compute_sari(sources, predictions, references):
    try:
        from easse.sari import corpus_sari
        return float(corpus_sari(
            orig_sents=sources,
            sys_sents=predictions,
            refs_sents=[references],
        ))
    except Exception as e:
        print(f"  SARI failed: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="MediSimplifier Evaluation")
    parser.add_argument("--model", default="openbio", choices=list(MODELS.keys()))
    parser.add_argument("--adapter-path", default="GuyDor007/MediSimplifier-LoRA-Adapters")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--output-dir", default="/output/eval")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Evaluate without adapter (zero-shot baseline)")
    args = parser.parse_args()

    model_info = MODELS[args.model]
    adapter = None if args.zero_shot else args.adapter_path

    print(f"\n{'='*60}")
    print(f"MediSimplifier Evaluation on Nebius")
    print(f"  Model:   {model_info['hf_path']}")
    print(f"  Adapter: {adapter or 'None (zero-shot)'}")
    print(f"  Split:   {args.split}")
    print(f"{'='*60}\n")

    dataset = load_dataset("GuyDor007/medisimplifier-dataset")[args.split]
    print(f"Loaded {len(dataset)} samples from {args.split} split")

    model, tokenizer = load_model(model_info["hf_path"], adapter)

    print("Generating predictions...")
    predictions = generate_predictions(
        model, tokenizer, dataset, model_info["format"]
    )

    references = [s["output"] for s in dataset]
    sources = [s["input"] for s in dataset]

    print("Computing metrics...")
    rouge_l = compute_rouge_l(predictions, references)
    fk_grade = compute_fk_grade(predictions)
    bertscore = compute_bertscore(predictions, references)
    sari = compute_sari(sources, predictions, references)

    results = {
        "model": args.model,
        "adapter": adapter,
        "split": args.split,
        "n_samples": len(dataset),
        "metrics": {
            "ROUGE-L": round(rouge_l, 4),
            "SARI": round(sari, 2),
            "BERTScore-F1": round(bertscore, 4),
            "FK-Grade": round(fk_grade, 2),
        },
    }

    print(f"\n{'='*40}")
    print(f"Results — {args.model} ({'zero-shot' if args.zero_shot else 'fine-tuned'})")
    print(f"  ROUGE-L:     {rouge_l:.4f}")
    print(f"  SARI:        {sari:.2f}")
    print(f"  BERTScore:   {bertscore:.4f}")
    print(f"  FK-Grade:    {fk_grade:.2f}")
    print(f"{'='*40}\n")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{args.model}_{'zeroshot' if args.zero_shot else 'finetuned'}_{args.split}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
