"""
MediSimplifier — LoRA Training Script for Nebius Serverless AI Jobs
Trains OpenBioLLM-8B with optimal LoRA configuration discovered through
ablation studies (r=32, all_attn, rsLoRA=True).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
import wandb

# ── Configuration ────────────────────────────────────────────────────────────

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
- Keep the same patient reference style (e.g., "The patient" stays "The patient", not "You")
- Output plain text only (no markdown, no bold, no headers, no bullet points)
- Do not include empty lines or separator characters like "---" """

SYSTEM_MESSAGE = "You are a helpful medical assistant that simplifies complex medical text for patients."

CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}

{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

MISTRAL_TEMPLATE = """[INST] <<SYS>>
{system}
<</SYS>>
{instruction}

{input} [/INST] {output}"""

LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "use_rslora": True,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

TRAINING_CONFIG = {
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "bf16": True,
    "max_seq_length": 2048,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_sample(sample: dict, model_format: str) -> str:
    """Format a dataset sample into the model-specific prompt."""
    template = CHATML_TEMPLATE if model_format == "chatml" else MISTRAL_TEMPLATE
    return template.format(
        system=SYSTEM_MESSAGE,
        instruction=TASK_INSTRUCTION,
        input=sample["input"],
        output=sample["output"],
    )

def load_and_format_dataset(model_format: str):
    """Load dataset from HuggingFace and apply prompt formatting."""
    print("Loading dataset: GuyDor007/medisimplifier-dataset")
    dataset = load_dataset("GuyDor007/medisimplifier-dataset")

    def format_fn(sample):
        return {"text": format_sample(sample, model_format)}

    dataset = dataset.map(format_fn, remove_columns=["instruction", "input", "output"])
    print(f"  Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")
    return dataset

def build_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load base model with optional 4-bit quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if use_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right-padding for SFT: loss mask aligns with tokens; evaluate.py uses "left" for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def apply_lora(model):
    """Apply LoRA adapters with optimal configuration."""
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    return model

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MediSimplifier LoRA Training")
    parser.add_argument("--model", default="openbio", choices=list(MODELS.keys()))
    parser.add_argument("--output-dir", default="/output/adapter")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"])
    parser.add_argument("--rank", type=int, default=LORA_CONFIG["r"],
                        help="LoRA rank (ablation: 8/16/32)")
    parser.add_argument("--modules", default="all_attn",
                        choices=["q_only", "q_v", "all_attn"],
                        help="Target modules (ablation)")
    parser.add_argument("--data-size", type=int, default=7999,
                        help="Training samples (ablation: 2000/4000/7999)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "medisimplifier"),
        name=f"{args.model}-r{args.rank}-{args.modules}-{args.data_size}",
        config={
            "model": args.model,
            "rank": args.rank,
            "modules": args.modules,
            "data_size": args.data_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "learning_rate": 2e-4,
            "batch_size": 4,
            "grad_accumulation": 4,
            "rslora": True,
            "lora_alpha": 64,
        },
        tags=["lora", "medical", "nebius", "h100"],
    )

    import random
    import numpy as np
    from transformers import set_seed as transformers_set_seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    transformers_set_seed(args.seed)

    # Resolve module configs for ablation
    MODULE_MAP = {
        "q_only":   ["q_proj"],
        "q_v":      ["q_proj", "v_proj"],
        "all_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    LORA_CONFIG["r"] = args.rank
    LORA_CONFIG["lora_alpha"] = args.rank * 2
    LORA_CONFIG["target_modules"] = MODULE_MAP[args.modules]

    model_info = MODELS[args.model]
    print(f"\n{'='*60}")
    print(f"MediSimplifier Training — {args.model}")
    print(f"  Model:   {model_info['hf_path']}")
    print(f"  LoRA:    r={args.rank}, modules={args.modules}, rsLoRA=True")
    print(f"  Epochs:  {args.epochs} | Data: {args.data_size} samples")
    print(f"  Output:  {args.output_dir}")
    print(f"{'='*60}\n")

    dataset = load_and_format_dataset(model_info["format"])

    # Apply data size limit for ablation runs
    if args.data_size < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(args.data_size))
        print(f"  Ablation mode: using {args.data_size} training samples")

    model, tokenizer = build_model_and_tokenizer(model_info["hf_path"])
    model = apply_lora(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        bf16=TRAINING_CONFIG["bf16"],
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        save_safetensors=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=TRAINING_CONFIG.get("max_seq_length", 2048),
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    wandb.finish()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\nAdapter saved to: {output_path}")

    # Save run metadata for reproducibility
    metadata = {
        "model": args.model,
        "hf_path": model_info["hf_path"],
        "lora_rank": args.rank,
        "lora_alpha": args.rank * 2,
        "target_modules": MODULE_MAP[args.modules],
        "use_rslora": True,
        "epochs": args.epochs,
        "data_size": args.data_size,
        "dataset": "GuyDor007/medisimplifier-dataset",
    }
    with open(output_path / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved.")

if __name__ == "__main__":
    main()
