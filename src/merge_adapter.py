import os
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODELS = {
    "openbio": "aaditya/Llama3-OpenBioLLM-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "biomistral": "BioMistral/BioMistral-7B-DARE",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openbio")
    parser.add_argument("--adapter-path", default="/mnt/adapters/full_training")
    parser.add_argument("--output-path", default="/output/merged_openbio")
    args = parser.parse_args()

    hf_path = MODELS[args.model]
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {hf_path}")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    print(f"Loading base model: {hf_path}")
    base = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_path}")
    merged.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print("Done! Merged model saved.")

if __name__ == "__main__":
    main()
