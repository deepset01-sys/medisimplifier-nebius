"""
Save adapter from last checkpoint — run after training if final save failed.
"""
import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True,
                       help="Path to training output dir (contains checkpoint-* folders)")
    parser.add_argument("--output-path", required=True,
                       help="Where to save the final adapter")
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()

    # Find last checkpoint
    checkpoint_dirs = sorted(
        Path(args.checkpoint_path).glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-")[1])
    )
    if not checkpoint_dirs:
        print(f"No checkpoints found in {args.checkpoint_path}")
        return

    last_checkpoint = checkpoint_dirs[-1]
    print(f"Loading from: {last_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True)

    model = PeftModel.from_pretrained(base, str(last_checkpoint))
    model.eval()

    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Saving adapter to: {output}")
    model.save_pretrained(str(output), safe_serialization=False)
    tokenizer.save_pretrained(str(output))
    print("Done!")

if __name__ == "__main__":
    main()
