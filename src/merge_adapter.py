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
    parser.add_argument("--output-path", default="/tmp/merged_openbio")
    parser.add_argument("--bucket", default="medisimplifier-adapters")
    parser.add_argument("--bucket-key", default="merged_openbio")
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
    import os
    print(f"Contents of {args.adapter_path}:")
    try:
        files = os.listdir(args.adapter_path)
        print(f"Files: {files}")
    except Exception as e:
        print(f"Error listing directory: {e}")

    print(f"Contents of /mnt/adapters:")
    try:
        files = os.listdir('/mnt/adapters')
        print(f"Files: {files}")
    except Exception as e:
        print(f"Error listing /mnt/adapters: {e}")

    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to local disk: {args.output_path}")
    merged.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)

    import os
    files = os.listdir(args.output_path)
    print(f"Files saved locally: {files}")

    print(f"Uploading to bucket: {args.bucket}/{args.bucket_key}")
    import subprocess as sp

    # Configure AWS CLI for Nebius
    sp.run(['pip', 'install', 'awscli', '--quiet'], check=True)

    key_id = os.getenv('AWS_ACCESS_KEY_ID')
    secret = os.getenv('AWS_SECRET_ACCESS_KEY')

    env_vars = os.environ.copy()
    env_vars['AWS_ACCESS_KEY_ID'] = key_id
    env_vars['AWS_SECRET_ACCESS_KEY'] = secret
    env_vars['AWS_DEFAULT_REGION'] = 'eu-north1'

    for f in files:
        local_path = os.path.join(args.output_path, f)
        s3_path = f's3://{args.bucket}/{args.bucket_key}/{f}'
        print(f"Uploading: {f} → {s3_path}")
        result = sp.run([
            'aws', 's3', 'cp', local_path, s3_path,
            '--endpoint-url', 'https://storage.eu-north1.nebius.cloud',
        ], env=env_vars, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Uploaded: {f}")
        else:
            print(f"Error: {result.stderr}")
            raise Exception(f"Failed to upload {f}: {result.stderr}")

    print("Done! All files uploaded.")

if __name__ == "__main__":
    main()
