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

    import subprocess as sp
    sp.run(['pip', 'install', 'boto3', '--quiet'], check=True)
    print("boto3 installed.")
    key_id = os.getenv('AWS_ACCESS_KEY_ID', 'NOT SET')
    secret = os.getenv('AWS_SECRET_ACCESS_KEY', 'NOT SET')
    print(f"AWS_ACCESS_KEY_ID: {key_id[:10]}..." if key_id != 'NOT SET' else "AWS_ACCESS_KEY_ID: NOT SET")
    print(f"AWS_SECRET_ACCESS_KEY: {'SET (length=' + str(len(secret)) + ')' if secret != 'NOT SET' else 'NOT SET'}")
    print(f"Uploading to bucket: {args.bucket}/{args.bucket_key}")
    import boto3
    from botocore.config import Config

    import os
    s3 = boto3.client(
        's3',
        endpoint_url='https://storage.eu-north1.nebius.cloud:443',
        region_name='eu-north1',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        verify=True,
    )

    for f in files:
        local_path = os.path.join(args.output_path, f)
        key = f'{args.bucket_key}/{f}'
        print(f"Uploading: {f} → s3://{args.bucket}/{key}")
        s3.upload_file(local_path, args.bucket, key)
        print(f"Uploaded: {f}")

    print("Done! Merged model uploaded to bucket.")

if __name__ == "__main__":
    main()
