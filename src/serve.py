import sys
import types

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
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI(title="MediSimplifier API", version="1.0")

ADAPTER_REPO = "/mnt/adapters/full_training"
BASE_MODEL = "aaditya/Llama3-OpenBioLLM-8B"

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

INFERENCE_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input}<|im_end|>\n<|im_start|>assistant\n"

model = None
tokenizer = None


class SimplifyRequest(BaseModel):
    text: str
    max_new_tokens: int = 1024
    temperature: float = 0.1


class SimplifyResponse(BaseModel):
    simplified: str
    model: str = BASE_MODEL
    adapter: str = ADAPTER_REPO


@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"Loading LoRA adapter: {ADAPTER_REPO}")
    model = PeftModel.from_pretrained(base, ADAPTER_REPO)
    model.eval()
    print("Model ready.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": BASE_MODEL,
        "adapter": ADAPTER_REPO,
        "max_input_chars": 8000,
        "max_new_tokens": 512,
        "batching": "single-request (one inference per call)",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/simplify", response_model=SimplifyResponse)
async def simplify(request: SimplifyRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="text field is required and cannot be empty")
    if len(request.text) > 8000:
        raise HTTPException(status_code=400, detail="text too long — maximum 8000 characters")

    prompt = INFERENCE_TEMPLATE.format(
        system=SYSTEM_MESSAGE,
        instruction=TASK_INSTRUCTION,
        input=request.text,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=503, detail="GPU out of memory — try shorter input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    simplified = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return SimplifyResponse(simplified=simplified)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
