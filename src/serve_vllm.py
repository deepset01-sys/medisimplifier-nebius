import sys
import types

# torchaudio stub
_torchaudio = types.ModuleType('torchaudio')
_torchaudio.__version__ = '0.0.0'
_torchaudio.__spec__ = types.SimpleNamespace(
    name='torchaudio', origin=None, submodule_search_locations=[]
)
sys.modules['torchaudio'] = _torchaudio

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/mnt/adapters/merged_openbio")
MODEL_NAME = "aaditya/Llama3-OpenBioLLM-8B"

TASK_INSTRUCTION = """Simplify the following medical discharge summary in plain language for patients with no medical background.
Guidelines:
- Replace medical jargon with everyday words
- Keep all important information
- Use short, clear sentences
- Aim for 6th-grade reading level
- Output plain text only"""

SYSTEM_MESSAGE = "You are a helpful medical assistant that simplifies complex medical text for patients."

app = FastAPI(title="MediSimplifier API", version="2.0.0")

tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    print(f"Loading tokenizer from: {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading merged model from: {ADAPTER_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        ADAPTER_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model ready.")


class SimplifyRequest(BaseModel):
    text: str


class SimplifyResponse(BaseModel):
    simplified: str
    model: str
    adapter: str


# OpenAI-compatible message format
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "medisimplifier"
    messages: list[Message]
    max_tokens: int = 512


class ChatResponse(BaseModel):
    id: str = "chatcmpl-medisimplifier"
    object: str = "chat.completion"
    model: str = "medisimplifier"
    choices: list[dict]


@app.post("/simplify", response_model=SimplifyResponse)
async def simplify(request: SimplifyRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="text field is required")
    if len(request.text) > 8000:
        raise HTTPException(status_code=400, detail="text too long — maximum 8000 characters")

    prompt = f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n{TASK_INSTRUCTION}\n\n{request.text}<|im_end|>\n<|im_start|>assistant\n"

    try:
        inputs = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return SimplifyResponse(
            simplified=decoded.strip(),
            model=ADAPTER_PATH,
            adapter="merged (LoRA fused into base)",
        )
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(status_code=503, detail="GPU out of memory")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible endpoint"""
    user_text = next(
        (m.content for m in request.messages if m.role == "user"), ""
    )
    result = await simplify(SimplifyRequest(text=user_text))
    return ChatResponse(
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": result.simplified},
            "finish_reason": "stop"
        }]
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": ADAPTER_PATH,
        "adapter": "merged LoRA",
        "max_input_chars": 8000,
        "max_new_tokens": 512,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
