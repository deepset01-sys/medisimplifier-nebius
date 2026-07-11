"""
safe_endpoint.py — Safe Simplification Endpoint
FastAPI wrapper: POST /v1/simplify → vLLM (local) + dual-judge safety gate (Token Factory)

Deploy as a Nebius Job (see jobs/safe_endpoint.yaml):
  nebius ai job create -f jobs/safe_endpoint.yaml

Or locally with Docker:
  docker run --gpus all -p 8000:8000 \
    -e NEBIUS_API_KEY=<key> -e HF_TOKEN=<token> \
    chambul/medisimplifier:endpoint-v1
"""

import os
import time
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from safety_gate import evaluate_safety

app = FastAPI(title="MediSimplifier Safe Endpoint", version="1.0")

VLLM_URL   = "http://127.0.0.1:8001/v1/completions"
MODEL_NAME = "chambul/MediSimplifier-OpenBioLLM-merged"

SIMPLIFY_PROMPT = """Simplify the following medical discharge summary in plain language for patients with no medical background.

Guidelines:
- Replace medical jargon with everyday words
- Keep all important information
- Use short, clear sentences
- Aim for a 6th-grade reading level
- Do not add or omit information

Medical Text:
{text}

Simplified:"""


class SimplifyRequest(BaseModel):
    text: str
    max_tokens: int = 512
    safety_mode: str = "flag"   # "block" or "flag"


class SimplifyResponse(BaseModel):
    simplified_text: Optional[str]
    blocked: bool
    safety: dict
    latency_ms: dict


@app.get("/health")
async def health():
    """Check vLLM + Token Factory availability."""
    vllm_ok = False
    tf_ok = bool(os.environ.get("NEBIUS_API_KEY"))
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get("http://127.0.0.1:8001/health")
            vllm_ok = r.status_code == 200
    except Exception:
        pass
    return {"vllm": vllm_ok, "token_factory": tf_ok, "ready": vllm_ok and tf_ok}


@app.post("/v1/simplify", response_model=SimplifyResponse)
async def simplify(req: SimplifyRequest):
    """Simplify medical text and run dual-judge safety gate."""
    t0 = time.time()

    # ── Step 1: vLLM simplification ──────────────────────────────────
    prompt = SIMPLIFY_PROMPT.format(text=req.text)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            vllm_resp = await client.post(VLLM_URL, json={
                "model":       MODEL_NAME,
                "prompt":      prompt,
                "max_tokens":  req.max_tokens,
                "temperature": 0,
            })
            vllm_resp.raise_for_status()
            simplified = vllm_resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"vLLM error: {e}")

    t_vllm = (time.time() - t0) * 1000

    # ── Step 2: Dual-judge safety gate ───────────────────────────────
    safety = evaluate_safety(req.text, simplified, safety_mode=req.safety_mode)
    t_total = (time.time() - t0) * 1000

    return SimplifyResponse(
        simplified_text=None if safety["blocked"] else simplified,
        blocked=safety["blocked"],
        safety=safety,
        latency_ms={"vllm_ms": round(t_vllm), "total_ms": round(t_total)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
