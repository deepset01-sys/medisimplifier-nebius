"""
Unit tests for MediSimplifier metric functions.
Run with: python -m pytest tests/ -v

Note: requires the training Docker image (all dependencies pre-installed):
  docker run --rm chambul/medisimplifier:train-v20 python -m pytest tests/ -v
Tests will fail on a bare environment without torch, transformers, peft, etc.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate import compute_rouge_l, compute_fk_grade, build_prompt


# ── compute_rouge_l ──────────────────────────────────────────────────

def test_rouge_l_identical():
    """Identical prediction and reference → ROUGE-L = 1.0"""
    texts = ["the patient had a heart attack"]
    mean, per_sample = compute_rouge_l(texts, texts)
    assert mean == pytest.approx(1.0, abs=1e-4)
    assert len(per_sample) == 1
    assert per_sample[0] == pytest.approx(1.0, abs=1e-4)

def test_rouge_l_empty_prediction():
    """Empty prediction → ROUGE-L = 0.0"""
    mean, per_sample = compute_rouge_l([""], ["the patient had a heart attack"])
    assert mean == pytest.approx(0.0, abs=1e-4)

def test_rouge_l_returns_per_sample():
    """Returns per-sample list of same length as inputs"""
    preds = ["heart attack", "high blood pressure", "diabetes"]
    refs  = ["myocardial infarction", "hypertension", "diabetes mellitus"]
    mean, per_sample = compute_rouge_l(preds, refs)
    assert len(per_sample) == 3
    assert all(0.0 <= s <= 1.0 for s in per_sample)
    assert mean == pytest.approx(sum(per_sample) / len(per_sample), abs=1e-4)


# ── compute_fk_grade ─────────────────────────────────────────────────

def test_fk_grade_simple_text():
    """Simple short text → low FK grade"""
    grade = compute_fk_grade(["The cat sat on the mat."])
    assert isinstance(grade, float)
    assert grade < 5.0

def test_fk_grade_complex_text():
    """Complex medical text → higher FK grade than simple text"""
    simple  = compute_fk_grade(["The cat sat on the mat."])
    complex_ = compute_fk_grade([
        "The patient presented with acute myocardial infarction "
        "requiring immediate percutaneous coronary intervention."
    ])
    assert complex_ > simple

def test_fk_grade_empty_list():
    """Empty list → returns 0.0 without crashing"""
    assert compute_fk_grade([]) == 0.0


# ── build_prompt ─────────────────────────────────────────────────────

def test_build_prompt_chatml_contains_input():
    """ChatML prompt contains the sample input text"""
    sample = {"input": "Patient had hypertension."}
    prompt = build_prompt(sample, "chatml")
    assert "Patient had hypertension." in prompt

def test_build_prompt_mistral_contains_input():
    """Mistral prompt contains the sample input text"""
    sample = {"input": "Patient had hypertension."}
    prompt = build_prompt(sample, "mistral")
    assert "Patient had hypertension." in prompt

def test_build_prompt_formats_differ():
    """ChatML and Mistral prompts have different format"""
    sample = {"input": "Patient had hypertension."}
    chatml  = build_prompt(sample, "chatml")
    mistral = build_prompt(sample, "mistral")
    assert chatml != mistral
