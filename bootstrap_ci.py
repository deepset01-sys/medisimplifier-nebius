"""
bootstrap_ci.py — Paired bootstrap significance test for MediSimplifier
Reads per-sample ROUGE-L from three eval JSON files and computes:
  1. 95% CI for each model (n=10,000 bootstrap samples)
  2. Paired bootstrap on per-sample differences between models
  3. One-sided p-values: P(model_A > model_B)

Usage:
    python bootstrap_ci.py

Output: prints CI table + pairwise significance results
"""

import json
import numpy as np
from pathlib import Path

EVIDENCE_DIR = Path("results/nebius_evidence")
MODELS = {
    "OpenBioLLM-8B": "bootstrap_input.json",
    "Mistral-7B":    "eval_persamples_mistral.json",
    "BioMistral-7B": "eval_persamples_biomistral.json",
}
N_BOOTSTRAP = 10_000
SEED = 42


def load_per_sample(fname: str) -> np.ndarray:
    path = EVIDENCE_DIR / fname
    with open(path) as f:
        data = json.load(f)
    scores = data["rouge_l_per_sample"]
    return np.array(scores, dtype=float)


def bootstrap_ci(scores: np.ndarray, n: int, rng: np.random.Generator):
    """95% CI via percentile bootstrap."""
    means = np.array([
        rng.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n)
    ])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi


def paired_bootstrap_pvalue(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator):
    """One-sided p-value: P(mean(A) > mean(B)) via paired bootstrap on differences."""
    diffs = a - b
    obs_diff = diffs.mean()
    # Shift diffs to null hypothesis (mean=0)
    shifted = diffs - obs_diff
    null_means = np.array([
        rng.choice(shifted, size=len(shifted), replace=True).mean()
        for _ in range(n)
    ])
    p_value = (null_means >= obs_diff).mean()
    return obs_diff, p_value


def main():
    rng = np.random.default_rng(SEED)
    scores = {name: load_per_sample(fname) for name, fname in MODELS.items()}

    print("=" * 60)
    print("95% Confidence Intervals (percentile bootstrap, n=10,000)")
    print("=" * 60)
    print(f"{'Model':<20} {'Mean':>8} {'CI Lower':>10} {'CI Upper':>10}")
    print("-" * 60)
    cis = {}
    for name, arr in scores.items():
        lo, hi = bootstrap_ci(arr, N_BOOTSTRAP, rng)
        cis[name] = (arr.mean(), lo, hi)
        print(f"{name:<20} {arr.mean():>8.4f} {lo:>10.4f} {hi:>10.4f}")

    print()
    print("=" * 60)
    print("Paired Bootstrap Significance (one-sided, n=10,000)")
    print("=" * 60)
    model_names = list(scores.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a_name, b_name = model_names[i], model_names[j]
            obs_diff, p = paired_bootstrap_pvalue(
                scores[a_name], scores[b_name], N_BOOTSTRAP, rng
            )
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{a_name} > {b_name}: diff={obs_diff:+.4f}, p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
