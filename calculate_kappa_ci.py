"""Bootstrapped confidence intervals for Cohen's kappa on the safety eval results.

For each of v2 and v3:
  - extract (llama_verdict, qwen_verdict) pairs from `per_sample`
  - drop rows where either verdict is not a valid label (SAFE / UNSAFE) --
    a handful of rows carry "ERROR"; excluding them reproduces the kappa
    values already stored in each file (v2=0.1114, v3=0.0431)
  - compute Cohen's kappa and a 95% percentile bootstrap CI

Then a paired bootstrap test for the difference in kappa between v2 and v3
(the two versions score the SAME underlying samples, aligned by `idx`, so the
difference test resamples the shared samples jointly).

Reproducibility: n_bootstrap=10000, random_seed=42.
Dependencies: numpy only.
"""

import json
import os
import sys

import numpy as np

# UTF-8 stdout so the kappa symbol prints on legacy Windows codepages.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "v2": os.path.join(BASE_DIR, "results", "nebius_evidence", "safety_results_v2.json"),
    "v3": os.path.join(BASE_DIR, "results", "nebius_evidence", "safety_results_v3.json"),
}

VALID_LABELS = {"SAFE", "UNSAFE"}
LABEL_TO_INT = {"SAFE": 0, "UNSAFE": 1}

N_BOOTSTRAP = 10000
RANDOM_SEED = 42


def load_pairs(path):
    """Return {idx: (llama_int, qwen_int)} for rows with two valid verdicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = {}
    for s in data["per_sample"]:
        lv, qv = s.get("llama_verdict"), s.get("qwen_verdict")
        if lv in VALID_LABELS and qv in VALID_LABELS:
            pairs[s["idx"]] = (LABEL_TO_INT[lv], LABEL_TO_INT[qv])
    return pairs


def cohen_kappa(a, b):
    """Cohen's kappa for two binary (0/1) rating arrays.

    Returns np.nan when it is undefined (agreement by chance == 1, i.e. one
    rater used a single category in this sample).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = a.size
    po = np.mean(a == b)
    pa1, pb1 = np.mean(a), np.mean(b)
    pe = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    denom = 1.0 - pe
    if denom == 0:
        return np.nan
    return (po - pe) / denom


def bootstrap_ci(a, b, rng, n_boot=N_BOOTSTRAP):
    """95% percentile bootstrap CI for kappa on paired arrays a, b."""
    n = a.size
    reps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)          # resample pairs with replacement
        reps[i] = cohen_kappa(a[idx], b[idx])
    reps = reps[np.isfinite(reps)]                # drop undefined replicates
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return lo, hi


def bootstrap_diff_pvalue(a2, b2, a3, b3, rng, n_boot=N_BOOTSTRAP):
    """Paired bootstrap two-sided p-value for (kappa_v2 - kappa_v3).

    a2/b2 and a3/b3 are aligned to the SAME shared samples (same order), so
    each replicate resamples one index vector and applies it to both versions.
    """
    n = a2.size
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = cohen_kappa(a2[idx], b2[idx]) - cohen_kappa(a3[idx], b3[idx])
    diffs = diffs[np.isfinite(diffs)]
    # two-sided p: how often the bootstrap difference falls on/across zero
    p = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return min(p, 1.0)


def main():
    for v, path in FILES.items():
        if not os.path.exists(path):
            sys.exit(f"Input file not found: {path}")

    rng = np.random.default_rng(RANDOM_SEED)

    pairs = {v: load_pairs(path) for v, path in FILES.items()}

    results = {}
    for v in ("v2", "v3"):
        a = np.array([lv for lv, qv in pairs[v].values()])
        b = np.array([qv for lv, qv in pairs[v].values()])
        k = cohen_kappa(a, b)
        lo, hi = bootstrap_ci(a, b, rng)
        results[v] = (k, lo, hi)

    # Difference test on the shared samples (idx present & valid in BOTH versions)
    common = sorted(set(pairs["v2"]) & set(pairs["v3"]))
    a2 = np.array([pairs["v2"][i][0] for i in common])
    b2 = np.array([pairs["v2"][i][1] for i in common])
    a3 = np.array([pairs["v3"][i][0] for i in common])
    b3 = np.array([pairs["v3"][i][1] for i in common])
    delta = results["v2"][0] - results["v3"][0]
    p_value = bootstrap_diff_pvalue(a2, b2, a3, b3, rng)

    k2, lo2, hi2 = results["v2"]
    k3, lo3, hi3 = results["v3"]
    print(f"v2: κ={k2:.4f} [95% CI: {lo2:.4f}–{hi2:.4f}]")
    print(f"v3: κ={k3:.4f} [95% CI: {lo3:.4f}–{hi3:.4f}]")
    p_str = "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"
    print(f"Δκ = {delta:.4f}, p={p_str}")


if __name__ == "__main__":
    main()
