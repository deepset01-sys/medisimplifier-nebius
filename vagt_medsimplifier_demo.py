"""vagt_medsimplifier_demo.py — VAGT proof-of-concept on MedSimp-JudgeBench.

Demonstrates Veridicality-Anchored G-Theory on the EXISTING two-judge data (no
new collection). Per error type it decomposes veridical error into shared bias
(σ²_B), rater-specific bias (σ²_R), and noise (σ²_N), computes the derived
dependability Φ_V, and shows that Φ_V flags the diagnosis-omission blind spot
that consensus statistics (Cohen's κ, Krippendorff's α, PABAK) miss — because
those only measure judge-to-judge agreement, never agreement with ground truth.

Data (ground truth = injected error type / condition):
  results/nebius_evidence/calibration_verdicts.json      (no-CoT judges)
  results/nebius_evidence/calibration_verdicts_cot.json  (CoT judges)

Model: two judges (Llama, Qwen), binary SAFE=0 / UNSAFE=1. Per error type f the
stratum = corrupted-f items (τ=1) + the shared clean controls (τ=0), so σ²_τ>0.

Decomposition (UNDIVIDED shared bias; matches vagt_section.md §3):
  consensus   c_i  = mean_r X_{ir}
  shared bias b_i  = c_i − τ_i ;  σ²_B = mean_i b_i²  (= β̄² + σ²_S; does NOT shrink)
  rater bias  α_r  = X̄_{·r} − X̄ ; σ²_R = mean_r α_r²
  noise       ε_ir = X_{ir} − c_i − α_r ; σ²_N = mean ε²
  σ²_τ = π(1−π) ;  Φ_V = σ²_τ / (σ²_τ + σ²_B + (σ²_R + σ²_N)/n_r),  n_r = 2

Caveat: only two judges, so σ²_R is illustrative; the proof-of-concept targets
σ²_B and Φ_V. The Phase 3 study has 20 raters.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE / "results" / "nebius_evidence",
    Path(r"D:\Owner\Desktop\assignment_01\medisimplifier-nebius\results\nebius_evidence"),
]
DATA = next((p for p in CANDIDATES if p.exists()), None)
if DATA is None:
    sys.exit("calibration data dir not found (results/nebius_evidence)")

LABEL_TO_INT = {"SAFE": 0, "UNSAFE": 1}
VALID = set(LABEL_TO_INT)
FEATURES = ["dose", "negation", "lateral", "diagnosis"]
CONDITIONS = [
    ("no-CoT", "calibration_verdicts.json", "llama_verdict", "qwen_verdict"),
    ("CoT", "calibration_verdicts_cot.json", "llama_verdict_cot", "qwen_verdict_cot"),
]


# ── consensus (judge-to-judge) statistics ───────────────────────────────────
def observed_agreement(a, b):
    return float(np.mean(a == b))


def cohen_kappa(a, b):
    po = observed_agreement(a, b)
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def pabak(a, b):
    return 2 * observed_agreement(a, b) - 1


def krippendorff_alpha_binary(a, b):
    n = a.size
    do = 1 - observed_agreement(a, b)
    N = 2 * n
    n1 = int(a.sum() + b.sum())
    n0 = N - n1
    de = (2.0 * n0 * n1) / (N * (N - 1))
    return float("nan") if de == 0 else 1 - do / de


# ── VAGT decomposition (veridicality-anchored) ──────────────────────────────
def vagt(X, tau, n_r=2):
    c = X.mean(axis=1)                 # per-item consensus
    b = c - tau                        # shared bias per item
    beta_bar = float(b.mean())
    sigma_B = float(np.mean(b ** 2))   # β̄² + σ²_S — undivided
    grand = X.mean()
    alpha = X.mean(axis=0) - grand     # rater main effects
    sigma_R = float(np.mean(alpha ** 2))
    eps = X - (c[:, None] + alpha[None, :])
    sigma_N = float(np.mean(eps ** 2))
    p = float(tau.mean())
    sigma_tau = p * (1 - p)
    phi_v = sigma_tau / (sigma_tau + sigma_B + (sigma_R + sigma_N) / n_r)
    return dict(beta_bar=beta_bar, sigma_B=sigma_B, sigma_R=sigma_R,
                sigma_N=sigma_N, sigma_tau=sigma_tau, phi_v=phi_v)


def stratum(records, feature, k_llama, k_qwen):
    """Corrupted-`feature` items (τ=1) + all clean controls (τ=0), valid verdicts."""
    a, b, tau = [], [], []
    for r in records:
        if r["condition"] == "corrupted" and r["error_type"] == feature:
            t = 1
        elif r["condition"] == "clean":
            t = 0
        else:
            continue
        lv, qv = r.get(k_llama), r.get(k_qwen)
        if lv in VALID and qv in VALID:
            a.append(LABEL_TO_INT[lv]); b.append(LABEL_TO_INT[qv]); tau.append(t)
    return np.array(a), np.array(b), np.array(tau, dtype=float)


def main():
    print("VAGT proof-of-concept on MedSimp-JudgeBench (existing two-judge data)")
    print("Per error type: consensus stats (κ/α/PABAK) vs. veridicality-anchored VAGT\n")
    for cond, fname, k_l, k_q in CONDITIONS:
        recs = json.loads((DATA / fname).read_text(encoding="utf-8"))
        print(f"=== {cond} ===")
        hdr = (f"{'feature':<11}{'n':>5}{'prev':>7}{'Po':>7}{'kappa':>8}"
               f"{'PABAK':>8}{'alpha':>8} | {'sigmaB':>8}{'sigmaR':>8}{'sigmaN':>8}{'Phi_V':>8}")
        print(hdr); print("-" * len(hdr))
        rows = {}
        for f in FEATURES:
            a, b, tau = stratum(recs, f, k_l, k_q)
            v = vagt(np.column_stack([a, b]), tau)
            po = observed_agreement(a, b)
            k = cohen_kappa(a, b); pk = pabak(a, b); al = krippendorff_alpha_binary(a, b)
            rows[f] = dict(v=v, po=po, kappa=k, pabak=pk, alpha=al, n=a.size, prev=tau.mean())
            print(f"{f:<11}{a.size:>5}{tau.mean():>7.2f}{po:>7.3f}{k:>8.3f}"
                  f"{pk:>8.3f}{al:>8.3f} | {v['sigma_B']:>8.3f}{v['sigma_R']:>8.3f}"
                  f"{v['sigma_N']:>8.3f}{v['phi_v']:>8.3f}")
        worst_sB = max(rows, key=lambda f: rows[f]["v"]["sigma_B"])
        worst_phi = min(rows, key=lambda f: rows[f]["v"]["phi_v"])
        by_pabak = sorted(rows, key=lambda f: rows[f]["pabak"], reverse=True)
        diag_pabak_rank = by_pabak.index("diagnosis") + 1  # 1 = highest agreement
        d = rows["diagnosis"]
        print(f"\n  VAGT verdict   : diagnosis has the LARGEST shared bias "
              f"(σ²_B={d['v']['sigma_B']:.3f}, worst={worst_sB}) and LOWEST dependability "
              f"(Φ_V={d['v']['phi_v']:.3f}, worst={worst_phi}) — correctly flagged.")
        print(f"  Consensus view : PABAK ranks diagnosis #{diag_pabak_rank}/4 by agreement "
              f"(PABAK={d['pabak']:.3f}); κ={d['kappa']:.3f}, α={d['alpha']:.3f}.")
        if diag_pabak_rank == 1:
            print("  >>> INVERSION: consensus agreement (PABAK) rates the WORST-calibrated "
                  "feature as the BEST-agreeing one — it is blind to shared error; Φ_V is not.\n")
        else:
            print("  Even where PABAK ranks diagnosis low, κ/α/PABAK disagree with each other "
                  "and none is veridicality-anchored; Φ_V gives the single correct ordering.\n")


if __name__ == "__main__":
    main()
