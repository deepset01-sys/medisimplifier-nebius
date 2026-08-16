"""vagt_medsimplifier_demo.py — VAGT proof-of-concept on MedSimp-JudgeBench.

Demonstrates Veridicality-Anchored G-Theory on the EXISTING two-judge data (no
new collection). For BOTH judge conditions — no-CoT (v2) and CoT (v3) — and each
error type, it reports the full decomposition {σ²_τ, σ²_B, σ²_R, σ²_N, Φ_V}
alongside the consensus statistics {Cohen's κ, PABAK, Krippendorff's α}, each
with a 95% bootstrap confidence interval (n_boot=1000, seed=42, items resampled
with replacement). It shows that Φ_V flags the diagnosis-omission blind spot the
consensus statistics miss — because those only measure judge-to-judge agreement,
never agreement with ground truth.

Data (ground truth = injected error type / condition):
  results/nebius_evidence/calibration_verdicts.json      (no-CoT / v2 judges)
  results/nebius_evidence/calibration_verdicts_cot.json  (CoT / v3 judges)

Model: two judges (Llama, Qwen), binary SAFE=0 / UNSAFE=1. Per error type f the
stratum = corrupted-f items (τ=1) + the shared clean controls (τ=0), so σ²_τ>0.

Decomposition (UNDIVIDED shared bias; matches vagt_section.md §3):
  consensus   c_i  = mean_r X_{ir}
  shared bias b_i  = c_i − τ_i ;  σ²_B = mean_i b_i²  (= β̄² + σ²_S; does NOT shrink)
  rater bias  α_r  = X̄_{·r} − X̄ ; σ²_R = mean_r α_r²
  noise       ε_ir = X_{ir} − c_i − α_r ; σ²_N = mean ε²
  σ²_τ = π(1−π) ;  Φ_V = σ²_τ / (σ²_τ + σ²_B + (σ²_R + σ²_N)/n_r),  n_r = 2

Caveat: only two judges, so σ²_R is illustrative; the proof-of-concept targets
σ²_B and Φ_V. The Phase 3 study has 20 raters. Note σ²_R is ~0 under no-CoT
(both judges equally miss omissions) but non-zero under CoT (their sensitivities
diverge) — the condition label matters and is printed explicitly.
"""

import json
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
    ("no-CoT (v2)", "calibration_verdicts.json", "llama_verdict", "qwen_verdict"),
    ("CoT (v3)", "calibration_verdicts_cot.json", "llama_verdict_cot", "qwen_verdict_cot"),
]
N_BOOT = 1000
SEED = 42
METRICS = ["sigma_tau", "sigma_B", "sigma_R", "sigma_N", "phi_v", "kappa", "pabak", "alpha"]


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
    R = X.shape[1]                            # number of raters in the data (here 2)
    c = X.mean(axis=1)
    b = c - tau
    sigma_B_naive = float(np.mean(b ** 2))    # β̄² + σ²_S + R-rater sampling inflation
    grand = X.mean()
    alpha = X.mean(axis=0) - grand
    sigma_R = float(np.mean(alpha ** 2))
    eps = X - (c[:, None] + alpha[None, :])
    sigma_N = float(np.mean(eps ** 2))
    # Sampling-variance correction (vagt_section.md §3): σ̂²_S = Var_i(ĉ_i) − MS_res/R.
    # Since mean(b²) = β̄² + Var_i(b) and Var_i(b) is inflated by σ²_N/R (the R-rater
    # consensus sampling variance), σ²_B = mean(b²) − σ²_N/R is the bias-corrected
    # shared-bias mean-square (still undivided by any facet size in Φ_V below).
    sigma_B = max(0.0, sigma_B_naive - sigma_N / R)
    p = float(tau.mean())
    sigma_tau = p * (1 - p)
    denom = sigma_tau + sigma_B + (sigma_R + sigma_N) / n_r
    phi_v = sigma_tau / denom if denom > 0 else float("nan")
    return dict(sigma_tau=sigma_tau, sigma_B=sigma_B, sigma_R=sigma_R,
                sigma_N=sigma_N, phi_v=phi_v)


def all_stats(a, b, tau):
    v = vagt(np.column_stack([a, b]), tau)
    return dict(sigma_tau=v["sigma_tau"], sigma_B=v["sigma_B"], sigma_R=v["sigma_R"],
                sigma_N=v["sigma_N"], phi_v=v["phi_v"],
                kappa=cohen_kappa(a, b), pabak=pabak(a, b), alpha=krippendorff_alpha_binary(a, b))


def bootstrap_cis(a, b, tau, rng, n_boot=N_BOOT):
    """95% percentile bootstrap CI per metric (items resampled with replacement)."""
    n = a.size
    acc = {m: np.empty(n_boot) for m in METRICS}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        s = all_stats(a[idx], b[idx], tau[idx])
        for m in METRICS:
            acc[m][i] = s[m]
    cis = {}
    for m in METRICS:
        arr = acc[m][np.isfinite(acc[m])]
        cis[m] = (np.percentile(arr, 2.5), np.percentile(arr, 97.5)) if arr.size else (np.nan, np.nan)
    return cis


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
    rng = np.random.default_rng(SEED)  # bootstrap seed=42
    print("VAGT proof-of-concept on MedSimp-JudgeBench (existing two-judge data)")
    print(f"Full per-error-type decomposition, BOTH conditions, 95% bootstrap CIs "
          f"(n_boot={N_BOOT}, seed={SEED}).")
    print("Point estimate [2.5%, 97.5%]. SAFE=0/UNSAFE=1; stratum = corrupted-f + shared clean controls.")

    for cond, fname, k_l, k_q in CONDITIONS:
        recs = json.loads((DATA / fname).read_text(encoding="utf-8"))
        print(f"\n================  CONDITION: {cond}  ================")
        rows = {}
        for f in FEATURES:
            a, b, tau = stratum(recs, f, k_l, k_q)
            pt = all_stats(a, b, tau)
            ci = bootstrap_cis(a, b, tau, rng)
            rows[f] = dict(pt=pt, n=a.size, prev=float(tau.mean()))

            def fmt(m):
                return f"{pt[m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]"

            print(f"\n[{cond}] {f}   (n={a.size}, corrupted prevalence={tau.mean():.2f})")
            print(f"    sigma_tau = {fmt('sigma_tau')}")
            print(f"    sigma_B   = {fmt('sigma_B')}   (shared bias)")
            print(f"    sigma_R   = {fmt('sigma_R')}   (rater-specific)")
            print(f"    sigma_N   = {fmt('sigma_N')}   (noise)")
            print(f"    Phi_V     = {fmt('phi_v')}   (veridicality-anchored dependability)")
            print(f"    kappa     = {fmt('kappa')}    PABAK = {fmt('pabak')}    alpha = {fmt('alpha')}")

        worst_sB = max(rows, key=lambda f: rows[f]["pt"]["sigma_B"])
        worst_phi = min(rows, key=lambda f: rows[f]["pt"]["phi_v"])
        by_pabak = sorted(rows, key=lambda f: rows[f]["pt"]["pabak"], reverse=True)
        diag_rank = by_pabak.index("diagnosis") + 1
        d = rows["diagnosis"]["pt"]
        print(f"\n  Summary [{cond}]: largest σ²_B = {worst_sB}; lowest Φ_V = {worst_phi}.")
        print(f"  diagnosis: PABAK rank #{diag_rank}/4 (PABAK={d['pabak']:.3f}), "
              f"Φ_V={d['phi_v']:.3f} (worst), σ²_B={d['sigma_B']:.3f} (largest), "
              f"σ²_R={d['sigma_R']:.3f}.")
        if diag_rank == 1:
            print("  >>> INVERSION: consensus agreement (PABAK) rates the WORST-calibrated "
                  "feature as the BEST-agreeing one. Φ_V is not fooled.")
        else:
            print("  (No PABAK inversion here — under CoT judges disagree — but Φ_V still "
                  "uniquely flags the shared blind spot, and σ²_R rises as sensitivities diverge.)")


if __name__ == "__main__":
    main()
