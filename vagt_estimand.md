# VAGT — Formal Estimand and Identifiability

## 1. Generative model (probit, truth as known offset)

Items i=1..N each carry an OBSERVED ground truth τ_i ∈ {0,1} (perturbation:
corrupted=1, clean=0). Raters r=1..R give binary ratings X_{ir} ∈ {0,1}
(UNSAFE=1). Latent probit:

    X_{ir} = 1[ η_{ir} > 0 ],    η_{ir} = θ_{τ_i} + b_i + u_r + ε_{ir},

  θ_0, θ_1 ∈ ℝ    fixed truth-conditional intercepts (calibration parameters);
                  a perfectly veridical pool has θ_1 → +∞, θ_0 → −∞, and finite
                  θ_1 encodes under-detection.
  b_i ~ N(0,σ²_S) item-level shared deviation — miscalibration common to ALL
                  raters on item i, beyond the truth-class mean.
  u_r ~ N(0,σ²_R) rater severity (main effect), Σ_r u_r = 0.
  ε_ir ~ N(0,1)   idiosyncratic residual; variance fixed to 1 (= σ²_N on the
                  latent scale) for probit identification.

τ_i enters as a KNOWN offset through θ_{τ_i} — not a latent to be inferred.
This external anchor is what distinguishes VAGT from classical G-Theory, whose
"true score" is the rater-population mean and therefore absorbs shared bias.

## 2. Estimands

Free parameters ψ = (θ_0, θ_1, σ²_S, σ²_R); residual σ²_N ≡ 1 (fixed).
Derived quantities (functionals of ψ), on the probability scale to match
reporting:

  population consensus   c_i = Φ( (θ_{τ_i} + b_i) / √(1 + σ²_R) )
  shared bias            b̃_i = c_i − τ_i
  σ²_B ≡ E_i[ b̃_i² ]  = ∫ (Φ((θ_{τ_i}+b)/√(1+σ²_R)) − τ_i)² φ(b;0,σ²_S) db,
                          averaged over the τ-mixture (prevalence π).
  σ²_τ = π(1 − π)         DESIGN CONSTANT (τ observed), not estimated.
  Φ_V  = σ²_τ / ( σ²_τ + σ²_B + (σ²_R + σ²_N)/n_r ).

σ²_B is thus a closed functional of ψ (evaluated by 1-D quadrature over b),
UNDIVIDED by any facet size; β̄² and σ²_S(prob) are recoverable as
β̄ = E_i b̃_i and Var_i b̃_i, with σ²_B = β̄² + σ²_S(prob).

## 3. Identifiability conditions

(C1) τ_i observed for every item ⇒ σ²_τ is a design constant and the θ-anchor
     dissolves the shared-bias / true-score confound that makes σ²_S
     non-identifiable in classical G-Theory.
(C2) Both truth classes present, π ∈ (0,1) ⇒ θ_0 and θ_1 both estimable and
     σ²_τ > 0 (Φ_V defined). REQUIRES the clean-control (τ=0) stratum.
(C3) N per feature large enough to estimate the b_i distribution: Var_i(consensus)
     identifies σ²_S after correcting the R-rater sampling variance
     (σ̂²_S = Var_i ĉ_i − σ̂²_N/R). Practically N ≥ ~40/feature.
(C4) R ≥ 2. u_r is item-INVARIANT (a rater constant) while ε_ir is item-specific,
     so the two raters' marginal UNSAFE-rate difference identifies the severity
     contrast and the residual within-item disagreement identifies σ²_N — they
     remain separable at R=2.
(C5) Latent scale fixed (σ²_N ≡ 1): standard probit identification.

n_r = 2 (the demo's configuration — the hardest case): the TARGET estimands
σ²_B and Φ_V are well-identified even at R=2, because both depend on the
consensus-vs-known-truth gap, anchored by τ and robust to the σ²_R/σ²_N split.
Only σ²_R itself is fragile at R=2 (a single contrast, 1 df); it enters Φ_V
solely through the /n_r term and does not affect σ²_B. Hence the demo's
σ²_B = 0.358 and Φ_V ≈ 0.40 are trustworthy, while its σ²_R ≈ 0 is a
low-precision estimate — consistent with, but not load-bearing for, the
shared-error conclusion. Phase 3 (R=20) gives σ²_R full df and tightens every
component.

## 4. Estimation & pre-registered parameter recovery

Estimation: Bayesian hierarchical probit (Stan / brms) with τ as offset,
weakly-informative priors (θ ~ N(0, 2.5²); σ_S, σ_R ~ half-Normal(0,1));
posterior draws propagate to σ²_B and Φ_V via the §2 functional (1-D quadrature).

Recovery (pre-registered): simulate M = 10,000 datasets from the model across a
grid of (θ_0, θ_1, σ²_S, σ²_R) spanning the observed MedSimp regime, at
R ∈ {2, 5, 20} and N/feature ∈ {50, 150}; refit; report bias and 95% CI coverage
for σ²_B and Φ_V. Recovery at R=2 establishes the demo estimates are unbiased;
R=20 establishes Phase-3 precision.
