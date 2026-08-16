# Phase 1 Extension — Veridicality-Anchored Generalizability Theory (VAGT)

**Motivation.** Classical Generalizability Theory defines an item's universe
(true) score as the expectation of ratings *over the universe of raters*,
μ_p = E_r[X_{pr}]. Any error shared by all raters is therefore absorbed into
μ_p and counted as true-score variance, not error — G-Theory, like κ,
Krippendorff's α, and many-facet Rasch, is **structurally blind to shared
miscalibration** (the Δα↑/Δd′↓ cell of RQ1). Perturbation calibration removes
this blind spot by supplying the one thing naturalistic coding lacks: an
external criterion τ_p (planted ground truth). VAGT anchors the variance
decomposition to τ_p rather than to μ_p.

## 1. Shared-bias decomposition

For rater r on item p with planted truth τ_p ∈ {0,1}, write

    X_{pr} = τ_p + β̄ + β̃_p + α_r + ε_{pr},

with β̃_p, α_r, ε_{pr} mean-zero. The veridical error of a single rating
decomposes into four orthogonal parts:

    σ²_E = β̄² + σ²_S + σ²_R + σ²_ε

| Component | Symbol | Meaning | Seen by classical G-Theory? |
|---|---|---|---|
| Grand calibration offset | β̄² | all raters over-/under-flag on average | No — folded into true score |
| **Item-level shared bias** | σ²_S = Var_p(β̃_p) | raters agree *and are wrong*, varying by item | **No — the invisible term** |
| Rater-specific bias | σ²_R = Var_r(α_r) | severity / leniency | Yes (σ²_r); MFRM too |
| Residual noise | σ²_ε | rater×item interaction + error | Yes |

σ²_S (and β̄²) are precisely what no consensus-based coefficient can see.

## 2. Estimators

Two-way items × raters, one rating per cell; from the two-way ANOVA mean
squares, with b_p = X̄_{p·} − τ_p:

- σ̂²_ε = MS_res
- σ̂²_R = (MS_rater − MS_res) / N_items
- β̄̂ = X̄_{··} − τ̄   (τ̄ = planted prevalence)
- σ̂²_S = Σ_p (b_p − b̄)² / (N−1) − MS_res / R
- σ²_τ = τ̄(1 − τ̄)   (known exactly)

Classical σ̂²_p = (MS_item − MS_res)/R equals σ²_τ + σ²_S + 2·Cov(τ, β̃);
VAGT uses τ to split it. **Preferred estimation** reuses the hierarchical
logistic model already adopted in Repair #3:

    logit P(X_{pr} = 1) = θ_{τ(p)} + a_r + u_p,

where the truth-conditional fixed means give β̄, the rater random-intercept
variance gives σ²_R, and Var(u_p | τ_p) gives σ²_S. VAGT is thus a re-reading
of the model already in the design, not new machinery.

## 3. Modified D-study

For a decision from the mean of n_r raters, veridical error is

    σ²_Δ,V = β̄² + σ²_S + (σ²_R + σ²_ε) / n_r,
    Φ_V(n_r) = σ²_τ / (σ²_τ + σ²_Δ,V).

The shared-bias terms **do not shrink in n_r**. Hence a dependability ceiling:

    Φ_V(∞) = σ²_τ / (σ²_τ + β̄² + σ²_S).

*Actionable consequence classical G-Theory cannot give:* when σ²_S dominates,
adding raters is futile — only a **calibration regime** (perturbation-anchored
training, criterion setting, or judge choice) that reduces β̄ and σ²_S raises
the ceiling. Running the D-study pre- and post-calibration quantifies that as
ΔΦ_V, yielding an evidence-based raters-vs-calibration tradeoff.

## 4. CAI as a derived index

Restricting the decomposition to feature f gives the principled replacement for
the ad-hoc product:

    CAI(f) ≡ Φ_V(f; n_r)
           = σ²_τ(f) / [ σ²_τ(f) + β̄_f² + σ²_{S,f} + (σ²_{R,f} + σ²_{ε,f}) / n_r ].

This resolves every objection to the v6 CAI:

1. **Chance-corrected** — σ²_τ(f) = π_f(1 − π_f) plays the prevalence role that
   κ / PABAK use.
2. **Identifiable** — report the component profile (β̄_f, σ²_{S,f}, σ²_{R,f},
   σ²_{ε,f}), not just the scalar.
3. **Has a sampling distribution** — CIs by cluster bootstrap over items and
   raters.

The original CAI₀ = agreement × sensitivity × specificity is a monotone
heuristic proxy (on binary features β̄_f and σ²_{S,f} are functions of
sensitivity / specificity / prevalence, and σ²_ε of pairwise agreement) —
retained only as an informal readout.

## Assumptions & scope

- Estimated on the latent logit scale; the LPM/ANOVA form above is a
  first-order approximation for quick estimates.
- β̄ is the calibration of the *sampled* rater panel; a crossed random-rater
  design averages it toward the population offset.
- τ is trusted only for Tier-1 unanimous items; Tier-2 items enter σ²_S
  diagnostics via the majority key of Repair #4.
- **VAGT requires the clean-control (τ = 0) items flagged as missing in the v6
  review** — it presupposes that fix, and in return gives those controls a
  second, theoretical payoff.

## Phase 3 — Benchmark composition, power & budget

The Phase 3 calibration benchmark comprises **350 items: 200 perturbed (50 per
feature × 4 features) + 150 clean controls** — corrupted prevalence 57.1%,
**σ²_τ = 0.245** (near the 0.25 balanced maximum). Clean controls are
unperturbed segments confirmed error-free by the adjudication panel (lighter
than the three-expert unanimous validation for perturbations) and serve as
τ = 0 trials **for all four features simultaneously**, supplying the false-alarm
baseline that specificity, d′, and the VAGT negative stratum require. Each of
the 20 raters codes all 350 items under a within-subjects, counterbalanced
design (175 items/condition: 100 perturbed [25/feature] + 75 clean), so Δd′ and
Δα are estimated within rater.

Estimation uses crossed rater × item hierarchical random-effects logistic
regression (per Repair #3); d′, specificity, and CAI(f) are derived from the
fitted truth-conditional means, with CIs by cluster bootstrap over items and
raters. A 10,000-iteration probit-SDT power simulation (baseline d′ = 1.0,
unbiased criterion; ICC_rater × ICC_item ∈ {0.10, 0.20, 0.30}²; rater-level
paired-d′ estimator) gives, for the primary contrast: **> 80% power for
Δd′ = 0.5 at ICC ≤ 0.20 (range 0.821–0.868); 0.77–0.82 at ICC = 0.30,
consistent with the conservative rater-level paired estimator; GLMM-based
estimates expected to be higher given item-level information pooling.** For the
secondary SESOI Δd′ = 0.25, power is 0.19–0.27 across the grid; we therefore
**pre-register the secondary as exploratory** — a non-significant result will
not be read as evidence of absence, and the contrast will be reported with its
effect-size confidence interval rather than a reject/retain decision.

**VAGT budget justification:** *The 150 clean controls are the τ = 0 stratum
without which σ²_τ = 0, leaving specificity, d′, and the entire VAGT
decomposition (β̄, σ²_S, CAI) mathematically undefined — a prerequisite, not an
enhancement.*
