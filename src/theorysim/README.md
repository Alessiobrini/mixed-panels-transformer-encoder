# `theorysim` — MPTE theory-validation simulation (JBES revision add-on)

This package is a **self-contained add-on** for the JBES R2 revision. It validates the *linear* estimator the
inferential theory describes (Theorems 1-3: PCA on the attention-weighted panel `Z̃ = B Z A_z` with fixed,
frozen operators) in a controlled setting with known ground truth, and bridges it to the nonlinear model.
It produces a standalone paper-like report under `reports/theory_validation/`.

## Isolation contract (do not break)

This code **never modifies or overwrites** the existing empirical / mixed-frequency pipeline. It only:
- lives entirely under new paths: `src/theorysim/`, `src/config/cfg_theorysim.yaml`,
  `outputs/theorysim/`, `reports/theory_validation/`, `tests/test_theorysim_*.py`;
- reuses existing code **by importing pure functions** (e.g. `get_sinusoidal_encoding`,
  `read_aggregate_attention.plot_heatmap`, `utils.config.Config`) or by **copying** patterns, never by
  editing the originals;
- writes outputs only under `outputs/theorysim/` — nothing under `outputs/experiments/`, `outputs/tables/`,
  `outputs/replications/`, or `revision_exhibits/`.

Files NOT touched by this package: `src/data/simulate_to_long.py`,
`src/models/mixed_frequency_transformer.py`, `src/train.py`, the empirical evaluators, and `revision_exhibits/`.

All work is on the dedicated git branch `theory-validation-sim`.

## Experiments

- **E1** consistency (Theorem 1): estimation error vs the theoretical rate `ᾱ`.
- **E2** coverage / asymptotic normality (Theorems 2-3): the closed-form variance is taken directly from the
  paper; for the baseline iid DGP it reduces to `Ω=σ²Σ_F,y`, `Ξ=σ²Σ_Λ,ys` (exact analytic plug-in).
- **E3** efficiency / transfer learning (Remark on efficiency): joint vs Y-only.
- **E4** nonlinear bridge: the nonlinear model recovers the same factor space on linear-truth data.

## Model

A new axial-attention encoder on the **wide** equal-frequency panel `Z ∈ R^{T×N}` (X and Y stacked as
columns). Attention over columns → `A_z` (N×N), over rows → `B` (T×T). Axial avoids the `T·N` flattened-token
blow-up (e.g. 400×400 → 160,000 tokens) that the long-format empirical model would incur; cost is `N²+T²`.
