# theorysim build log (for end-of-build Q&A)

Running log of each stage, what was decided/built, how it was verified, and any open
question worth raising. Newest stage at the bottom. Branch: `theory-validation-sim`.

---

## Stage 0 — Isolation scaffold
- **Built:** dedicated git branch `theory-validation-sim`; new package `src/theorysim/`
  (`__init__.py` banner + `README.md` isolation contract); new dirs `outputs/theorysim/`,
  `reports/theory_validation/`.
- **Decision:** every artifact carries the `theorysim` / `theory_validation` prefix; no
  existing file is edited; reuse only by import or copy.
- **Verified:** `git status` shows only new files; no `M` on existing source (the one
  pre-existing `M cfg_equity.yaml` predates this work).
- **Open Q:** none.

## Stage 1 — Estimator spine + closed-form unit tests
- **Built:** `dgp.py` (union-factor linear DGP with ground-truth `F`,`Λ`,`C` export),
  `estimator.py` (PCA on `Z̃=BZA_z`), `alignment.py` (LS rotation, error metrics),
  `rates.py` (three-term `ᾱ`, diagnostics), `operators.py` (scaling, within-block
  restriction, oracle, IO), `src/config/cfg_theorysim.yaml`, `src/utils/plotting.py`
  (copied golden-ratio `set_size`).
- **Decision:** estimator is the explicit two-step PCA, NOT the transformer forecast;
  alignment is least squares (not orthogonal Procrustes); operator value map `W^v=I`
  (so `Z̃=BZA_z`), matching paper eq. just after the softmax definition.
- **Verified (closed form = ground truth):** 17/17 pytest pass.
  - DGP: `C==FΛᵀ` exact; 2×2 block zeros; cross-Gram rank 1 (B.3); factor var≈1;
    reproducibility; save/load round-trip.
  - Estimator: noiseless oracle recovers `Λ̂ᵀΛ̂/N=I`, `e_C<1e-18`, `e_Λ,e_F<1e-16`;
    error falls as N,T grow.
  - Operators: `tr(A_zᵀA_z)=N`, `tr(BᵀB)=T` after scaling; within-block zeros cross terms.
  - Rate: `ᾱ → 1/T + 2/N` at `A_z=I,B=I`.
- **Note:** benign `RuntimeWarning ... in matmul` on identity matmuls (numpy 2.2 + Apple
  Accelerate); results correct (exact assertions pass). Will filter in runners.
- **Open Q:** none on the spine.

## Stage 2 — Variance question resolved against the paper
- **Finding:** the closed-form asymptotic variance IS in `paper/main.tex` (Theorems 2-3:
  `σ²_{C,it,F}`, `σ²_{C,it,Λ}`, three-regime limit). For the baseline iid DGP the long-run
  covariances collapse to `Ω=σ²Σ_F,y`, `Ξ=σ²Σ_Λ,ys`, so E2 uses an exact analytic plug-in.
- **Decision:** do NOT ask Kate for the variance; E2 is unblocked. Only the (deferred)
  dependent-noise variant needs a HAC kernel/bandwidth, which the paper leaves unspecified.
- **Open Q (future only):** HAC kernel/bandwidth for `Ω`,`Ξ` under dependent noise.

## Stage 3 — Axial-attention model
- **Built:** `axial_model.py` (`AxialAttentionMPTE`, `fit_forecast`). Wide input `Z∈R^{T×N}`,
  two softmax attentions exactly as in the paper: temporal `B=softmax(Q_τK_τᵀ/√d)` from the
  rows of `Z`, cross-sectional `A_z=softmax(Q_zK_zᵀ/√d)` from the rows of `Zᵀ`; value map
  `W^v=I` so `Z̃=BZA_z`; sinusoidal PE on the time axis; k-wide latent read-out; head
  forecasts the (masked) target column. `use_nonlinearity` flag = AB1 (linear) vs E4.
- **Decisions to flag:**
  (1) Training objective = forecast the designated target column with that column masked
      from the input (forces both attentions to carry signal). "Same spirit as empirical
      AB1" but the exact objective is my choice — worth confirming with Kate.
  (2) `nhead=1`, `num_layers=1` (as agreed); multi-head averaging deferred.
- **Verified:** `test_theorysim_model.py` (3/3): A_z (N×N), B (T×T) row-stochastic and
  nonneg; forward returns (T,); training reduces MSE; `extract_and_average` stays
  row-stochastic and scales to `tr=N`, `tr=T`.

## Stage 4 — E1 runner + smokes + first diagnostic finding
- **Built:** `exp_e1_consistency.py` (oracle / independent / same_sample arms; per-(T,N)
  operator build, PCA+LS-alignment per rep, records `e_Λ,e_F,e_C,ᾱ`; `--smoke`; benign
  matmul warnings filtered).
- **Verified — oracle arm validates Theorem 1:** on a diagonal grid (40→320),
  `e_C` ∝ `ᾱ` with **log-log slope = 1.064** (target 1). The estimator + rate math is correct.
- **Finding — learned arm exhibits concentrated attention (referee item 2 / A.7):** the
  trained AB1's `A_z` is concentrated (row-max ≈ 0.20, far from uniform ≈ 0.01-0.03) and
  `‖A_z‖_op` GROWS with N (≈3.9, 4.8, 6.8 at N=40,80,160), essentially unchanged by more
  training (800 epochs). This is exactly the Assumption-A.7 "hub/concentrated attention"
  case the design flags as item 2. Consequence: in the learned arm the rate's term 2 does
  not vanish, so consistency is governed by whether `‖A_z‖_op=O(1)` — the empirical question
  E1 + the diagnostic plot are meant to answer. Pipeline is correct (errors finite,
  operators valid); this is a result to characterize, not a bug.
- **Open Qs for end-of-build — RESOLVED by user:**
  (a) Training objective = masked-target forecast: **confirmed** ("same spirit as empirical AB1").
  (b) Learned-arm A.7/op-norm concentration: **report as-is this pass** (failure-as-result);
      no within-block restriction for E1 now.
  (c) Temporal operator `B`: **keep learned** (not B=I).
- **Test status:** full `tests/test_theorysim_*.py` = **20 passed**.

## Stage 5 — Training-length / convergence check
- **Question (user):** how long was MPTE trained?
- **Used so far:** smoke 60 epochs (lr 1e-3); op-norm diagnostic 60 & 800 (lr 1e-2).
- **Convergence trace (N=T=80, full-batch Adam, lr 1e-2):** MSE 0.519→0.190→0.029→0.017→0.002
  at epochs 50/150/400/800/2000, but **‖A_z‖_op plateaus at 6.164 by 400 epochs** and is
  unchanged through 2000. The later loss gains come from the read-out/head (which E1 does not
  use), so the operator the estimator depends on is converged by ~400 epochs. Confirms the
  concentration finding is structural, not under-training. Wall-clock <0.6s even at 2000 epochs.
- **Decision:** set the E1 runner default to **800 epochs, lr 1e-2** (safely past the operator
  plateau; still sub-second per fit). `exp_e1_consistency.py` updated.
- **Open Q:** none.

## Stage 6 — E4 nonlinear bridge
- **Built:** `exp_e4_bridge.py`; `canonical_correlations` added to `alignment.py`.
  Trains the nonlinear model (`use_nonlinearity=True`) on linear-truth data, extracts the
  k-dim latent (`return_latent`), CCA vs true attended factors `F^(B)`, plus an output check
  (nonlinear vs linear vs truth on the target column).
- **Verified:** `test_theorysim_cca.py` (3/3): rotation -> all canonical corr 1; independent
  -> <1; linear PCA recovers the factor subspace (>0.99) at low noise.
- **Smoke finding (N=80, 400 epochs):** nonlinear latent canonical corr vs `F^(B)` =
  [0.97, 0.85, 0.49, 0.18]; linear PCA = [0.97, 0.93, 0.86, 0.47]. The forecast-trained
  nonlinear latent strongly recovers the TARGET-relevant factors and weakly the X-only
  factors (it is trained to predict the target, not to span the whole space). Output check
  needs the full N,T to show clean coincidence. Qualitatively the nonlinear model tracks
  the linear one on linear data, which is E4's claim.
- **Open Q:** at full scale, confirm the output check coincides near the noise floor.

## Stage 7 — E3 efficiency / transfer learning
- **Built:** `exp_e3_efficiency.py` (joint vs Y-only common-component MSE, sweep N_x; oracle
  operators so the comparison isolates the transfer gain).
- **Smoke finding:** ratio (Y-only / joint) rises above 1 and grows with N_x
  (0.96 at N_x=20 -> 1.12 -> 1.17 -> 1.18), saturating because only factor 1 is shared
  (X loads Y-strong factor 1 only). Matches the efficiency Remark.
- **Open Q:** none.

## Stage 8 — E2 coverage + a real variance fix
- **Built:** `exp_e2_coverage.py` (analytic plug-in coverage of CIs for `C_{y,i,t}`, three
  regimes, QQ of the studentized statistic). First-pass scope: pure Y-strong model
  (`k_R=0`) so `Σ_Λ,ys=I` under the PCA normalization and the common component is exactly
  the Y-strong component; oracle operators; baseline iid so `Ω=σ²Σ_F,y`, `Ξ=σ²Σ_Λ,ys`.
- **Finding + fix (important):** the single-term corner variances UNDER-COVER off their
  corner. The F-only SE drops the `σ²_Λ/T` term, so coverage degrades as N->T (z_sd 1.14 at
  N=50 -> 1.45 at N=400=T). Fix: studentize with the GENERAL Theorem-3(iii) variance
  `Var(Ĉ-C)=σ²_F/N + σ²_Λ/T` (the corners are its limits). With the combined SE, coverage is
  near-nominal in all settings: cov95 ≈ 0.93 (F-dom), 0.95 (Λ-dom), 0.96 (mixed), and the
  previously-broken balanced 400×400 goes 0.83 -> 0.94. Code now uses the combined SE always.
- **Verified:** smoke cov95 = 0.927 / 0.950 / 0.960 across the three regimes; z_sd ≈ 1.
- **Open Q:** residual ~0.93-0.94 at 95% in the off-corner regimes is mild finite-sample
  plug-in noise; tightens with N,T. k_R>0 with explicit Y-strong identification is a follow-up.
- **Test status:** full `tests/test_theorysim_*.py` = **23 passed**.

## Stage 9 — Paper-like report PDF
- **Built:** `plots.py` (E1 rate, E2 QQ, E3 efficiency, E4 CCA, operator heatmaps; sized via
  `set_size`), `tables.py` (\best/\second scriptsize tables), `make_report_assets.py`
  (figures+tables+heatmaps from CSVs), `reports/theory_validation/{report.tex,
  build_report.sh, exhibit_manifest.csv}`. One command builds everything:
  `bash reports/theory_validation/build_report.sh` (QUICK=1 fast, QUICK=0 full reps).
- **Built end-to-end in 37s** (QUICK): runs all four experiments, makes assets, writes the
  reproducibility appendix from the manifest, latexmk -> `report.pdf` (5 pages).
- **Headline numbers in the PDF:**
  - E1 oracle log-log slope **0.958** (Theorem 1 holds); learned-arm slope **0.173**
    (the A.7-concentration consequence, shown honestly).
  - E2 coverage @2000 reps: cov95 = **0.936 / 0.941 / 0.947** (F-dom / Λ-dom / mixed); QQ on
    the N(0,1) line.
  - E3 ratio (Y-only/joint) **0.91 -> 1.16** as N_x grows (transfer gain).
  - E4 canonical corr nonlinear vs F^(B) = [0.97, 0.85, 0.60, 0.18], tracking linear PCA;
    nonlinear recovers the target-relevant factors, weaker on X-only factors.
  - Operator heatmaps show the concentrated "hub" stripes behind the ‖A_z‖_op growth.
- **Visual QA done** (global LaTeX rule): compiled, log greps **0 overfull boxes** (fixed the
  appendix longtable with break points), all 5 figures + 3 tables rasterized and inspected.
- **Open Q:** none on the report mechanics. Substantive items for Kate: the learned-arm A.7
  concentration (report-as-is, agreed) and the E4 output-check coincidence at full scale.

## Summary of open questions to raise with Kate
1. Learned-arm A.7/item-2 concentration: ‖A_z‖_op grows with N -> learned-arm E1 slope ~0.17
   (vs oracle ~1). Reported as-is this pass; is "failure-as-result" the framing she wants, or
   should we add the within-block restriction / sharpen attention?
2. E4 at full scale: confirm the output check (nonlinear ≈ linear ≈ truth) coincides near the
   noise floor; current smoke has nonlinear RMSE > linear at N=150.
3. E2 scope: first pass uses k_R=0 (pure Y-strong) + oracle operators; extend to k_R>0 with
   explicit Y-strong identification and learned operators?
4. (Future) HAC kernel/bandwidth for the dependent-noise DGP variant.

## Stage 10 — Same-Lambda fix + Option B (why the learned arm fails, and how to fix it)
- **Same-Lambda sample-splitting (correctness fix):** found that training and estimation
  panels were redrawing Lambda, violating the design's "independent draw, same Lambda."
  Added `loadings_seed` to `dgp.draw` (Lambda from a separate rng, F/e from `seed`); E1 now
  shares one Lambda across train/extract/estimation. Effect: learned-arm slope 0.17 -> 0.24.
  Real correctness improvement, kept regardless of framing.
- **Within-block restriction does NOT fix E1.** Added `independent_wb` arm. Within-block is an
  identification tool for E2/E3 (Assumptions B.1/B.3); it does nothing for the E1 consistency
  RATE (governed by A.7). Slope with wb ~0.15 (no better). Documented: don't apply wb to E1.
- **Diagnosis of the learned-arm failure (A.7).** After scaling, ||A_z||_op = sqrt(N)
  ||S_z||_op/||S_z||_F. A.7 (bounded op-norm) needs an identity-like operator. The trained
  softmax attention is hub-concentrated -> op-norm grows with N -> consistency stalls. Both
  hub-concentrated AND uniform attention give op-norm ~sqrt(N); only identity-like passes.
- **Option B: three modeling levers recover the slope (0.22 -> ~0.95), diagnostic grid
  N=40..320, 40 reps, 500 epochs.** Added to the model: `fit_reconstruct`/`decoder` (autoencoder
  objective, Prop 3 ~ PCA), `tie_qk` (share Q/K -> Gram/symmetric attention, diagonal-dominant),
  `attn_temp`. Findings:
  - objective: forecast 0.22  ->  reconstruct 0.56  (masked-target objective concentrates on hubs)
  - + tie_qk:  0.78          (symmetric attention is identity-like at small/moderate N, op-norm 2)
  - + wider d_model: 64 -> clean declining e_C at N<=160; 128 -> slope 0.95 (op-norm growth was a
    d_model=16 bottleneck: 16 dims can't separate 320 variables, factor-mates collide off-diagonal).
  - temperature (0.3): mild, not decisive.
- **Conclusion (for Kate):** the theory-empirics gap is NOT fundamental. A trained softmax
  attention satisfies A.7 and inherits Theorem 1 when it is (a) trained to reconstruct, (b)
  symmetric (tied Q/K), (c) given enough projection width to resolve the cross-section. This is
  a constructive answer to the referees, stronger than "the learned arm fails."
- **Decision point parked for Kate:** d_model=16 was the locked choice ("smallest"); the
  A.7-friendly operator wants d_model~128. NOTE these are independent of the k=4 latent readout
  (dim_feedforward=k, for E4), so widening d_model does NOT break E4 comparability. Options:
  keep d_model=16 and report partial recovery (~0.78) / failure-as-result, OR widen d_model for
  the operator-learning step and report near-full recovery (~0.95).
- **Caveat:** diagnostic numbers are noisy (40 reps, 500 epochs); slope 0.95 is encouraging but
  per-cell e_C is non-monotone. A clean confirmation needs more reps/epochs/grid points before
  it goes in the report.
- **Verified:** `tests/test_theorysim_model.py` + `_operators.py` = 8 passed (changes are
  additive with defaults; existing behavior unchanged).

## Stage 11 — Clean confirmation + "remove the d_model transform" (user's idea wins)
- **d_model and the theory:** the theorems are operator-agnostic (take A_z,B as fixed, assume
  only A.7 on the result). d_model lives entirely in the construction layer and is absent from
  the proofs. It is "overlooked" only as an unstated condition for A.7 to hold.
- **Metric correction (important):** slope of log(e_C) vs log(alpha_bar) is CONTAMINATED when
  op-norm grows, because alpha_bar's operator-norm term inflates. The earlier "d_model=128 ->
  slope 0.95 ~ recovered" was an artifact: at higher reps that config's e_C is FLAT ~1.0
  (0.63,1.19,0.99,0.98), i.e. NOT converging, while the slope still reads 0.93. Use the raw
  consistency test e_C -> 0 directly; keep slope-vs-alpha_bar only for A.7-satisfying operators
  (oracle), where it correctly reads ~1.
- **Parameter-free operator (W=I, tied Q=K, no projection):** A_z=softmax(Z^T Z/(T*temp)),
  B=softmax(Z Z^T/(N*temp)). Clean run, 80 reps, grid N=T=40..640:
  - temp 0.1: e_C 0.53,0.31,0.20,0.21,0.17 -> DECLINING (consistent), ||S_z||_F/sqrt(N)~0.9,
    op-norm 1.5->6.3 (still grows). Effective rate e_C ~ N^-0.44.
  - lower temp -> more identity-like, lower error level; rate not fully restored.
- **Ranking by the correct test (e_C -> 0), grid 40..320:** oracle beta~1.0 > parameter-free
  beta~0.44 (converges) >> learned reconstruct+tie+d128 beta~0 (flat) > learned forecast.
  => the learned d_model transform HURTS; removing it is the only adaptive operator that
  actually converges. User's instinct confirmed.
- **Mechanism / theory finding for Kate:** op-norm growth is driven by the FACTOR STRUCTURE
  (correlated variables attend in blocks/hubs), intrinsic to any data-adaptive attention, not a
  parameterization artifact. A.7 (bounded op-norm) is in tension with what attention does. Only
  the strict-identity oracle satisfies it. Candidate paper move: relax A.7 to a slowly-growing
  op-norm, or restate the rate; report the parameter-free kernel as the practical adaptive
  operator that stays consistent.
- **Model changes (all additive, defaults preserve old behavior):** axial_model gains
  fit_reconstruct/decoder, tie_qk, attn_temp; operators.train_operators gains objective/tie_qk/
  attn_temp. Tests still 8 passed. Diagnostics live in scratchpad (diag_objective.py,
  diag_freeops.py) -- to be promoted into the package once Kate signs off on the framing.
- **TODO before report rebuild:** (1) decide consistency metric = e_C->0 (raw) for adaptive arms;
  (2) add a parameter-free operator arm to operators.py + E1; (3) rerun E2/E3 deciding oracle vs
  parameter-free operators; (4) put the A.7-tension note in report.tex.

## Stage 12 — Independent code audit (Sonnet) + metric correction (MAJOR)
- **Audit verdict:** NO math/implementation bugs. All formulas (DGP, PCA estimator, ls_align,
  alpha_bar, operator scaling) verified; 23/23 tests pass. Oracle slope confirmed ~1.04.
- **Critical finding (task E):** absolute e_C = ||C_hat-C_true||^2/(NT) is NOT comparable across
  arms, because C_true = B C A_z has an operator-dependent scale. The learned operator's
  concentration inflates ||C_true||^2 ~5x, and since its op-norm GROWS with N, ||C_true|| grows
  with N too. So a shrinking RELATIVE error appeared as a flat ABSOLUTE error -> I mis-read it as
  "the learned arm stalls." That conclusion was a measurement artifact, not a real stall.
- **Correction (decisive recompute, relative e_C = ||C_hat-C_true||^2/||C_true||^2, 80 reps,
  grid N=T=40..320):**
  - oracle:        rel e_C 0.145->0.019, slope vs N = -1.00
  - parameter-free: rel e_C 0.115->0.020, slope vs N = -0.85
  - learned(forecast,masked): rel e_C 0.157->0.031, slope vs N = -0.77
  ALL THREE ARMS CONVERGE. The learned (headline) arm reaches near-oracle error by N=320.
- **Reversed conclusion:** the theory DOES bridge to the data-driven operator. Kate's "matching
  across arms" criterion is substantially met. A.7 op-norm growth costs a mild rate penalty
  (-0.77 vs -1.0) but does NOT break consistency. Earlier Stage 10-11 framing ("learned arm
  fails / theory doesn't bridge") is RETRACTED as a metric artifact.
- **Audit's other notes:** (1) report relative e_C as the primary E1 metric (done in recompute);
  (2) plot e_C vs N, not vs alpha_bar (alpha_bar non-monotone in N for adaptive arms); (3) PE is
  applied when learning operators (Z+pe) but not in the estimator (raw Z) -- by design, document
  it, may slightly worsen the learned arm; (4) within_block_restrict breaks row-stochasticity
  pre-scaling (intentional); (5) optional large-N (1000-2000) floor test.
- **TODO:** switch E1 to relative e_C in the package; rerun E1 all arms + parameter-free at full
  reps/finer grid; then the paper-faithful lagged-target variant is polish, not a rescue.

## Stage 13 — Coherence check + relative-metric rerun to N=1000 (point 1-2)
- **Coherence vs paper (main.tex Sec 3.3 + Remark 1):** our axial S_z/S_tau construction IS the
  paper's (line 381); the parameter-free Q=K=Z kernel IS the paper's "simplest case" (line 358);
  train-freeze-apply matches Remark 1 (forecast objective). DGP/dimensions/estimator all match.
  A.7 is stated by the paper (line 428) as its claimed condition -> our op-norm probe tests
  exactly that. FAITHFUL = axial + separate Q/K + forecast objective + oracle + parameter_free.
  Only faithfulness fix = lagged target (point 4). tie_qk/attn_temp/reconstruct = non-paper
  diagnostics, set aside.
- **Package changes:** alignment.common_error_relative (scale-invariant, now primary E1 metric);
  operators.parameter_free_operators (paper simplest case); exp_e1 records e_C_rel + op_norm and
  gains the parameter_free arm. 23/23 tests pass.
- **Rerun (relative e_C, grid N=T=80..1000, 100 reps), slope vs N:**
  - oracle:        0.078->0.0059, slope -1.02, op-norm 1.0
  - parameter_free: 0.062->0.0072, slope -0.85, op-norm ->~6
  - independent:   0.160->0.011,  slope -0.98, op-norm ->18
- **CONFIRMED (favorable):** all three arms converge at ~rate 1. Kate's "matching across arms"
  criterion is MET. The learned (headline) arm inherits Theorem 1 at the FULL rate despite
  op-norm growing to 18. So A.7's literal op-norm bound is violated yet consistency is robust to
  it. RETRACT both earlier framings: "learned arm stalls" (Stage 10-11) AND "mild rate penalty"
  (Stage 12) -- both were metric artifacts / overstatements. CSV: outputs/theorysim/e1/relative_rerun.csv.
- **Open note for Kate:** op-norm grows but rate is unaffected here -> A.7 may be sufficient but
  stronger than necessary, OR the floor bites only at much larger N (auditor's large-N test).

## Stage 14 — Paper-faithful lagged-target arm (point 4)
- **Change:** axial_model gains target_mode ("mask" zeros the target column; "lag" feeds the
  target's own 1-step lag, paper-faithful: the target's PAST is a legitimate predictor, only the
  contemporaneous value is withheld). Threaded through train_operators; new E1 arm
  `independent_lag`. 23/23 tests pass.
- **Faithful rerun (relative e_C, N=T=80..1000, 100 reps):** independent_lag rel_eC
  0.174->0.0125, slope vs N = -1.04 (op-norm grows 6->14). CSV: relative_rerun_lag.csv.
- **Final E1 picture (all relative metric, slope vs N):** oracle -1.02, parameter_free -0.85,
  independent(masked) -0.98, independent_lag(faithful) -1.04. ALL converge at ~rate 1; the
  faithful learned arm is the cleanest (= oracle). Kate's "matching across arms" MET. Theory
  bridges to the data-driven operator at the FULL rate. A.7 op-norm bound violated (grows) but
  consistency robust -> report as: A.7 sufficient, stronger than necessary in these designs.
- **Report status:** report.tex still has the OLD absolute-metric E1 (oracle ~1, independent
  ~0.17 artifact). Needs: E1 -> relative e_C vs N, 3 paper arms (oracle/parameter_free/
  independent_lag) all ~ -1, op-norm-growth note. E2/E3 (oracle ops) stand. E4 still partial.
  Awaiting user decision on report direction before rebuild.

## Stage 15 — Report rebuild (E1 corrected, point D)
- **User decisions:** E1 shows 3 paper arms (oracle, parameter_free, independent_lag); the
  op-norm/A.7 point is stated as fact and flagged for Kate (no theory change committed).
- **Assets:** plots.plot_e1_relative (2 panels: (a) relative e_C vs N log-log, slope ~ -1 for
  all arms + slope-(-1) guide; (b) ||A_z||_op vs N growing for data-driven arms); tables.
  make_e1_relative_table (slope, rel e_C, op-norm at N_max). make_report_assets + build_report.sh
  now consume relative_rerun*.csv (rerun_e1_relative.py) instead of the old absolute exp_e1.
- **report.tex E1 section rewritten:** relative scale-invariant metric, three paper-grounded
  arms all converging at slope ~ -1 (matching = design criterion met), plus "A note on
  Assumption A.7" (op-norm grows yet rate holds -> A.7 sufficient but stronger than necessary;
  relaxation left open for Kate). exhibit_manifest updated.
- **Verified:** report compiles, 0 overfull boxes, 6 pages; E1 fig/table/prose visually QA'd
  (pages 1-2). E2/E3/E4/operator exhibits unchanged from prior build (those experiments stand).
- **Remaining:** E4 nonlinear bridge still partial; section titles use em-dashes (pre-existing
  doc header style, optional uniform cleanup); a full QUICK=0 rebuild regenerates all CSVs from
  scratch (slow at N=1000) but the committed relative_rerun*.csv already back the current PDF.

## Stage 16 — Improved E4 (faithful) + full QUICK=0 build
- **E4 fixes:** same-Lambda (loadings_seed); paper-faithful lagged-target objective for BOTH the
  nonlinear and linear-ablation models; Y-strong-focused CCA (latent vs B F_ys); larger scale
  (N=T=300). Output check rewritten to compare the two MODELS' forecasts (nonlinear vs linear
  ablation, the Table-1 bridge) against the OBSERVED target, relative to target sd.
- **What we found (tried everything):**
  - CCA (the reliable bridge): linear PCA recovers the FULL attended factor space (cc ~ [0.99,
    0.99,0.98,0.85]); the nonlinear latent recovers the TARGET-RELEVANT direction strongly
    (cc1 ~ 0.98) but not the orthogonal Y-strong factor (Y-strong cc ~ [0.98, 0.59]). Reason: a
    single-target forecast objective only needs the one direction that predicts its target.
  - Output/forecast check is NOT reliable here: the deliberately minimal model (d_model=16, 1
    layer, k=4) is a weak forecaster even in-sample (train RMSE ~1.8 > noise floor sigma~1.41);
    PE contributes a little OOS overfit (rel 1.17->1.06 with PE off) but the main cause is model
    capacity. So we do NOT use the forecast head as the E4 headline.
  - Honest E4 = subspace statement: nonlinear recovers the target signal direction; linear
    recovers the full space; forecast-accuracy equivalence stays the paper's Table 1.
- **report.tex E4 section rewritten** to this honest framing; plot_e4_cca filter fixed (the new
  cca_*_ys_ keys were colliding with the all-k bars). build_report.sh E4 now N=T=300, epochs
  1200 (E4N/E4EP); manifest updated. 23/23 tests pass.
- **Launched full QUICK=0 build** (E1 to N=1000, E2/E3 full reps, improved E4, heatmaps, compile).
- **Open for user/Kate:** whether E4's "nonlinear recovers the target direction, linear the full
  space" framing is acceptable, or whether to also run a reconstruction-objective nonlinear arm
  (would recover the full subspace like PCA, but departs from the paper's forecast objective).

## Stage 16b — Full QUICK=0 build complete + visual QA
- Build finished. Full-scale numbers in the report:
  - E1 (relative e_C, N up to 1000): oracle slope -1.02, parameter_free -0.85, independent_lag
    (faithful) -1.04. All converge; op-norm panel shows growth (oracle 1, others to ~6/14).
  - E2: cov95 = 0.936/0.941/0.947, z_sd ~1.0; QQ on the line.
  - E3: Y-only/joint ratio 0.91 -> 1.15.
  - E4 (N=T=300, 50 oos): Y-strong cc nonlinear [0.985, 0.60], linear [0.991, 0.891]; all-k
    nonlinear [0.993,0.917,0.622,0.202], linear [0.996,0.991,0.981,0.853]. Honest framing:
    nonlinear recovers target-relevant direction, linear recovers full space.
- Visual QA: compiled, 0 overfull boxes, 5 pages; E1/E2/E3/E4/operators/appendix all rasterized
  and inspected. Report is the corrected, faithful, full-scale deliverable.
