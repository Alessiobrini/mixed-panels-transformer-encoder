# What changed in every exhibit — lead=2 rerun (2026-06-13)

REVISED exhibits are inserted directly BELOW each original in `paper/main.tex` (blue `REVISED`
caption, `_REV` labels, figures = `*_REVISED.pdf`). Originals untouched. Pushed to Overleaf.

## Convention
- **MPTE** and its monthly-ingesting ablations (**AB1–AB4**) get the lead → values CHANGE.
- **AB5** (quarterly-only), **AR**, **MIDAS**, **OLS** are lead-invariant → reproduce the paper EXACTLY
  (verified: sim AR/MIDAS at the reference seed to 1e-5; FRED AR/MIDAS/OLS byte-identical).
- **XGB/NN** are lead-invariant but not bit-reproducible (multi-thread / training noise) → the paper's
  values are CARRIED OVER unchanged (author decision).

---

## Table 1 — simulation (`Tab:evals_simulation`)
- **Structure change:** single-seed point estimates → **mean (SD) over 100 Monte-Carlo replications**
  (the std-in-parentheses you asked for). 4 explosive linear seeds dropped (paths blew up for every
  model incl. AR).
- **Headline:** with lead=2, MPTE now beats MIDAS in ALL regimes and AB5 in the nonlinear ones — the old
  high-regime tie is gone.

| regime | MPTE RMSE (new) | vs MIDAS | vs AB5 | vs AR |
|---|---|---|---|---|
| Linear | 1.178 (0.13) | 81% (p=1e-9) | 55% tie | 86% |
| Mild | 1.223 (0.14) | 87% (p=5e-15) | 64% (p=0.01) | 93% |
| High | 1.368 (0.25) | 80% (p=9e-10) | 76% (p=4e-7) | 94% |

Note: the published single-seed high MPTE (1.158) was a favorable draw; the honest N=100 mean is 1.368,
still below MIDAS (1.421) and AB5 (1.433). AR/MIDAS/AB5 columns are statistically unchanged by the lead.

## Tables 2 & 3 — FRED competing models (`Tab:empirical1/2`)
- MPTE row changes on every target; AR/MIDAS/OLS reproduce exactly; XGB/NN carried over.
- MPTE improves on **10/13** targets vs its old self; beats MIDAS **9/13**, XGB **9/13** (was 5/13), AR ~all.
- In the `empirical1` (MPTE-wins) group: MPTE improved on 4/5; still loses GDPC1 to MIDAS (0.0135 vs
  0.0116) — the known R2-10 case, now closer.
- XGB still wins the smooth trade/labor series (EXPGSC1, FPIx, IMPGSC1, UNRATE).

## Table 4 — win counts (`Tab:empirical3`)
- Recomputed from the new MPTE; MPTE's win counts rise (it now leads more targets on RMSE).

## Tables 5 & 6 — empirical ablations (`Tab:empirical_abl1/2`)
- MPTE + AB1–AB4 rows change (all see the lead); **AB5 row unchanged** (quarterly-only, lead-invariant).

## Figure 3 — forecast plots (`Fig:preds`)
- Same style/targets (GDPC1, OUTNFB). MPTE and MIDAS lines reflect the lead2 run; XGB line is the
  carried-over paper draw. Visually very close to the old (data/style identical, MPTE line shifts).

## Figures 4–7 — attention heatmaps (`Fig:Az_*`, `Fig:B_*`) — BIGGEST visual change
- The lead2 model is a different model (new info set + re-tuning), so the patterns differ substantially.
- **Temporal B:** now **26 lags instead of 24** (the 2 within-quarter lead months enter the context),
  and attention concentrates on the **most recent lags** (the lead months) rather than older lags. This
  is interpretable and SUPPORTS the lead story (the model exploits the timely data).
- **Cross-sectional Ax:** emphasizes different variables (e.g. GDPC1: PAYEMS/employment/S&P vs old
  HWI/consumption) and is more diffuse.
- **Implication:** the Section 7.2 narrative (which names specific variables/lags) must be rewritten to
  match the new figures — the *direction* of the new temporal story is arguably stronger.

## Appendix Tables 7–10 — DM & MCS (`Tab:DM_*`, `Tab:MCS_*`)
- All involve MPTE, so all entries change. DM stats recomputed (deterministic); MCS recomputed
  (seed 42, regenerated since MPTE moved). AR-vs-others structure unchanged.

---

## To decide next ("what to do")
1. Once you've eyeballed the side-by-side on Overleaf, comment out / remove the OLD exhibits and renumber.
2. Rewrite the Section 7.2 attention narrative to match the new heatmaps.
3. Soften/refresh prose tied to old numbers (R2-10 GDPC1, high-regime claims, etc.).
4. Decide final caption wording (currently old caption + blue REVISED prefix).
