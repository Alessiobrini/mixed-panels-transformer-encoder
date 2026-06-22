# Reproducing the exhibits in the paper

Every table and figure in the paper regenerates from the small committed bundle in
`replication/data/` (~1.2 MB), with no model retraining. The ten tables reproduce
**byte-for-byte** against the blocks in `paper/main.tex`; the figures regenerate from
byte-identical underlying data (the PDF bytes themselves differ only in matplotlib
metadata, which is unavoidable).

## Requirements
- The `tsa-dev` conda env (Python 3.11): `pandas`, `numpy`, `scikit-learn`, `arch`
  (Model Confidence Set), `matplotlib`. A LaTeX install is needed for figure text.

## Bundle contents (`replication/data/`)
- `experiments/` — the 78 per-target/variant folders (13 targets x {MPTE, AB1-AB5}),
  each holding only the prediction CSVs the exhibits read (~570 KB total).
- `sim_replications_long.csv` — the per-replication simulation metrics behind Table 1
  (100 replications x 3 regimes; ~290 KB).
- `attention/` — compact mean attention matrices (`overall_mean.Ax`, `overall_mean.B`,
  axis labels) for the GDPC1/OUTNFB x {MPTE, AB1} heatmaps (~260 KB; extracted from the
  17 MB-per-file raw dumps, which stay on the cluster).

## Commands
Run from the repo root. `<OUT>` is any output directory.

1. **Tables 2-10** (FRED competing, ablation, win-counts, Diebold-Mariano, MCS):
   ```
   python src/evaluation/build_empirical_tables.py \
       --experiment-date 2026-06-12_lead2 \
       --data-dir replication/data/experiments --outdir <OUT>
   ```
2. **Table 1** (simulation, mean (SD) over replications):
   ```
   python src/evaluation/aggregate_replications.py \
       --from-long replication/data/sim_replications_long.csv \
       --out-prefix repl100_lead2 --outdir <OUT>
   ```
3. **Figure: out-of-sample forecasts** (`Fig:preds`):
   ```
   python src/evaluation/plot_forecasts_paper.py \
       --experiment-date 2026-06-12_lead2 \
       --data-dir replication/data/experiments --outdir paper/figs
   ```
4. **Figures: attention heatmaps** (`Fig:Az_*`, `Fig:B_*`):
   ```
   python src/evaluation/plot_attention_heatmaps.py \
       --data-dir replication/data/attention --outdir paper/figs
   ```

The generated `<OUT>/{empirical1,empirical2,empirical3,empirical_abl1,empirical_abl2,
dm_competing,mcs_competing,dm_ablations,mcs_ablations}.tex` and
`<OUT>/repl100_lead2_replications.tex` are byte-identical to the corresponding
`\begin{table}...\end{table}` blocks in `paper/main.tex`.

## Static (hand-written) exhibits
The hyperparameter search-space tables, the monthly/quarterly variable lists, and the
wide-panel-vs-long-sequence TikZ diagram are static LaTeX in `paper/main.tex`; they have
no data dependency.

## Full end-to-end (optional, heavy)
To regenerate the prediction data itself from scratch (rather than using the committed
bundle), retrain with the committed code and configs:
`src/train.py` / `src/univariate_targets.sh` (empirical) and
`src/run_simulation_replications.py` (simulation). This requires the cluster; the bundle
above exists so reviewers can skip it.
