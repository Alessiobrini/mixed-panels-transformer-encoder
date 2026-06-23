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

## Modifying the package for a future review

**Golden rule:** the table `.tex` blocks and the figure PDFs are *generated*. Never hand-edit
them in `paper/main.tex` or `paper/figs/`. Edit the generator, rerun it, then swap the new
block into `main.tex` (see the rebuild loop below). Otherwise the next regeneration silently
reverts the manual edit (which is exactly the failure mode we hit earlier).

### Changes that need NO retraining (edit one constant, rerun a generator)
These use the committed prediction CSVs, so they are instant. All constants are at the top of
the file named.

| Reviewer asks to change... | Edit | In |
|---|---|---|
| Which targets are in the win vs lose tables | `PAPER_WIN` / `PAPER_LOSE` (or pass `--regroup` to recompute from data) | `build_empirical_tables.py` |
| Which competing / ablation models are shown or counted | `COMPETING`, `ABLATION_ROWS` (tables), `ROW_ORDER` (sim table) | `build_empirical_tables.py`, `aggregate_replications.py` |
| The metrics reported | `METRICS` | both table generators |
| The COVID subsample cutoff | `COVID_CUT` | `build_empirical_tables.py` **and** `plot_forecasts_paper.py` |
| Table font size / column spacing | the `fontsize` arg / the `\setlength{\tabcolsep}` wrap | `build_empirical_tables.py`, `aggregate_replications.py` |
| Models drawn in the forecast plots | `PLOT_MODELS` | `plot_forecasts_paper.py` |
| Which targets/variants get heatmaps | `SOURCES` (and add the matching compact file under `replication/data/attention/`) | `plot_attention_heatmaps.py` |

### Changes that DO need new data (retrain on the cluster, then drop into the bundle)
- **A new target or a new ablation/model variant.** Train it (`src/train.py` /
  `src/univariate_targets.sh` / `src/run_simulation_replications.py`), copy only its
  `*_preds_*.csv` into `replication/data/experiments/<TARGET>_<date>[ _<scen>]/`, add it to
  `TARGETS` and the relevant model list, and rerun. For a new heatmap, also extract its
  `overall_mean.{Ax,B}` into a compact JSON (the extraction snippet is in the commit that
  added `replication/data/attention/`).
- **A fresh re-run of everything** (new seed, revised DGP): rerun the pipeline, refresh the
  bundle, and pass the new `--experiment-date`.

### The rebuild loop (after any generator change)
```
# 1. regenerate the affected exhibit(s)
python src/evaluation/build_empirical_tables.py --experiment-date 2026-06-12_lead2 \
    --data-dir replication/data/experiments --outdir outputs/tables
# 2. swap the new block into main.tex (matches on the \label, replaces the enclosing table)
python - <<'PY'
from pathlib import Path
m=Path("paper/main.tex"); t=m.read_text()
for lab,fn in [("Tab:empirical1","empirical1.tex")]:        # list the blocks you changed
    g=Path(f"outputs/tables/{fn}").read_text(); gb=g[g.index(r"\begin{table}"):g.rindex(r"\end{table}")+11]
    li=t.index("\\label{"+lab+"}"); b=t.rindex(r"\begin{table}",0,li); e=t.index(r"\end{table}",li)+11
    t=t[:b]+gb+t[e:]
m.write_text(t)
PY
# 3. recompile and visually QA, then commit + push (pull Overleaf first)
```
`revision_exhibits/insert_revised_into_paper.py` is an older helper for the same job.

## Full end-to-end (optional, heavy)
To regenerate the prediction data itself from scratch (rather than using the committed
bundle), retrain with the committed code and configs:
`src/train.py` / `src/univariate_targets.sh` (empirical) and
`src/run_simulation_replications.py` (simulation). This requires the cluster; the bundle
above exists so reviewers can skip it.
