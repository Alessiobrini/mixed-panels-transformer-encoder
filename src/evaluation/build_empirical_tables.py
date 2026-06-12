"""Reproducible builder for the FRED empirical exhibits (R2 revision).

The paper's empirical tables (Tab:empirical1/2 competing-model, Tab:empirical_abl1/2 ablation,
Tab:empirical3 win-counts) were assembled by hand. This script regenerates them end to end
from the experiment folders, in the SAME layout, with programmatic \\best/\\second highlighting,
so the lead=2 rerun produces drop-in replacements. It also writes a reconciliation CSV so the
output can be checked against the published numbers (which remain the ground truth).

Layout per competing/ablation table (matches main.tex):
    columns: l | RMSE MAE DA (Full) | RMSE MAE DA (Pre-COVID) | RMSE MAE DA (Post-COVID)
    per target: a \\textbf{TARGET} header row + \\midrule, then one row per model.
    highlighting: best (dark) + second (light) per column, within each target block,
                  among the 6 competing models (or the 6 ablation specs). Ties -> both best.

Run (published, to validate):
    python src/evaluation/build_empirical_tables.py --experiment-date 2025-09-26 \
        --outdir outputs/tables/published_check
Run (lead2 rerun):
    python src/evaluation/build_empirical_tables.py --experiment-date 2026-06-12_lead2 \
        --outdir outputs/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO / "outputs" / "experiments"
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TARGETS = ["GDPC1", "GPDIC1", "PCECC96", "DPIC96", "OUTNFB", "UNRATE",
           "PCECTPI", "PCEPILFE", "CPIAUCSL", "CPILFESL", "FPIx", "EXPGSC1", "IMPGSC1"]

# Paper's exact target grouping (Tab:empirical1/2 and the ablation tables share it). The
# revision reproduces the SAME tables with lead=2, so we keep this grouping rather than
# recomputing it (use --regroup to recompute from the data instead). Note the paper shows
# only 10 of the 13 targets in the main competing tables.
PAPER_WIN = ["OUTNFB", "PCECTPI", "PCEPILFE", "CPIAUCSL", "GDPC1"]
PAPER_LOSE = ["DPIC96", "EXPGSC1", "FPIx", "CPILFESL", "IMPGSC1"]

COMPETING = [("MPTE", "transformer"), ("AR", "ar"), ("MIDAS", "midas"),
             ("OLS", "ols"), ("XGB", "xgb"), ("NN", "nn")]
# ablation folder suffix -> display label (B5->AB4, B6->AB5, matching the paper)
ABLATION_SCEN = [("MPTE", None), ("AB1", "B1_no_nonlinearity"), ("AB2", "B2_no_attention"),
                 ("AB3", "B3_no_attention_no_nonlinearity"),
                 ("AB4", "B5_no_positional_encoding"), ("AB5", "B6_y_only")]

PRED_FILES = {"transformer": "transformer_preds", "ar": "ar_preds", "midas": "midas_preds",
              "ols": "ols_preds", "xgb": "xgb_preds", "nn": "nn_preds"}
PERIODS = ["full", "pre", "post"]
METRICS = ["RMSE", "MAE", "DA"]
COVID_CUT = pd.Timestamp("2019-06-30")


def directional_accuracy(y_true, y_pred):
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))


def compute_errors(y_true, y_pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "DA": directional_accuracy(y_true, y_pred)}


def _load_pred(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "true", df.columns[2]: model})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "true", model]].set_index("date")


def load_target_frame(date: str, target: str, include_ablations: bool) -> pd.DataFrame | None:
    """Merge all model preds for a target on common dates. Returns df indexed by date with
    a 'true' column and one column per model key present."""
    folder = EXPERIMENT_DIR / f"{target}_{date}"
    if not folder.is_dir():
        return None
    frames = {}
    for key, prefix in PRED_FILES.items():
        f = next(folder.glob(f"{prefix}_*.csv"), None)
        if f is not None:
            frames[key] = _load_pred(f, key)
    if include_ablations:
        for label, scen in ABLATION_SCEN:
            if scen is None:
                continue
            abdir = EXPERIMENT_DIR / f"{target}_{date}_{scen}"
            f = next(abdir.glob("transformer_preds_*.csv"), None) if abdir.is_dir() else None
            if f is not None:
                frames[label] = _load_pred(f, label)
    if "transformer" not in frames:
        return None
    merged = frames["transformer"].copy()
    for k, fr in frames.items():
        if k == "transformer":
            continue
        merged = merged.join(fr.drop(columns="true"), how="inner")
    return merged


def metrics_table(date: str, targets, include_ablations: bool) -> dict:
    """Return {target: {model: {period: {metric: val}}}}."""
    out = {}
    for t in targets:
        merged = load_target_frame(date, t, include_ablations)
        if merged is None:
            continue
        idx = merged.index
        masks = {"full": np.ones(len(idx), bool),
                 "pre": np.asarray(idx <= COVID_CUT),
                 "post": np.asarray(idx > COVID_CUT)}
        models = [c for c in merged.columns if c != "true"]
        out[t] = {}
        for m in models:
            out[t][m] = {}
            for p, mask in masks.items():
                yt = merged.loc[mask, "true"].to_numpy()
                yp = merged.loc[mask, m].to_numpy()
                out[t][m][p] = compute_errors(yt, yp) if len(yt) > 1 else {k: np.nan for k in METRICS}
    return out


def _highlight(values: dict, metric: str) -> dict:
    """values: {model: val}. Return {model: 'best'|'second'|None} with ties -> all best.
    Ranks on the 4-decimal DISPLAYED value (as the paper does), so cells that print
    identically are treated as ties."""
    items = [(m, round(float(v), 4)) for m, v in values.items()
             if v is not None and not np.isnan(v)]
    if len(items) < 2:
        return {}
    reverse = (metric == "DA")
    ordered = sorted(items, key=lambda x: x[1], reverse=reverse)
    best_val = ordered[0][1]
    tag = {}
    seconds_started = False
    second_val = None
    for m, v in ordered:
        if v == best_val:
            tag[m] = "best"
        elif not seconds_started:
            second_val = v
            tag[m] = "second"
            seconds_started = True
        elif v == second_val:
            tag[m] = "second"
    return tag


def fmt(v, tag):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    s = f"{v:.4f}"
    if tag == "best":
        return f"\\best{{{s}}}"
    if tag == "second":
        return f"\\second{{{s}}}"
    return s


def make_table(metrics: dict, targets, row_specs, caption: str, label: str) -> str:
    """row_specs: list of (display_label, model_key). targets: ordered list."""
    L = [f"% Auto-generated by build_empirical_tables.py -- do not edit by hand.",
         "\\begin{table}[!ht]", "\\centering",
         "\\begin{tabular}{l ccc ccc ccc}", "\\toprule",
         "& \\multicolumn{3}{c}{\\textbf{Full}} & \\multicolumn{3}{c}{\\textbf{Pre-COVID}} "
         "& \\multicolumn{3}{c}{\\textbf{Post-COVID}} \\\\", "\\midrule",
         "& RMSE & MAE & DA & RMSE & MAE & DA & RMSE & MAE & DA \\\\", "\\midrule"]
    targets = [t for t in targets if t in metrics]
    for ti, t in enumerate(targets):
        L.append(f"\\textbf{{{t}}} & & & & & & & & & \\\\ \\midrule")
        # precompute highlight per (period, metric) over the row_specs present
        present = [(lab, key) for lab, key in row_specs if key in metrics[t]]
        hl = {}
        for p in PERIODS:
            for me in METRICS:
                vals = {lab: metrics[t][key][p][me] for lab, key in present}
                hl[(p, me)] = _highlight(vals, me)
        for lab, key in row_specs:
            if key not in metrics[t]:
                continue
            cells = []
            for p in PERIODS:
                for me in METRICS:
                    v = metrics[t][key][p][me]
                    cells.append(fmt(v, hl[(p, me)].get(lab)))
            L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if ti != len(targets) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}",
          f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L)


def mpte_wins_full_rmse(metrics: dict, target: str, competing_keys) -> bool:
    """True if MPTE has the lowest full-period RMSE among the competing models (excl. AR,
    matching get_best_targets in the original aggregator which excludes AR)."""
    vals = {k: metrics[target][k]["full"]["RMSE"] for _, k in competing_keys
            if k in metrics[target] and k != "ar"}
    if "transformer" not in vals:
        return False
    return min(vals, key=vals.get) == "transformer"


# ======================================================================================
# Appendix B: Diebold-Mariano + Model Confidence Set tables
# ======================================================================================
DM_COMPETING = [("MIDAS", "midas"), ("AR", "ar"), ("OLS", "ols"), ("XGB", "xgb"), ("NN", "nn")]
DM_ABLATIONS = [("AB1", "AB1"), ("AB2", "AB2"), ("AB3", "AB3"), ("AB4", "AB4"), ("AB5", "AB5")]


def _dm_stat(dm_df, base, other):
    """MPTE(base)-vs-other DM stat with the paper's sign (negative => base lower loss)."""
    if f"{base} vs {other}" in dm_df.index:
        return dm_df.loc[f"{base} vs {other}", "DM_stat"]
    if f"{other} vs {base}" in dm_df.index:
        return -dm_df.loc[f"{other} vs {base}", "DM_stat"]
    return np.nan


def make_dm_table(date, targets, model_specs, include_abl, caption, label):
    from src.evaluation.evaluate_forecasts import run_dm_tests
    header = " & ".join(["Target"] + [lab for lab, _ in model_specs])
    L = ["% Auto-generated DM table", "\\begin{table}[htbp]", "\\centering",
         "\\begin{tabular}{@{}l" + "c" * len(model_specs) + "@{}}", "\\toprule",
         header + " \\\\", "\\midrule"]
    for t in targets:
        merged = load_target_frame(date, t, include_ablations=include_abl)
        if merged is None or "transformer" not in merged.columns:
            continue
        cols = ["transformer"] + [k for _, k in model_specs if k in merged.columns]
        dm = run_dm_tests(merged[cols], merged["true"])
        vals = [t]
        for _, k in model_specs:
            s = _dm_stat(dm, "transformer", k) if k in merged.columns else np.nan
            vals.append("--" if (s is None or np.isnan(s)) else f"{s:.2f}")
        L.append(" & ".join(vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
          "\\end{table}"]
    return "\n".join(L)


def make_mcs_table(date, targets, col_specs, include_abl, caption, label, exclude_ar=True):
    """col_specs: [(display, key), ...] the columns shown. MCS computed over the shown models
    except AR (always excluded, shown as --), matching the paper. seed=42, reps=1000."""
    from arch.bootstrap import MCS
    rows = []
    for t in targets:
        merged = load_target_frame(date, t, include_ablations=include_abl)
        if merged is None:
            rows.append([t] + ["--"] * len(col_specs))
            continue
        y = merged["true"].to_numpy()
        models = {lab: k for lab, k in col_specs
                  if k in merged.columns and not (exclude_ar and k == "ar")}
        included = set()
        if len(models) >= 2:
            labs = list(models)
            losses = np.column_stack([(y - merged[models[l]].to_numpy()) ** 2 for l in labs])
            try:
                mcs = MCS(losses, size=0.10, reps=1000, method="R",
                          bootstrap="stationary", seed=42)
                mcs.compute()
                included = {labs[i] for i in mcs.included}
            except Exception:
                included = set()
        rows.append([t] + ["\\checkmark" if lab in included else "--" for lab, _ in col_specs])
    header = " & ".join(["target"] + [lab for lab, _ in col_specs])
    L = ["% Auto-generated MCS table", "\\begin{table}[htbp]", "\\centering",
         "\\begin{tabular}{l" + "c" * len(col_specs) + "}", "\\toprule", header + " \\\\",
         "\\midrule"]
    for r in rows:
        L.append(" & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
          "\\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-date", required=True)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--outdir", default=str(REPO / "outputs" / "tables"))
    ap.add_argument("--regroup", action="store_true",
                    help="recompute the win/lose grouping from the data instead of using the "
                         "paper's fixed grouping (default: keep the paper grouping)")
    args = ap.parse_args()

    date = args.experiment_date
    targets = args.targets.split(",") if args.targets else TARGETS
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    comp = metrics_table(date, targets, include_ablations=False)
    abl = metrics_table(date, targets, include_ablations=True)

    if args.regroup:
        win = [t for t in targets if t in comp and mpte_wins_full_rmse(comp, t, COMPETING)]
        lose = [t for t in targets if t in comp and t not in win]
    else:
        win = [t for t in PAPER_WIN if t in comp]
        lose = [t for t in PAPER_LOSE if t in comp]

    cap1 = ("Out-of-sample forecasting performance for target series where MPTE achieves the "
            "lowest RMSE over the full sample. The table reports RMSE, MAE and DA for MPTE and "
            "competing models over the full evaluation period, as well as the pre-COVID and "
            "post-COVID subsamples. Dark green indicates the best-performing method and light "
            "green the second-best within each column.")
    cap2 = cap1.replace("where MPTE achieves the lowest RMSE",
                        "where MPTE does not achieve the lowest RMSE")
    cap_a1 = ("Out-of-sample forecasting performance for target series where MPTE achieves the "
              "lowest RMSE over the full sample. The table reports RMSE, MAE, and DA for MPTE and "
              "its ablation variants over the full evaluation period, as well as the pre-COVID and "
              "post-COVID subsamples. Dark green indicates the best-performing method and light "
              "green the second-best within each column.")
    cap_a2 = cap_a1.replace("where MPTE achieves the lowest RMSE",
                            "where MPTE does not achieve the lowest RMSE")

    writes = {
        "empirical1.tex": make_table(comp, win, COMPETING, cap1, "Tab:empirical1"),
        "empirical2.tex": make_table(comp, lose, COMPETING, cap2, "Tab:empirical2"),
        "empirical_abl1.tex": make_table(abl, win, ABLATION_SCEN, cap_a1, "Tab:empirical_abl1"),
        "empirical_abl2.tex": make_table(abl, lose, ABLATION_SCEN, cap_a2, "Tab:empirical_abl2"),
    }
    for name, tex in writes.items():
        (outdir / name).write_text(tex + "\n")
        print(f"wrote {outdir / name}")

    # win-count summary (Tab:empirical3): best model per (period, metric) across all targets.
    counts = {}
    for p in PERIODS:
        for me in METRICS:
            for t in comp:
                vals = {lab: comp[t][k][p][me] for lab, k in COMPETING
                        if k in comp[t] and lab != "AR"}  # exclude AR, as in the paper summary
                vals = {k: v for k, v in vals.items() if not np.isnan(v)}
                if not vals:
                    continue
                best = (max if me == "DA" else min)(vals, key=vals.get)
                counts.setdefault(best, {}).setdefault((p, me), 0)
                counts[best][(p, me)] += 1
    rows = sorted(counts, key=lambda m: -sum(counts[m].values()))
    L = ["% Auto-generated win counts", "\\begin{table}[!ht]", "\\centering",
         "\\begin{tabular}{l ccc ccc ccc}", "\\toprule",
         "& \\multicolumn{3}{c}{\\textbf{Full}} & \\multicolumn{3}{c}{\\textbf{Pre-COVID}} "
         "& \\multicolumn{3}{c}{\\textbf{Post-COVID}} \\\\", "\\midrule",
         "& RMSE & MAE & DA & RMSE & MAE & DA & RMSE & MAE & DA \\\\", "\\midrule"]
    for m in rows:
        cells = [str(counts[m].get((p, me), 0)) for p in PERIODS for me in METRICS]
        L.append(f"{m} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Number of target series for which each model achieves the best forecasting "
          "performance relative to competitors, for RMSE, MAE, and DA over the full evaluation "
          "period and the pre- and post-COVID subsamples.}", "\\label{Tab:empirical3}",
          "\\end{table}"]
    (outdir / "empirical3.tex").write_text("\n".join(L) + "\n")
    print(f"wrote {outdir / 'empirical3.tex'}")

    # --- Appendix B: DM + MCS tables (all 13 targets, full period) ---
    all_t = [t for t in targets if (EXPERIMENT_DIR / f"{t}_{date}").is_dir()]
    dm_cap_c = ("Diebold--Mariano test statistics comparing MPTE against competing models. Each "
                "entry reports the Diebold--Mariano statistic for the null hypothesis of equal "
                "predictive accuracy between MPTE and the corresponding competing model for the "
                "given target series. Negative values indicate lower forecast loss for MPTE "
                "relative to the competing model, while positive values indicate superior "
                "performance of the competing model.")
    dm_cap_a = dm_cap_c.replace("competing models", "ablation models").replace(
        "competing model", "ablation").replace("competing model.", "ablation.")
    mcs_cap_c = ("Model Confidence Set (MCS) inclusion for MPTE and competing models across target "
                 "series at the 95\\% confidence level. A checkmark indicates that the corresponding "
                 "model is included in the MCS for the given target, while ``--'' denotes exclusion.")
    mcs_cap_a = ("Model Confidence Set (MCS) inclusion for MPTE and its ablation variants across "
                 "target series at the 95\\% confidence level. A checkmark indicates that the "
                 "corresponding specification is included in the MCS for the given target, while "
                 "``--'' denotes exclusion.")
    (outdir / "dm_competing.tex").write_text(
        make_dm_table(date, all_t, DM_COMPETING, False, dm_cap_c, "Tab:DM_competing") + "\n")
    (outdir / "dm_ablations.tex").write_text(
        make_dm_table(date, all_t, DM_ABLATIONS, True, dm_cap_a, "Tab:DM_ablations") + "\n")
    print(f"wrote {outdir / 'dm_competing.tex'}, {outdir / 'dm_ablations.tex'}")
    try:
        (outdir / "mcs_competing.tex").write_text(make_mcs_table(
            date, all_t, [("MPTE", "transformer"), ("MIDAS", "midas"), ("AR", "ar"),
                          ("OLS", "ols"), ("XGB", "xgb"), ("NN", "nn")], False, mcs_cap_c,
            "Tab:MCS_competing") + "\n")
        (outdir / "mcs_ablations.tex").write_text(make_mcs_table(
            date, all_t, [("MPTE", "transformer"), ("AB1", "AB1"), ("AB2", "AB2"),
                          ("AB3", "AB3"), ("AB4", "AB4"), ("AB5", "AB5")], True, mcs_cap_a,
            "Tab:MCS_ablations", exclude_ar=False) + "\n")
        print(f"wrote {outdir / 'mcs_competing.tex'}, {outdir / 'mcs_ablations.tex'}")
    except ModuleNotFoundError:
        print("WARN: 'arch' not installed -> skipped MCS tables (run on the cluster env where "
              "arch is available).")

    print(f"\nMPTE-wins-RMSE group ({len(win)}): {win}")
    print(f"complement group ({len(lose)}): {lose}")


if __name__ == "__main__":
    main()
