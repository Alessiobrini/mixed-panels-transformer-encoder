"""Aggregate Monte Carlo replications (referee R2-12) into a mean +/- SD table.

Reads a manifest written by src/run_simulation_replications.py, recomputes
RMSE/MAE/DA per (regime, variant, seed) with the same definitions used for the
published Table 1 (reusing load_folder_metrics from build_simulation_table), and
emits the mean +/- SD across replications per (regime x model x metric) as both a
CSV and a combined LaTeX table for the paper.

Baselines (AR, MIDAS) are read from the 'full' variant folder of each replication
(they are identical across variants for a given path). MPTE is the 'full' variant's
transformer; AB1..AB5 are the ablation variants' transformer.

Usage:
    python src/evaluation/aggregate_replications.py --manifest outputs/replications/repl_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.build_simulation_table import load_folder_metrics  # noqa: E402

REGIME_ORDER = ["linear", "mild", "high"]
REGIME_LABEL = {"linear": "Linear", "mild": "Mildly Nonlinear", "high": "Highly Nonlinear"}
ROW_ORDER = ["MPTE", "AR", "MIDAS", "AB1", "AB2", "AB3", "AB4", "AB5"]
METRICS = ["RMSE", "MAE", "DA"]

# DISPLAY-only relabel of the ablation variants. The manifest 'variant' column still carries the
# OLD label assignment from run_simulation_replications.VARIANTS (variant "AB3" = the both-off
# folder, "AB5" = the y-only folder); the paper's current AB definitions require the 3-cycle
# AB3->AB4, AB4->AB5, AB5->AB3. MPTE/AR/MIDAS/AB1/AB2/full are unchanged. On-disk folder names
# and manifest contents are NOT touched; only the emitted output-row label is remapped.
VARIANT_DISPLAY = {"AB3": "AB4", "AB4": "AB5", "AB5": "AB3"}


def collect(manifest: Path) -> pd.DataFrame:
    """Return long DataFrame: regime, sim_seed, init_seed, model, metric, value."""
    man = pd.read_csv(manifest)
    rows = []
    for _, r in man.iterrows():
        folder = Path(r["exp_folder"])
        if not folder.is_dir():
            print(f"WARN missing folder: {folder}")
            continue
        m = load_folder_metrics(folder)
        variant = r["variant"]

        # MPTE + baselines come from the 'full' folder; ablations contribute their transformer.
        emit = {}
        if variant == "full":
            if "transformer" in m:
                emit["MPTE"] = m["transformer"]
            if "ar" in m:
                emit["AR"] = m["ar"]
            if "midas" in m:
                emit["MIDAS"] = m["midas"]
        else:
            if "transformer" in m:
                emit[VARIANT_DISPLAY.get(variant, variant)] = m["transformer"]

        for model, metrics in emit.items():
            for metric in METRICS:
                rows.append({
                    "regime": r["regime"], "sim_seed": r["sim_seed"], "init_seed": r["init_seed"],
                    "model": model, "metric": metric, "value": metrics[metric],
                })
    return pd.DataFrame(rows)


def drop_diverged(long_df: pd.DataFrame, max_rmse: float) -> pd.DataFrame:
    """Drop entire (regime, seed) replications whose path is pathological -- any model's RMSE
    exceeds max_rmse. On rare seeds the simulated path is (near-)explosive, so EVERY model
    (incl. the deterministic AR) blows up; those are bad DGP draws, not model failures, and
    excluding the whole seed keeps the per-seed comparison balanced."""
    rmse = long_df[long_df["metric"] == "RMSE"]
    bad = set(map(tuple, rmse[rmse["value"] > max_rmse][["regime", "sim_seed"]]
                  .drop_duplicates().values))
    if bad:
        print(f"Dropping {len(bad)} diverged (regime, seed) replications (RMSE > {max_rmse}, "
              f"pathological/explosive paths):")
        for reg, seed in sorted(bad):
            print(f"  {reg} seed {seed}")
        key = list(zip(long_df["regime"], long_df["sim_seed"]))
        long_df = long_df[[k not in bad for k in key]].copy()
    return long_df


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby(["regime", "model", "metric"])["value"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def make_latex(summary: pd.DataFrame) -> str:
    piv = {}
    for _, r in summary.iterrows():
        piv[(r["regime"], r["model"], r["metric"])] = (r["mean"], r["std"], int(r["count"]))

    # best & second-best mean per (regime, metric): min for RMSE/MAE, max for DA
    best, second = {}, {}
    for regime in REGIME_ORDER:
        for metric in METRICS:
            vals = [(model, piv[(regime, model, metric)][0])
                    for model in ROW_ORDER if (regime, model, metric) in piv]
            if not vals:
                continue
            ordered = sorted(vals, key=lambda x: x[1], reverse=(metric == "DA"))
            best[(regime, metric)] = ordered[0][0]
            if len(ordered) > 1:
                second[(regime, metric)] = ordered[1][0]

    n = max((piv[k][2] for k in piv), default=0)
    # Layout matches the published Table 1 font size (no \resizebox). Each model spans two
    # rows: the mean (with best/second-best cell highlight) on top, and the SD in \scriptsize
    # parentheses directly below, so the table reads like Table 1 with the spread underneath.
    L = ["% Auto-generated by src/evaluation/aggregate_replications.py",
         f"% Mean over {n} Monte Carlo replications, with SD on the second line per cell.",
         "\\begin{table}[h!]", "\\centering",
         "{\\scriptsize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lccccccccc}", "\\toprule",
         "& \\multicolumn{3}{c}{\\textbf{Linear}} & \\multicolumn{3}{c}{\\textbf{Mildly Nonlinear}}"
         " & \\multicolumn{3}{c}{\\textbf{Highly Nonlinear}} \\\\", "\\midrule",
         "& RMSE & MAE & DA & RMSE & MAE & DA & RMSE & MAE & DA \\\\", "\\midrule"]
    for model in ROW_ORDER:
        mean_cells, std_cells = [], []
        for regime in REGIME_ORDER:
            for metric in METRICS:
                key = (regime, model, metric)
                if key not in piv:
                    mean_cells.append("--")
                    std_cells.append("")
                    continue
                mean, std, _ = piv[key]
                m = f"{mean:.4f}"
                if best.get((regime, metric)) == model:
                    m = f"\\best{{{m}}}"
                elif second.get((regime, metric)) == model:
                    m = f"\\second{{{m}}}"
                mean_cells.append(m)
                std_cells.append(f"{{\\scriptsize ({std:.4f})}}")
        L.append(f"{model}\n& " + " & ".join(mean_cells) + " \\\\")
        L.append("& " + " & ".join(std_cells) + " \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}}",
          "\\caption{Forecasting accuracy for the first low-frequency target series $Y_1$ across "
          "linear, mildly nonlinear, and highly nonlinear simulation designs. Each design includes "
          "30 high-frequency regressors and 5 low-frequency targets, with $Y_1$ predicted from its "
          "own lagged values together with the remaining low- and high-frequency regressors. Results "
          "are reported for MPTE, AR, MIDAS, and ablation variants of "
          f"MPTE. Each entry is the average over {n} replications of the data-generating process, "
          "each generated with a different random seed, with "
          "the standard deviation across those replications reported in parentheses below. "
          "Dark green indicates the best-performing method and light green the second-best "
          "within each column.}", "\\label{Tab:evals_simulation}", "\\end{table}"]
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None, help="path to the *_manifest.csv from the driver")
    p.add_argument("--from-long", default=None,
                   help="path to a saved *_replications_long.csv (the post-collect metrics). "
                        "Rebuilds Table 1 from the committed replication bundle without needing "
                        "the raw per-replication folders. Mutually exclusive with --manifest.")
    p.add_argument("--out-prefix", default=None, help="output basename (default: manifest stem)")
    p.add_argument("--outdir", default=None, help="output directory (default: <manifest>/../tables)")
    p.add_argument("--max-rmse", type=float, default=10.0,
                   help="drop (regime, seed) replications whose path explodes above this RMSE")
    args = p.parse_args()
    if not (args.manifest or args.from_long):
        p.error("provide --manifest or --from-long")

    if args.from_long:
        long_df = pd.read_csv(args.from_long)
        stem = args.out_prefix or Path(args.from_long).stem.replace("_replications_long", "")
        out_dir = Path(args.outdir) if args.outdir else Path(args.from_long).parent
    else:
        manifest = Path(args.manifest)
        long_df = collect(manifest)
        stem = args.out_prefix or manifest.stem.replace("_manifest", "")
        out_dir = Path(args.outdir) if args.outdir else manifest.parent.parent / "tables"
    if long_df.empty:
        print("No metrics collected — check the manifest folders.")
        return

    long_df = drop_diverged(long_df, args.max_rmse)
    summary = summarize(long_df)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = out_dir / f"{stem}_replications_long.csv"
    summ_path = out_dir / f"{stem}_replications_summary.csv"
    tex_path = out_dir / f"{stem}_replications.tex"
    long_df.to_csv(long_path, index=False)
    summary.to_csv(summ_path, index=False)
    tex_path.write_text(make_latex(summary) + "\n")

    print(f"Replications per cell: {sorted(long_df.groupby(['regime','model']).size().unique())}")
    print("\n=== mean (SD) summary ===")
    show = summary.copy()
    show["mean"] = show["mean"].round(4)
    show["std"] = show["std"].round(4)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(show.to_string(index=False))
    print(f"\nSaved:\n  {long_path}\n  {summ_path}\n  {tex_path}")


if __name__ == "__main__":
    main()
