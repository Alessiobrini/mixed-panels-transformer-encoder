"""Reproducible source for the empirical out-of-sample forecast plots (Fig:preds).

These figures previously had NO committed source script (the project rule requires one). This
reproduces them from the prediction CSVs in the experiment folders, in the SAME style as the
in-module plotting block of aggregate_univariate_results.py (Actual k--, MPTE #003f5c, MIDAS
#bc6c25, XGB #4c956c; pre-COVID cutoff 2019-06-30; usetex serif; figsize (10, 4) to match the
original look so the REVISED figures are directly comparable to the old ones).

Writes, for each target, {TARGET}_preds.pdf (pre-COVID window) and {TARGET}_preds_full.pdf
(full sample) into <outdir>/<FOLDER>/, where FOLDER matches the paths in main.tex
(GDPC1 -> figs/GDPC1, OUTNFB -> figs/OUTFNB -- the existing typo folder).

Run (lead2):
    python src/evaluation/plot_forecasts_paper.py --experiment-date 2026-06-12_lead2 \
        --outdir paper/figs --reference-outdir outputs/revision_exhibits/figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO / "outputs" / "experiments"

# target -> figs subfolder name used in main.tex (note the OUTFNB typo folder for OUTNFB).
FOLDER = {"GDPC1": "GDPC1", "OUTNFB": "OUTFNB"}
COVID_CUT = pd.Timestamp("2019-06-30")
PLOT_MODELS = ["transformer", "midas", "xgb"]
LABELS = {"transformer": "MPTE", "midas": "MIDAS", "xgb": "XGB"}
STYLES = {"transformer": {"color": "#003f5c", "linewidth": 2.2, "alpha": 1.0},
          "midas": {"color": "#bc6c25", "linewidth": 1.4, "alpha": 0.85},
          "xgb": {"color": "#4c956c", "linewidth": 1.4, "alpha": 0.85}}
LEGEND_LOC = {"GDPC1": "lower left", "OUTNFB": "lower left"}
PRED_FILES = {"transformer": "transformer_preds", "midas": "midas_preds", "xgb": "xgb_preds"}


def _load(folder: Path, prefix: str, name: str):
    f = next(folder.glob(f"{prefix}_*.csv"), None)
    if f is None:
        return None
    df = pd.read_csv(f)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "true", df.columns[2]: name})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "true", name]].set_index("date")


def merged_frame(date: str, target: str):
    folder = EXPERIMENT_DIR / f"{target}_{date}"
    if not folder.is_dir():
        return None
    base = _load(folder, PRED_FILES["transformer"], "transformer")
    if base is None:
        return None
    merged = base.copy()
    for m in ["midas", "xgb"]:
        fr = _load(folder, PRED_FILES[m], m)
        if fr is not None:
            merged = merged.join(fr.drop(columns="true"), how="inner")
    return merged.reset_index()


def plot_one(merged, target, pre_covid_only: bool, out_paths):
    import matplotlib.pyplot as plt
    df = merged.copy()
    if pre_covid_only:
        df = df[df["date"] <= COVID_CUT]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["true"], "k--", label="Actual")
    for m in PLOT_MODELS:
        if m in df.columns:
            ax.plot(df["date"], df[m], label=LABELS[m], linestyle="-", **STYLES[m])
    ax.set_title(target)
    ax.legend(loc=LEGEND_LOC.get(target, "best"))
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-date", required=True)
    ap.add_argument("--targets", default="GDPC1,OUTNFB")
    ap.add_argument("--outdir", default=str(REPO / "paper" / "figs"),
                    help="primary output (the paper's figs/ tree)")
    ap.add_argument("--reference-outdir", default=None,
                    help="optional second copy for the reference comparison PDF")
    ap.add_argument("--no-usetex", action="store_true", help="disable LaTeX text rendering")
    args = ap.parse_args()

    if not args.no_usetex:
        mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"
    for k in ["font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize"]:
        mpl.rcParams[k] = 14

    targets = args.targets.split(",")
    outdir = Path(args.outdir)
    refdir = Path(args.reference_outdir) if args.reference_outdir else None

    for t in targets:
        merged = merged_frame(args.experiment_date, t)
        if merged is None:
            print(f"WARN: no preds for {t}_{args.experiment_date}; skipping")
            continue
        sub = FOLDER.get(t, t)
        for pre, tag in [(True, "preds"), (False, "preds_full")]:
            outs = [outdir / sub / f"{t}_{tag}.pdf"]
            if refdir:
                outs.append(refdir / sub / f"{t}_{tag}.pdf")
            plot_one(merged, t, pre, outs)
            print(f"wrote {outs[0]}")


if __name__ == "__main__":
    main()
