# theorysim (JBES theory-validation add-on) -- E1 full rerun on the relative metric.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Rerun E1 across a large (T,N) grid (up to 1000) on the scale-invariant relative metric,
for the three paper-grounded operator arms (oracle, parameter_free, independent/learned).

Primary diagnostic: log-log slope of mean relative e_C vs N (oracle ~ -1 validates Theorem 1).
Writes a CSV for the report and prints a summary table with slopes and op-norm growth.

Usage: PYTHONPATH=<repo> python -m src.theorysim.rerun_e1_relative [--reps 100] [--epochs 800]
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

from src.theorysim import dgp  # noqa: F401 (kept for parity / future use)
from src.theorysim.exp_e1_consistency import run_cell

warnings.filterwarnings("ignore")
np.seterr(all="ignore")


def _slope(xs, ys):
    A = np.vstack([np.log(xs), np.ones_like(xs, dtype=float)]).T
    return float(np.linalg.lstsq(A, np.log(ys), rcond=None)[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--grid", type=int, nargs="*", default=[80, 160, 320, 640, 1000])
    ap.add_argument("--arms", nargs="*", default=["oracle", "parameter_free", "independent"])
    ap.add_argument("--out", default="outputs/theorysim/e1/relative_rerun.csv")
    args = ap.parse_args()

    dims = dict(k_ys=2, k_R=2, sigma2=2.0)
    k = dims["k_ys"] + dims["k_R"]
    grid = [(g, g) for g in args.grid]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {}
    for arm in args.arms:
        Ns, rels, abss, ops = [], [], [], []
        for (T, N) in grid:
            cell = run_cell(arm, T, N, args.reps, dims, k, 16, epochs=args.epochs)
            rel = float(np.mean([c["e_C_rel"] for c in cell]))
            ab = float(np.mean([c["e_C"] for c in cell]))
            op = float(np.mean([c["op_norm"] for c in cell]))
            Ns.append(N); rels.append(rel); abss.append(ab); ops.append(op)
            rows.append(dict(arm=arm, N=N, T=T, e_C_rel=rel, e_C_abs=ab, op_norm=op,
                             alpha_bar=cell[0]["alpha_bar"], reps=args.reps))
            print(f"[{arm:14s}] N={N:4d}  rel_eC={rel:.4f}  op_norm={op:6.2f}", flush=True)
        slope = _slope(np.array(Ns, float), np.array(rels))
        summary[arm] = slope
        print(f"==> {arm:14s} slope(relative e_C ~ N) = {slope:.3f}\n", flush=True)

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("SLOPES:", {a: round(s, 3) for a, s in summary.items()})
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
