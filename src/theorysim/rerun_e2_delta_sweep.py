# theorysim (JBES theory-validation add-on) -- E2 blend delta-sweep (boundary exhibit).
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Sweep the blend shrinkage delta from 0 (the concentrated clip end) toward large (the blend
converges to the identity/oracle) and record coverage against the realized effective rank. The
x-axis is the participation ratio PR/N, equivalently the middle term of alpha_bar (= 1/PR): as
delta grows, the identity restores effective rank, PR/N rises through the A.7 threshold
c_A/kappa0^2, and inference switches on. This is the boundary exhibit in theorem-native units.

Averaged over operator seeds for a clean curve. Reports the general-plug-in and Monte Carlo
coverage per (regime, delta).

Usage: python -m src.theorysim.rerun_e2_delta_sweep [--reps 1500]
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

from src.theorysim import dgp, operators as op, exp_e2_coverage as e2

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

DIMS = dict(k_ys=2, k_R=0, sigma2=2.0)
REGIMES = {"F_dominant": (50, 400), "Lambda_dominant": (400, 50), "mixed": (200, 200)}


def _learned_wb(N, T, seed, epochs):
    N_y = max(2, round(N / 3))
    dd = dict(DIMS, T=T, N_x=N - N_y, N_y=N_y)
    tr = dgp.draw(seed=seed, dims=dd)
    m = op.train_operators(tr, k=2, d_model=16, epochs=epochs, lr=1e-2, seed=seed,
                           target_mode="lag")
    oos = [dgp.draw(seed=seed + 1000 + j, dims=dd) for j in range(5)]
    Sz, St = op.extract_and_average(m, oos)
    Sz = op.within_block_restrict(Sz, tr.idx_x, tr.idx_y)
    return op.scale_operators(Sz, St, N, T)


def _mc_cov(res):
    e = res["errors"]
    return float((np.abs(e / e.std()) <= e2.Z95).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--kappa0", type=float, default=3.0)
    ap.add_argument("--op-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--deltas", type=float, nargs="*",
                    default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--out", default="outputs/theorysim/e2/delta_sweep.csv")
    args = ap.parse_args()

    rows = []
    print(f"{'regime':15} {'delta':>6} {'PR/N':>7} {'op':>6} {'gen':>6} {'mc':>6}", flush=True)
    for name, (N, T) in REGIMES.items():
        base_ops = [_learned_wb(N, T, s, args.epochs) for s in args.op_seeds]
        for d in args.deltas:
            prs, opns, gens, mcs = [], [], [], []
            for (Az_wb, B) in base_ops:
                # Blend BOTH operators with the same delta (the temporal axis matters too).
                Ab = op.blend_operator(Az_wb, args.kappa0, d, N)
                Bb = op.blend_operator(B, args.kappa0, d, T)
                opn, trn, prn = op.a7_diagnostics(Ab)
                rg = e2.run_regime(name, N, T, args.reps, DIMS, A_z=Ab, B=Bb, variance="general")
                ri = e2.run_regime(name, N, T, args.reps, DIMS, A_z=Ab, B=Bb, variance="iid")
                prs.append(prn); opns.append(opn)
                gens.append(rg["coverage_95"]); mcs.append(_mc_cov(ri))
            row = dict(regime=name, delta=d, pr_over_n=np.mean(prs), op_norm=np.mean(opns),
                       gen_cov=np.mean(gens), mc_cov=np.mean(mcs),
                       alpha_mid=1.0 / (np.mean(prs) * N))
            rows.append(row)
            print(f"{name:15} {d:>6.2f} {row['pr_over_n']:>7.3f} {row['op_norm']:>6.2f} "
                  f"{row['gen_cov']:>6.3f} {row['mc_cov']:>6.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
