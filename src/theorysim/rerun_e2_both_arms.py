# theorysim (JBES theory-validation add-on) -- E2 coverage under oracle vs learned operators.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""E2, both operator arms: empirical coverage of the Y-strong common-component CIs under the
oracle operators (A_z=I, B=I) and under the learned, frozen operators read off the trained
linear model. The oracle arm is the one in the main report; this add-on shows both side by
side (referee request). Writes a CSV consumed by tables.make_e2_both_arms_table.

Usage: python -m src.theorysim.rerun_e2_both_arms [--reps 2000]
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


def _learned_ops(N, T, seed=0, epochs=800, lr=1e-2):
    N_y = max(2, round(N / 3))
    N_x = N - N_y
    dd = dict(DIMS, T=T, N_x=N_x, N_y=N_y)
    tr = dgp.draw(seed=seed, dims=dd)
    m = op.train_operators(tr, k=2, d_model=16, epochs=epochs, lr=lr, seed=seed,
                           target_mode="lag")
    oos = [dgp.draw(seed=seed + 1000 + j, dims=dd) for j in range(5)]
    Sz, St = op.extract_and_average(m, oos)
    return op.scale_operators(Sz, St, N, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--out", default="outputs/theorysim/e2/coverage_both_arms.csv")
    args = ap.parse_args()

    rows = []
    print(f"reps={args.reps}", flush=True)
    for name, (N, T) in REGIMES.items():
        ro = e2.run_regime(name, N, T, args.reps, DIMS)
        rows.append(dict(regime=name, N=N, T=T, arm="oracle",
                         coverage_90=ro["coverage_90"], coverage_95=ro["coverage_95"],
                         z_sd=ro["z_sd"], op_norm=1.0))
        A_z, B = _learned_ops(N, T, epochs=args.epochs)
        rl = e2.run_regime(name, N, T, args.reps, DIMS, A_z=A_z, B=B)
        opn = float(np.linalg.norm(A_z, 2))
        rows.append(dict(regime=name, N=N, T=T, arm="learned",
                         coverage_90=rl["coverage_90"], coverage_95=rl["coverage_95"],
                         z_sd=rl["z_sd"], op_norm=opn))
        print(f"{name:16s}  oracle cov95={ro['coverage_95']:.3f} zsd={ro['z_sd']:.2f}  |  "
              f"learned cov95={rl['coverage_95']:.3f} zsd={rl['z_sd']:.2f}  op={opn:.1f}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
