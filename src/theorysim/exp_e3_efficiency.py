# theorysim (JBES theory-validation add-on) -- E3 efficiency / transfer learning.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""E3: the joint (X, Y) estimator of the Y common component beats the Y-only estimator.

For Y units the baseline common component is C_{y,i,t} = Lambda_{ys,i}^T F^S_{y,t}
(Lambda_y,R = 0), so it equals the full common component on the Y columns. We compare the
joint estimator (PCA on the full attended panel) to the Y-only estimator (PCA on the Y
block), as the auxiliary size N_x grows. The variance/MSE ratio (Y-only / joint) > 1
quantifies the transfer-learning gain (Remark on efficiency). Oracle operators are used so
the assumptions hold exactly and the comparison isolates the transfer gain.

Usage:
  python -m src.theorysim.exp_e3_efficiency --smoke
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

from src.theorysim import dgp, estimator

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
np.seterr(divide="ignore", over="ignore", invalid="ignore")


def _mse_to_truth(C_hat_block, C_true_block):
    return float(np.mean((C_hat_block - C_true_block) ** 2))


def run_e3(dims, N_y, N_x_values, T, reps, seed=0):
    """Return a list of dicts: per N_x, the joint and Y-only MSE and their ratio."""
    k_ys, k_R = dims["k_ys"], dims["k_R"]
    k = k_ys + k_R
    rows = []
    for N_x in N_x_values:
        joint_mses, yonly_mses = [], []
        for r in range(reps):
            d = dgp.draw(seed=seed + r, dims=dict(dims, T=T, N_x=N_x, N_y=N_y))
            C_true_Y = d.C[:, d.idx_y]                       # (T, N_y) truth on Y units

            # Joint estimator: PCA on the full panel (oracle operators => Z-tilde = Z).
            _, _, C_hat = estimator.pca_factors(d.Z, k)
            joint_mses.append(_mse_to_truth(C_hat[:, d.idx_y], C_true_Y))

            # Y-only estimator: PCA on the Y block alone (k_ys strong factors).
            Z_Y = d.Z[:, d.idx_y]
            _, _, C_hat_Y = estimator.pca_factors(Z_Y, k_ys)
            yonly_mses.append(_mse_to_truth(C_hat_Y, C_true_Y))

        mj, my = float(np.mean(joint_mses)), float(np.mean(yonly_mses))
        rows.append(dict(N_x=N_x, N_y=N_y, T=T, mse_joint=mj, mse_yonly=my,
                         ratio_yonly_over_joint=my / mj))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N-y", type=int, default=50)
    ap.add_argument("--N-x", type=int, nargs="*", default=[25, 50, 100, 200, 400])
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="outputs/theorysim/e3/results.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dims = dict(k_ys=2, k_R=2, sigma2=2.0)
    if args.smoke:
        rows = run_e3(dims, N_y=30, N_x_values=[20, 60, 150, 400], T=150, reps=60)
        for row in rows:
            print(f"  N_x={row['N_x']:4d}  mse_joint={row['mse_joint']:.4f}  "
                  f"mse_yonly={row['mse_yonly']:.4f}  ratio(Yonly/joint)={row['ratio_yonly_over_joint']:.3f}")
        print("SMOKE E3: ratio should exceed 1 and grow with N_x (transfer-learning gain).")
        return

    rows = run_e3(dims, N_y=args.N_y, N_x_values=args.N_x, T=args.T, reps=args.reps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
