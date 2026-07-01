# theorysim (JBES theory-validation add-on) -- E1 consistency experiment runner.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""E1: estimation error of the linear estimator vs the Theorem-1 rate alpha-bar.

Per (T, N) and operator arm: form the frozen, scaled operators once, then over many
independent estimation draws run PCA on Z-tilde = B Z A_z, align by least squares, and
record loading/factor/common-component errors and the theoretical rate. A unit slope of
log(error) vs log(alpha-bar) validates Theorem 1.

Arms (robustness): oracle (A_z=I, B=I), independent (train AB1 on a separate panel, freeze,
apply), same_sample (train on each estimation panel itself).

Usage:
  python -m src.theorysim.exp_e1_consistency --smoke
  python -m src.theorysim.exp_e1_consistency --arm independent --out outputs/theorysim/e1/results.csv
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

from src.theorysim import dgp, estimator, alignment, operators, rates

# Benign numpy-2.2 / Apple-Accelerate FP flags on some matmuls; results are exact.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
np.seterr(divide="ignore", over="ignore", invalid="ignore")


# Fixed loadings seed: training, operator-extraction, and estimation panels all share the
# SAME Lambda (sample splitting, design Sec 4.1 "independent draw, same Lambda"); only the
# factors and noise are resampled across panels.
LSEED = 777


def _operators_for_arm(arm, N, T, dims, k, d_model, op_seed, epochs, lr):
    """Return scaled (A_z, B) for the requested arm (trained once per (T,N))."""
    if arm == "oracle":
        return operators.oracle_operators(N, T)  # already tr=N, tr=T

    dd = _dims(dims, N, T)
    if arm == "parameter_free":  # paper's simplest case Q=K=Z, no learned projection
        oos = [dgp.draw(seed=op_seed + 1000 + j, dims=dd, loadings_seed=LSEED) for j in range(5)]
        return operators.parameter_free_operators(oos, N, T)

    # Learned arms: train AB1 on a training panel, extract+average, then scale.
    # target_mode "lag" (independent_lag) is the paper-faithful objective: the target's own
    # past is a predictor, only its contemporaneous value is withheld. "mask" zeros the column.
    target_mode = "lag" if arm == "independent_lag" else "mask"
    train_draw = dgp.draw(seed=op_seed, dims=dd, loadings_seed=LSEED)
    model = operators.train_operators(
        train_draw, k=k, d_model=d_model, use_nonlinearity=False,
        epochs=epochs, lr=lr, seed=op_seed, target_mode=target_mode,
    )
    # Average over OOS panels (independent F/e, SAME Lambda as the estimation panels below).
    oos = [dgp.draw(seed=op_seed + 1000 + j, dims=dd, loadings_seed=LSEED) for j in range(5)]
    S_z, S_tau = operators.extract_and_average(model, oos)
    if arm == "independent_wb":  # within-block restriction (zero X<->Y), per design for E2/E3
        S_z = operators.within_block_restrict(S_z, train_draw.idx_x, train_draw.idx_y)
    A_z, B = operators.scale_operators(S_z, S_tau, N, T)
    return A_z, B


def _dims(dims, N, T):
    """Split N into (N_x, N_y) keeping the baseline 2:1 ratio; set T."""
    N_y = max(2, round(N / 3))
    N_x = N - N_y
    out = dict(dims)
    out.update(T=T, N_x=N_x, N_y=N_y)
    out.pop("N", None)
    return out


def run_cell(arm, T, N, reps, dims, k, d_model, base_seed=0, epochs=800, lr=1e-2):
    """Run one (T, N) cell; return a list of per-rep result dicts."""
    rows = []
    if arm != "same_sample":
        A_z, B = _operators_for_arm(arm, N, T, dims, k, d_model, base_seed, epochs, lr)
        a_bar = rates.alpha_bar(A_z, B, N, T)
    for r in range(reps):
        d = dgp.draw(seed=base_seed + 1 + r, dims=_dims(dims, N, T), loadings_seed=LSEED)
        if arm == "same_sample":
            model = operators.train_operators(
                d, k=k, d_model=d_model, use_nonlinearity=False,
                epochs=epochs, lr=lr, seed=base_seed + 1 + r,
            )
            S_z, S_tau = operators.extract_and_average(model, [d])
            A_z, B = operators.scale_operators(S_z, S_tau, N, T)
            a_bar = rates.alpha_bar(A_z, B, N, T)

        Ztilde = estimator.attended_panel(B, d.Z, A_z)
        Lambda_hat, F_hat, C_hat = estimator.pca_factors(Ztilde, k)
        Lambda_A, F_B = estimator.true_attended_targets(A_z, B, d.Lambda, d.F)
        C_true = estimator.true_attended_common(A_z, B, d.C)
        rows.append(dict(
            arm=arm, T=T, N=N, rep=r,
            e_Lambda=alignment.loading_error(Lambda_hat, Lambda_A),
            e_F=alignment.factor_error(F_hat, F_B),
            e_C=alignment.common_error(C_hat, C_true),
            e_C_rel=alignment.common_error_relative(C_hat, C_true),
            e_F_rel=alignment.factor_error_relative(F_hat, F_B),
            e_L_rel=alignment.loading_error_relative(Lambda_hat, Lambda_A),
            op_norm=float(np.linalg.norm(A_z, 2)),
            alpha_bar=a_bar,
        ))
    return rows


def _fit_slope(x, y):
    """OLS slope of log(y) on log(x)."""
    lx, ly = np.log(np.asarray(x)), np.log(np.asarray(y))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="oracle",
                    choices=["oracle", "independent", "independent_lag", "independent_wb",
                             "parameter_free", "same_sample"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--T", type=int, nargs="*", default=[50, 100, 200, 400])
    ap.add_argument("--N", type=int, nargs="*", default=[50, 100, 200, 400])
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="outputs/theorysim/e1/results.csv")
    ap.add_argument("--smoke", action="store_true",
                    help="oracle arm, tiny diagonal grid, few reps; print rate slope")
    args = ap.parse_args()

    dims = dict(k_ys=2, k_R=2, sigma2=2.0)
    k = dims["k_ys"] + dims["k_R"]
    d_model = 16

    if args.smoke:
        arm = "oracle"
        grid = [(40, 40), (80, 80), (160, 160), (320, 320)]
        reps = 30
        means = []
        for (T, N) in grid:
            rows = run_cell(arm, T, N, reps, dims, k, d_model)
            e_c = np.mean([r["e_C"] for r in rows])
            a_bar = rows[0]["alpha_bar"]
            means.append((a_bar, e_c))
            print(f"  (T={T:4d}, N={N:4d})  alpha_bar={a_bar:.3e}  mean e_C={e_c:.3e}")
        slope = _fit_slope([m[0] for m in means], [m[1] for m in means])
        print(f"SMOKE oracle E1: log-log slope(e_C ~ alpha_bar) = {slope:.3f}  (target ~1)")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for T in args.T:
        for N in args.N:
            all_rows.extend(run_cell(args.arm, T, N, args.reps, dims, k, d_model,
                                     epochs=args.epochs, lr=args.lr))
            print(f"done arm={args.arm} T={T} N={N}")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
