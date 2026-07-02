# theorysim (JBES theory-validation add-on) -- E2 coverage / asymptotic normality.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""E2: empirical coverage of CIs for the Y-strong common component C_{y,i,t} (Theorems 2-3).

The closed-form asymptotic variance is taken from the paper. For the baseline iid DGP the
long-run covariances collapse to Omega = sigma^2 Sigma_F,y and Xi = sigma^2 Sigma_Lambda,ys,
so the studentized statistic uses an exact analytic plug-in:
  - F-dominant (N_eff/T -> 0):  sqrt(N_eff)(C_hat - C) ~ N(0, sigma^2_{C,it,F}),
        sigma^2_{C,it,F} = sigma^2 Lambda_i^T (Sigma_Lambda,ys)^{-1} Lambda_i.
  - Lambda-dominant (T/N_eff -> 0): sqrt(T)(C_hat - C) ~ N(0, sigma^2_{C,it,Lambda}),
        sigma^2_{C,it,Lambda} = sigma^2 F_t^T (Sigma_F,y)^{-1} F_t.
  - Mixed (T/N_eff -> c): sqrt(T)(C_hat - C) ~ N(0, sigma^2_Lambda + c sigma^2_F).

First-pass scope: a pure Y-strong factor model (k_R = 0) so the common component equals the
Y-strong component and Sigma_Lambda,ys = I under the PCA normalization (the k_R>0 case with
explicit Y-strong subspace identification is a follow-up). Oracle operators (A_z=I, B=I) so
the assumptions hold exactly and N_eff = N. The common component is rotation-invariant, so no
alignment is needed. Coverage is computed for a fixed representative (i, t) over noise reps
with the structural (Lambda, F) held fixed.

Usage:
  python -m src.theorysim.exp_e2_coverage --smoke
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

Z90, Z95 = 1.6448536269514722, 1.959963984540054


def run_regime(regime, N, T, reps, dims, A_z=None, B=None, struct_seed=0, noise_seed0=10_000):
    """Coverage and studentized stats for one regime at (N, T).

    A_z, B: operators to apply before PCA. None -> oracle (identity). When learned/frozen
    operators are passed, the target is the attended common component (B C A_z) and PCA runs
    on the attended panel; the same analytic plug-in SE is used, so coverage shows whether the
    paper's variance stays calibrated under the operators the model actually learns.
    """
    k_ys = dims["k_ys"]
    k = k_ys  # k_R = 0 for E2 this pass
    N_y = max(2, round(N / 3))
    N_x = N - N_y
    base = dgp.draw(seed=struct_seed,
                    dims=dict(dims, T=T, N_x=N_x, N_y=N_y))  # fixed structural draw
    sigma2 = base.meta["sigma2"]
    i = int(base.idx_y[0])     # representative Y unit
    t = T // 2                 # representative time
    if A_z is None:
        A_z = np.eye(N)
        B = np.eye(T)
    C_target = B @ base.C @ A_z             # attended common component (= C for oracle)
    C_true = float(C_target[t, i])

    zstats = []
    cov90 = cov95 = 0
    for r in range(reps):
        rng = np.random.default_rng(noise_seed0 + r)
        Z = base.C + np.sqrt(sigma2) * rng.standard_normal(base.C.shape)
        Ztilde = B @ Z @ A_z
        Lambda_hat, F_hat, C_hat = estimator.pca_factors(Ztilde, k)
        C_it = float(C_hat[t, i])
        sigma2_hat = float(((Ztilde - C_hat) ** 2).mean())

        lam_i = Lambda_hat[i]                          # (k,)
        var_F = sigma2_hat * float(lam_i @ lam_i)      # Sigma_Lambda,ys = I (normalization)
        Sigma_F = (F_hat.T @ F_hat) / T
        f_t = F_hat[t]
        var_L = sigma2_hat * float(f_t @ np.linalg.solve(Sigma_F, f_t))

        # General (Theorem 3(iii)) variance Var(C_hat - C) = sigma^2_F/N + sigma^2_Lambda/T,
        # of which the F-dominant (drop the /T term) and Lambda-dominant (drop the /N term)
        # are the limiting cases. The combined SE gives correct coverage for ANY (N, T);
        # using a single-term corner SE off its corner under-covers (the dropped term is
        # non-negligible). We therefore studentize with the combined SE in every setting.
        se = np.sqrt(var_F / N + var_L / T)
        z = (C_it - C_true) / se
        zstats.append(z)
        cov90 += abs(z) <= Z90
        cov95 += abs(z) <= Z95

    zstats = np.array(zstats)
    return dict(
        regime=regime, N=N, T=T, reps=reps,
        coverage_90=cov90 / reps, coverage_95=cov95 / reps,
        z_mean=float(zstats.mean()), z_sd=float(zstats.std()),
        zstats=zstats,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="outputs/theorysim/e2/coverage.csv")
    ap.add_argument("--qq-out", default="outputs/theorysim/e2/qq_mixed.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dims = dict(k_ys=2, k_R=0, sigma2=2.0)
    regimes = {
        "F_dominant": (50, 400),
        "Lambda_dominant": (400, 50),
        "mixed": (200, 200),
    }
    reps = 400 if args.smoke else args.reps

    results = []
    for name, (N, T) in regimes.items():
        res = run_regime(name, N, T, reps, dims)
        results.append(res)
        print(f"  {name:16s} N={N:4d} T={T:4d}  cov90={res['coverage_90']:.3f}  "
              f"cov95={res['coverage_95']:.3f}  z_mean={res['z_mean']:+.3f}  z_sd={res['z_sd']:.3f}")
    if args.smoke:
        print("SMOKE E2: coverage_90 ~ 0.90, coverage_95 ~ 0.95, z_sd ~ 1.0 expected.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "N", "T", "reps", "coverage_90", "coverage_95", "z_mean", "z_sd"])
        for r in results:
            w.writerow([r["regime"], r["N"], r["T"], r["reps"],
                        r["coverage_90"], r["coverage_95"], r["z_mean"], r["z_sd"]])
    qq = Path(args.qq_out)
    with qq.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z"])
        for z in next(r for r in results if r["regime"] == "mixed")["zstats"]:
            w.writerow([z])
    print(f"wrote -> {out} and {qq}")


if __name__ == "__main__":
    main()
