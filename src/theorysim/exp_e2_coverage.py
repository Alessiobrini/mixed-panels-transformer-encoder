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


def run_regime(regime, N, T, reps, dims, A_z=None, B=None, struct_seed=0, noise_seed0=10_000,
               variance="iid"):
    """Coverage and studentized stats for one regime at (N, T).

    A_z, B: operators to apply before PCA. None -> oracle (identity). When learned/frozen
    operators are passed, the target is the attended common component (B C A_z) and PCA runs
    on the attended panel; the same analytic plug-in SE is used, so coverage shows whether the
    paper's variance stays calibrated under the operators the model actually learns.

    variance: "iid" uses the iid-collapse plug-in Omega = sigma^2 Sigma_F, Xi = sigma^2 Sigma_L.
    "general" uses the feasible Theorem-3 plug-in for our frozen-operator DGP, where the attended
    noise e~ = B e A_z has Cov(e~_ti, e~_sj) = sigma^2 (BB^T)_ts (A_z^T A_z)_ij. It rebuilds the
    two score covariances with (A_z^T A_z) and (BB^T) inserted where the iid collapse had
    identities: the factor-CLT term (scales 1/N) uses M_L = Lambda^T (A_z^T A_z) Lambda / N
    scaled by (BB^T)_tt, and the loading-CLT term (scales 1/T) uses M_F = F^T (BB^T) F / T scaled
    by (A_z^T A_z)_ii. Under the DGP the factors have unit unconditional variance, so Sigma_F ~ I
    and the general SE reduces to the iid SE at the oracle (validated numerically).
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
    # Frozen-operator second moments for the general-variance plug-in (item 3c).
    AtA = A_z.T @ A_z
    BBt = B @ B.T
    AtA_ii = float(AtA[i, i])
    BBt_tt = float(BBt[t, t])

    zstats = []
    errors = []
    cov90 = cov95 = 0
    for r in range(reps):
        rng = np.random.default_rng(noise_seed0 + r)
        Z = base.C + np.sqrt(sigma2) * rng.standard_normal(base.C.shape)
        Ztilde = B @ Z @ A_z
        Lambda_hat, F_hat, C_hat = estimator.pca_factors(Ztilde, k)
        C_it = float(C_hat[t, i])
        sigma2_hat = float(((Ztilde - C_hat) ** 2).mean())

        lam_i = Lambda_hat[i]                          # (k,)
        Sigma_F = (F_hat.T @ F_hat) / T
        f_t = F_hat[t]

        if variance == "general":
            # Feasible Theorem-3 plug-in: insert (A_z^T A_z) and (BB^T) into the two score
            # covariances. sigma^2_hat = ||e~||_F^2 / (tr(BB^T) tr(A_z^T A_z)); for the scaled
            # operators tr(BB^T)=T, tr(A_z^T A_z)=N, so it equals the iid mean residual above.
            M_L = (Lambda_hat.T @ AtA @ Lambda_hat) / N      # attended loading Gram (k x k)
            M_F = (F_hat.T @ BBt @ F_hat) / T                # attended factor Gram (k x k)
            # Factor-CLT term (scales 1/N): Lambda_i^T Sigma_L^{-1} Gamma_t Sigma_L^{-1} Lambda_i
            # with Sigma_L = I and Gamma_t = sigma^2 (BB^T)_tt M_L. No Sigma_F sandwich under the
            # Lambda-normalization; reduces to sigma^2 Lambda_i^T Lambda_i at the oracle.
            var_F = sigma2_hat * BBt_tt * float(lam_i @ (M_L @ lam_i))
            # Loading-CLT term (scales 1/T): F_t^T Sigma_F^{-1} Phi_i Sigma_F^{-1} F_t with
            # Phi_i = sigma^2 (A_z^T A_z)_ii M_F; reduces to sigma^2 F_t^T Sigma_F^{-1} F_t.
            SinvF = np.linalg.solve(Sigma_F, f_t)
            var_L = sigma2_hat * AtA_ii * float(SinvF @ (M_F @ SinvF))
        else:
            # iid collapse Omega = sigma^2 Sigma_F, Xi = sigma^2 Sigma_L (Sigma_L = I).
            var_F = sigma2_hat * float(lam_i @ lam_i)
            var_L = sigma2_hat * float(f_t @ np.linalg.solve(Sigma_F, f_t))

        # Combined SE Var(C_hat - C) = sigma^2_F/N + sigma^2_Lambda/T, of which the F-dominant
        # (drop the /T term) and Lambda-dominant (drop the /N term) are the limiting cases.
        se = np.sqrt(var_F / N + var_L / T)
        z = (C_it - C_true) / se
        zstats.append(z)
        errors.append(C_it - C_true)
        cov90 += abs(z) <= Z90
        cov95 += abs(z) <= Z95

    zstats = np.array(zstats)
    errors = np.array(errors)
    return dict(
        regime=regime, N=N, T=T, reps=reps,
        coverage_90=cov90 / reps, coverage_95=cov95 / reps,
        z_mean=float(zstats.mean()), z_sd=float(zstats.std()),
        zstats=zstats, errors=errors,
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
