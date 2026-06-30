# theorysim (JBES theory-validation add-on) -- E4 nonlinear bridge.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""E4: the nonlinear model recovers the same factor space as the theory on linear truth.

Two checks (design Sec 5.4):
  - Subspace recovery (sharp): canonical correlations between the nonlinear encoder's k-dim
    latent and the true attended factors F^(B) = B F; all ~1 => same factor space. We also
    report the linear PCA estimator's canonical correlations as a reference.
  - Output check (robust): on linear data the nonlinear target reconstruction, the linear
    PCA reconstruction, and the truth should coincide near the noise floor.

Usage:
  python -m src.theorysim.exp_e4_bridge --smoke
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import torch

from src.theorysim import dgp, estimator, alignment, operators

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
np.seterr(divide="ignore", over="ignore", invalid="ignore")


def _dims(dims, N, T):
    N_y = max(2, round(N / 3))
    N_x = N - N_y
    out = dict(dims)
    out.update(T=T, N_x=N_x, N_y=N_y)
    out.pop("N", None)
    return out


# Shared loadings across train/extract/evaluation panels (same Lambda; sample splitting).
LSEED = 777


def run_e4(dims, N, T, n_oos=50, epochs=1000, lr=1e-2, k=4, d_model=16, seed=0):
    """Train nonlinear + linear models (paper-faithful lagged-target objective) and compare:
      - Y-strong subspace recovery: canonical correlations of the nonlinear latent (and of the
        linear PCA factors) with the true attended Y-STRONG factors B F_{ys}; ~1 means the
        target-relevant factor space the theory describes is recovered.
      - Fair output check: both predictions against the OBSERVED target Z_{tcol} (so each carries
        the same irreducible noise floor), relative to the target sd, plus the nonlinear-vs-linear
        agreement (small => the nonlinear model collapses to the linear solution on linear truth).
    """
    dd = _dims(dims, N, T)
    k_ys = dims.get("k_ys", 2)
    train = dgp.draw(seed=seed, dims=dd, loadings_seed=LSEED)

    nl_model = operators.train_operators(
        train, k=k, d_model=d_model, use_nonlinearity=True, epochs=epochs, lr=lr, seed=seed,
        target_mode="lag")
    lin_model = operators.train_operators(
        train, k=k, d_model=d_model, use_nonlinearity=False, epochs=epochs, lr=lr, seed=seed,
        target_mode="lag")

    # Frozen, scaled linear operators for the linear-PCA reference.
    oos_ops = [dgp.draw(seed=seed + 1000 + j, dims=dd, loadings_seed=LSEED) for j in range(5)]
    S_z, S_tau = operators.extract_and_average(lin_model, oos_ops)
    A_z, B = operators.scale_operators(S_z, S_tau, N, T)

    cca_nl, cca_lin, cca_nl_ys, cca_lin_ys = [], [], [], []
    rel_nl, rel_lin, rel_nl_lin = [], [], []
    series = None
    for j in range(n_oos):
        d = dgp.draw(seed=seed + 5000 + j, dims=dd, loadings_seed=LSEED)
        Zt = torch.as_tensor(d.Z, dtype=torch.float32)

        # Nonlinear latent and the model's own temporal operator.
        with torch.no_grad():
            _, latent = nl_model(Zt, return_latent=True)
            _, B_nl = nl_model.attention_matrices(Zt)
        latent = latent.numpy()
        F_B_nl = B_nl.numpy() @ d.F                   # attended factors via the model's own B
        cca_nl.append(alignment.canonical_correlations(latent, F_B_nl))
        cca_nl_ys.append(alignment.canonical_correlations(latent, B_nl.numpy() @ d.F[:, :k_ys]))

        # Linear PCA estimator (frozen linear operators).
        Ztilde = estimator.attended_panel(B, d.Z, A_z)
        _, F_hat, C_hat = estimator.pca_factors(Ztilde, k)
        cca_lin.append(alignment.canonical_correlations(F_hat, B @ d.F))
        cca_lin_ys.append(alignment.canonical_correlations(F_hat, B @ d.F[:, :k_ys]))

        # Fair output check: both MODELS forecast the OBSERVED target through their own heads
        # (nonlinear vs the linear ablation, the paper's Table-1 bridge), relative to target sd.
        tcol = d.target_col
        obs = d.Z[:, tcol]
        sd = float(obs.std())
        with torch.no_grad():
            nl_pred = nl_model(Zt).numpy()
            lin_pred = lin_model(Zt).numpy()
        rel_nl.append(np.sqrt(np.mean((nl_pred - obs) ** 2)) / sd)
        rel_lin.append(np.sqrt(np.mean((lin_pred - obs) ** 2)) / sd)
        rel_nl_lin.append(np.sqrt(np.mean((nl_pred - lin_pred) ** 2)) / sd)
        if series is None:
            series = dict(observed=obs, common=d.C[:, tcol], nonlinear=nl_pred, linear=lin_pred)

    return dict(
        cca_nl_mean=np.mean(cca_nl, axis=0),
        cca_lin_mean=np.mean(cca_lin, axis=0),
        cca_nl_ys_mean=np.mean(cca_nl_ys, axis=0),
        cca_lin_ys_mean=np.mean(cca_lin_ys, axis=0),
        rel_nl=float(np.mean(rel_nl)),
        rel_lin=float(np.mean(rel_lin)),
        rel_nl_lin=float(np.mean(rel_nl_lin)),
        series=series,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=150)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--n-oos", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="outputs/theorysim/e4/results.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dims = dict(k_ys=2, k_R=2, sigma2=2.0)
    if args.smoke:
        N, T, n_oos, epochs = 120, 150, 20, 600
    else:
        N, T, n_oos, epochs = args.N, args.T, args.n_oos, args.epochs

    res = run_e4(dims, N, T, n_oos=n_oos, epochs=epochs, lr=args.lr)
    print(f"E4 (N={N}, T={T}, n_oos={n_oos}, epochs={epochs}):")
    print("  [headline] Y-strong canonical corr  nonlinear:",
          np.array2string(res["cca_nl_ys_mean"], precision=3),
          " linear:", np.array2string(res["cca_lin_ys_mean"], precision=3))
    print("  all-k canonical corr  nonlinear:",
          np.array2string(res["cca_nl_mean"], precision=3),
          " linear:", np.array2string(res["cca_lin_mean"], precision=3))
    print(f"  output (rel. to target sd) vs OBSERVED target: nonlinear={res['rel_nl']:.3f}  "
          f"linear={res['rel_lin']:.3f}")
    print(f"  nonlinear-vs-linear agreement (rel.)={res['rel_nl_lin']:.3f} "
          f"(small => collapses to linear)")

    if not args.smoke:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for i, c in enumerate(res["cca_nl_mean"]):
                w.writerow([f"cca_nl_{i+1}", c])
            for i, c in enumerate(res["cca_lin_mean"]):
                w.writerow([f"cca_lin_{i+1}", c])
            for i, c in enumerate(res["cca_nl_ys_mean"]):
                w.writerow([f"cca_nl_ys_{i+1}", c])
            for i, c in enumerate(res["cca_lin_ys_mean"]):
                w.writerow([f"cca_lin_ys_{i+1}", c])
            w.writerow(["rel_nl", res["rel_nl"]])
            w.writerow(["rel_lin", res["rel_lin"]])
            w.writerow(["rel_nl_lin", res["rel_nl_lin"]])
        print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
