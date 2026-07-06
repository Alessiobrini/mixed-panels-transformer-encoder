# theorysim (JBES theory-validation add-on) -- E2 finite-sample bias vs N sweep.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Does the concentrated learned-operator coverage gap vanish as the panel grows, or plateau?

For each N=T we build the block-restricted learned operator, then measure, averaged over
operator seeds, structural draws, and a spread of (i,t) cells:
  - bias_sd  : |E[C_hat - C]| / sd, the standardized finite-sample bias of the common component;
  - mc_cov   : coverage after re-studentizing by the Monte Carlo sd (the infeasible true width);
  - gen_cov  : coverage under the feasible general-variance plug-in (single representative cell);
  - pr_over_n: the operator participation ratio PR/N = N / ||A_z^T A_z||_F^2 (effective dimension
               relative to N; ~1 for the identity, small for a concentrated operator).

The single-target attention concentrates on a bounded set of hubs, so PR/N shrinks with N and
the standardized bias does not vanish: coverage plateaus below nominal. This is the finite-sample
boundary behind the E2 learned-arm collapse, and it is why the A.7 concentration control matters
for inference.

Usage: python -m src.theorysim.rerun_e2_bias_sweep [--reps 400]
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

from src.theorysim import dgp, operators as op, estimator, exp_e2_coverage as e2, rates

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

DIMS = dict(k_ys=2, k_R=0, sigma2=2.0)


def _build(N, T, op_seed, epochs):
    Ny = max(2, round(N / 3))
    Nx = N - Ny
    dd = dict(DIMS, T=T, N_x=Nx, N_y=Ny)
    tr = dgp.draw(seed=op_seed, dims=dd)
    m = op.train_operators(tr, k=2, d_model=16, epochs=epochs, lr=1e-2, seed=op_seed,
                           target_mode="lag")
    oos = [dgp.draw(seed=op_seed + 1000 + j, dims=dd) for j in range(5)]
    Sz, St = op.extract_and_average(m, oos)
    Sz = op.within_block_restrict(Sz, tr.idx_x, tr.idx_y)
    return op.scale_operators(Sz, St, N, T)


def _pr_over_n(A):
    AtA = A.T @ A
    return float(np.trace(AtA) ** 2 / (AtA ** 2).sum()) / A.shape[0]


def _bias_cov(Az, B, N, T, reps, ncell, nstruct):
    Ny = max(2, round(N / 3))
    biases, covs = [], []
    for s in range(nstruct):
        base = dgp.draw(seed=5000 + s, dims=dict(DIMS, T=T, N_x=N - Ny, N_y=Ny))
        Ct = B @ base.C @ Az
        idx_y = base.idx_y
        sigma2 = base.meta["sigma2"]
        cells = [(int(idx_y[c % len(idx_y)]), (c + 1) * T // (ncell + 1)) for c in range(ncell)]
        errs = {c: [] for c in cells}
        for r in range(reps):
            rng = np.random.default_rng(20000 + 1000 * s + r)
            Z = base.C + np.sqrt(sigma2) * rng.standard_normal(base.C.shape)
            _, _, Ch = estimator.pca_factors(B @ Z @ Az, 2)
            for (i, t) in cells:
                errs[(i, t)].append(Ch[t, i] - Ct[t, i])
        for e in errs.values():
            e = np.array(e)
            biases.append(abs(e.mean()) / e.std())
            covs.append((np.abs(e / e.std()) <= 1.96).mean())
    return float(np.mean(biases)), float(np.mean(covs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--grid", type=int, nargs="*", default=[100, 200, 400, 800])
    ap.add_argument("--op-seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--ncell", type=int, default=10)
    ap.add_argument("--nstruct", type=int, default=2)
    ap.add_argument("--out", default="outputs/theorysim/e2/bias_sweep.csv")
    args = ap.parse_args()

    rows = []
    print(f"{'N=T':>6} {'bias/sd':>9} {'MCcov':>7} {'gen_cov':>8} {'PR/N':>7} "
          f"{'alpha':>7} {'sqrtT_a':>8} {'sqrtN_a':>8}", flush=True)
    for N in args.grid:
        T = N
        bs, cs, gs, prs, abs_ = [], [], [], [], []
        for os_ in args.op_seeds:
            Az, B = _build(N, T, os_, args.epochs)
            mb, mc = _bias_cov(Az, B, N, T, args.reps, args.ncell, args.nstruct)
            rg = e2.run_regime("sweep", N, T, 2 * args.reps, DIMS, A_z=Az, B=B, variance="general")
            bs.append(mb); cs.append(mc); gs.append(rg["coverage_95"]); prs.append(_pr_over_n(Az))
            abs_.append(rates.alpha_bar(Az, B, N, T))
        ab = float(np.mean(abs_))
        # Growth conditions behind Theorems 2-3: sqrt(T)*alpha_bar -> 0 and sqrt(N_eff)*alpha_bar
        # -> 0. With the trace scaling N_eff = tr(A^T A) = N, and alpha_bar's middle term = 1/PR,
        # so a bounded PR makes these products grow rather than vanish (the boundary, in the
        # paper's own quantities).
        row = dict(N=N, bias_sd=np.mean(bs), mc_cov=np.mean(cs), gen_cov=np.mean(gs),
                   pr_over_n=np.mean(prs), alpha_bar=ab,
                   sqrtT_alpha=np.sqrt(T) * ab, sqrtN_alpha=np.sqrt(N) * ab)
        rows.append(row)
        print(f"{N:>6} {row['bias_sd']:>9.3f} {row['mc_cov']:>7.3f} {row['gen_cov']:>8.3f} "
              f"{row['pr_over_n']:>7.3f} {ab:>7.3f} {row['sqrtT_alpha']:>8.3f} "
              f"{row['sqrtN_alpha']:>8.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
