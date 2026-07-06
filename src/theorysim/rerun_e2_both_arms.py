# theorysim (JBES theory-validation add-on) -- E2 coverage: operator arms + variance channels.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""E2 coverage of the Y-strong common-component CIs across operator arms and variance
channels (referee follow-up, July email).

Operator arms (E2 needs the within-block restriction so the learned A_z does not inject
X-only directions into the Y block -- B.1-B.3, separate from A.7):
  - oracle            : A_z = I, B = I.
  - learned_raw       : learned frozen A_z, NO block restriction (diagnostic for item 3b).
  - learned_wb        : learned frozen A_z, block-restricted (the E2-correct learned arm).
  - clipped_wb        : learned_wb with singular values clipped at kappa, so A.7 holds too.

Variance channels:
  - iid  : the analytic plug-in with the iid collapse Omega = sigma^2 Sigma_F, Xi = sigma^2 Sigma_L.
  - mc   : re-studentize the SAME draws by the Monte Carlo sd of the estimator error (item 3a).
           If coverage restores under mc and the QQ is normal, the CLT survives concentration
           and only the plug-in was wrong; if not, normality itself needs the A.7 boundedness.

Writes outputs/theorysim/e2/coverage_gate.csv and per-arm QQ CSVs (mixed regime).

Usage: python -m src.theorysim.rerun_e2_both_arms [--reps 2000] [--kappa 3.0]
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


def _build_ops(N, T, kappa, seed=0, epochs=800, lr=1e-2):
    """Train the linear AB1 once, freeze, and return the raw / block-restricted / clipped
    cross-sectional operators (temporal B shared, scaled)."""
    N_y = max(2, round(N / 3))
    N_x = N - N_y
    dd = dict(DIMS, T=T, N_x=N_x, N_y=N_y)
    tr = dgp.draw(seed=seed, dims=dd)
    m = op.train_operators(tr, k=2, d_model=16, epochs=epochs, lr=lr, seed=seed,
                           target_mode="lag")
    oos = [dgp.draw(seed=seed + 1000 + j, dims=dd) for j in range(5)]
    Sz, St = op.extract_and_average(m, oos)

    Az_raw, B = op.scale_operators(Sz, St, N, T)
    Sz_wb = op.within_block_restrict(Sz, tr.idx_x, tr.idx_y)
    Az_wb, _ = op.scale_operators(Sz_wb, St, N, T)
    Az_clip = op.clip_operator(Az_wb, kappa, N)
    return {"learned_raw": (Az_raw, B), "learned_wb": (Az_wb, B),
            "clipped_wb": (Az_clip, B)}


def _mc_coverage(res):
    """Re-studentize the same errors by their Monte Carlo sd (item 3a)."""
    e = res["errors"]
    sd = float(e.std())
    zmc = e / sd
    return (float((np.abs(zmc) <= e2.Z90).mean()),
            float((np.abs(zmc) <= e2.Z95).mean()),
            float(zmc.std()), zmc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--kappa", type=float, default=3.0)
    ap.add_argument("--out", default="outputs/theorysim/e2/coverage_gate.csv")
    ap.add_argument("--qq-dir", default="outputs/theorysim/e2")
    args = ap.parse_args()

    rows = []
    qq = {}

    def add(regime, N, T, arm, method, cov90, cov95, z_sd, op_norm):
        rows.append(dict(regime=regime, N=N, T=T, arm=arm, method=method,
                         coverage_90=cov90, coverage_95=cov95, z_sd=z_sd, op_norm=op_norm))

    print(f"reps={args.reps}  kappa={args.kappa}", flush=True)
    for name, (N, T) in REGIMES.items():
        # oracle (iid plug-in; A.7 and B.1-B.3 hold exactly). General channel is reported too,
        # as the check that it reduces to iid where the assumptions hold.
        ro = e2.run_regime(name, N, T, args.reps, DIMS, variance="iid")
        rog = e2.run_regime(name, N, T, args.reps, DIMS, variance="general")
        add(name, N, T, "oracle", "iid", ro["coverage_90"], ro["coverage_95"], ro["z_sd"], 1.0)
        add(name, N, T, "oracle", "general", rog["coverage_90"], rog["coverage_95"],
            rog["z_sd"], 1.0)

        ops = _build_ops(N, T, args.kappa, epochs=args.epochs)
        for arm, (A_z, B) in ops.items():
            opn = float(np.linalg.norm(A_z, 2))
            r = e2.run_regime(name, N, T, args.reps, DIMS, A_z=A_z, B=B, variance="iid")
            rg = e2.run_regime(name, N, T, args.reps, DIMS, A_z=A_z, B=B, variance="general")
            add(name, N, T, arm, "iid", r["coverage_90"], r["coverage_95"], r["z_sd"], opn)
            add(name, N, T, arm, "general", rg["coverage_90"], rg["coverage_95"], rg["z_sd"], opn)
            c90, c95, zsd, zmc = _mc_coverage(r)
            add(name, N, T, arm, "mc", c90, c95, zsd, opn)
            if name == "mixed":
                qq[f"{arm}_iid"] = r["zstats"]
                qq[f"{arm}_general"] = rg["zstats"]
                qq[f"{arm}_mc"] = zmc
            print(f"{name:15s} {arm:12s} op={opn:5.1f} | iid={r['coverage_95']:.3f} "
                  f"gen={rg['coverage_95']:.3f} mc={c95:.3f}", flush=True)
        if name == "mixed":
            qq["oracle_iid"] = ro["zstats"]
            qq["oracle_general"] = rog["zstats"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}", flush=True)

    qdir = Path(args.qq_dir)
    for key, z in qq.items():
        p = qdir / f"qq_{key}.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["z"])
            for v in z:
                w.writerow([float(v)])
    print(f"wrote {len(qq)} QQ files -> {qdir}", flush=True)


if __name__ == "__main__":
    main()
