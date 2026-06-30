# theorysim (JBES theory-validation add-on) -- union-factor linear DGP.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Union-factor linear data-generating process with ground-truth export.

Z = [X Y] = F Lambda^T + e, a single equal-frequency wide panel (X and Y observed at
the same times, stacked as columns). Unlike src/data/simulate_to_long.py (which discards
the latent objects), this module returns and persists the true factors F, loadings
Lambda, and common component C = F Lambda^T, which the theory-validation experiments
(E1-E4) check against.

Design (paper/MPTE_simulation_design.tex, Sec 3):
  - k = k_ys + k_R factors. Columns [0:k_ys] are Y-strong, [k_ys:k] are "rest" (R).
  - Rows [0:N_x] are X, [N_x:N] are Y.
  - Two independent stationary VAR(1) factor blocks, coef phi=0.5, innovation
    N(0, 0.75 I) so the unconditional factor variance is exactly 1.
  - 2x2 loading grid: X loads on Y-strong factor 1 only (not 2) -> rank-1 cross signal
    (Assumption B.3); Y loads on both Y-strong factors; Lambda_y,R = 0 baseline so the
    rest factors are X-only. Weak-leakage stress variant via lambda_yR_scale > 0.
  - Baseline idiosyncratic noise iid Gaussian, variance sigma2 = 2 (~50% common var).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


@dataclass
class SimDraw:
    """One realized draw of the DGP, including ground truth."""

    Z: np.ndarray            # (T, N) observable wide panel
    F: np.ndarray            # (T, k) true factors
    Lambda: np.ndarray       # (N, k) true loadings
    C: np.ndarray            # (T, N) true common component = F Lambda^T
    idx_x: np.ndarray        # indices of X columns/rows  (0 .. N_x-1)
    idx_y: np.ndarray        # indices of Y columns/rows  (N_x .. N-1)
    target_col: int          # designated Y target column
    meta: dict = field(default_factory=dict)


def make_loadings(rng, N_x, N_y, k_ys=2, k_R=2, lambda_yR_scale=0.0):
    """Build the 2x2-block loading matrix Lambda (N x k) and the X/Y index sets.

    lambda_yR_scale: variance of the weak Y-on-rest leakage (0 = baseline, Lambda_y,R=0;
    the stress variant uses ~ 1/N_y).
    """
    k = k_ys + k_R
    N = N_x + N_y
    Lambda = np.zeros((N, k))
    idx_x = np.arange(0, N_x)
    idx_y = np.arange(N_x, N)

    # X block, Y-strong columns: load on the FIRST Y-strong factor only.
    Lambda[np.ix_(idx_x, [0])] = rng.standard_normal((N_x, 1))
    # X block, rest columns: full N(0,1).
    if k_R > 0:
        Lambda[np.ix_(idx_x, np.arange(k_ys, k))] = rng.standard_normal((N_x, k_R))
    # Y block, Y-strong columns: full N(0,1) on all Y-strong factors.
    Lambda[np.ix_(idx_y, np.arange(0, k_ys))] = rng.standard_normal((N_y, k_ys))
    # Y block, rest columns: 0 baseline; weak leakage if requested.
    if k_R > 0 and lambda_yR_scale > 0.0:
        Lambda[np.ix_(idx_y, np.arange(k_ys, k))] = (
            np.sqrt(lambda_yR_scale) * rng.standard_normal((N_y, k_R))
        )

    return Lambda, idx_x, idx_y


def simulate_factors(rng, T, k_ys=2, k_R=2, phi=0.5, innov_var=0.75, burn_in=500):
    """Two independent stationary VAR(1) factor blocks, returned as F (T x k).

    Each factor follows AR(1) with coefficient phi and innovation variance innov_var;
    with phi=0.5, innov_var=0.75 the unconditional variance is 0.75/(1-0.25) = 1.
    The two blocks are independent by construction (independent innovations).
    """
    k = k_ys + k_R
    total = T + burn_in
    eps = np.sqrt(innov_var) * rng.standard_normal((total, k))
    F = np.zeros((total, k))
    for t in range(1, total):
        F[t] = phi * F[t - 1] + eps[t]
    return F[burn_in:]


def simulate_panel(
    rng,
    T,
    N_x,
    N_y,
    k_ys=2,
    k_R=2,
    sigma2=2.0,
    noise_kind="iid",
    lambda_yR_scale=0.0,
    phi=0.5,
    innov_var=0.75,
    burn_in=500,
    lambda_pack=None,
):
    """Draw one panel and return a SimDraw with the observable Z and the ground truth.

    lambda_pack=(Lambda, idx_x, idx_y) reuses fixed loadings (sample splitting: same Lambda
    across training and estimation panels, only F and e resampled). If None, loadings are
    drawn from rng.
    """
    F = simulate_factors(
        rng, T, k_ys=k_ys, k_R=k_R, phi=phi, innov_var=innov_var, burn_in=burn_in
    )
    if lambda_pack is None:
        Lambda, idx_x, idx_y = make_loadings(
            rng, N_x, N_y, k_ys=k_ys, k_R=k_R, lambda_yR_scale=lambda_yR_scale
        )
    else:
        Lambda, idx_x, idx_y = lambda_pack
    N = N_x + N_y
    C = F @ Lambda.T  # (T, N) common component

    if noise_kind == "iid":
        e = np.sqrt(sigma2) * rng.standard_normal((T, N))
    else:
        # Placeholder for the deferred AR(1)+Toeplitz variant; baseline is iid only.
        raise NotImplementedError(f"noise_kind={noise_kind!r} not implemented this pass")

    Z = C + e
    target_col = int(idx_y[0])  # first Y unit (loads on the Y-strong factors)

    meta = dict(
        T=T, N_x=N_x, N_y=N_y, k_ys=k_ys, k_R=k_R, k=k_ys + k_R,
        sigma2=sigma2, noise_kind=noise_kind, lambda_yR_scale=lambda_yR_scale,
        phi=phi, innov_var=innov_var, burn_in=burn_in, variant="baseline",
    )
    return SimDraw(Z=Z, F=F, Lambda=Lambda, C=C, idx_x=idx_x, idx_y=idx_y,
                   target_col=target_col, meta=meta)


def draw(seed, dims, variant="baseline", loadings_seed=None):
    """Convenience wrapper: one RNG seed -> one SimDraw.

    dims: dict with T, N_x, N_y, and optionally k_ys, k_R, sigma2.
    variant: 'baseline' (default) or 'weak_leakage' (turns on Lambda_y,R).
    loadings_seed: if given, draw the loadings Lambda from a SEPARATE rng so that panels
        sharing this value share Lambda (sample splitting), while F and e follow `seed`.
    """
    rng = np.random.default_rng(seed)
    kwargs = dict(dims)
    if variant == "weak_leakage" and "lambda_yR_scale" not in kwargs:
        kwargs["lambda_yR_scale"] = 1.0 / kwargs["N_y"]
    if loadings_seed is not None:
        lam_rng = np.random.default_rng(loadings_seed)
        Lambda, idx_x, idx_y = make_loadings(
            lam_rng, kwargs["N_x"], kwargs["N_y"],
            k_ys=kwargs.get("k_ys", 2), k_R=kwargs.get("k_R", 2),
            lambda_yR_scale=kwargs.get("lambda_yR_scale", 0.0),
        )
        return simulate_panel(rng, lambda_pack=(Lambda, idx_x, idx_y), **kwargs)
    return simulate_panel(rng, **kwargs)


def save_draw(draw_obj: SimDraw, out_dir):
    """Persist a draw (observable + ground truth) to out_dir as arrays.npz + meta.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "draw.npz",
        Z=draw_obj.Z, F=draw_obj.F, Lambda=draw_obj.Lambda, C=draw_obj.C,
        idx_x=draw_obj.idx_x, idx_y=draw_obj.idx_y,
        target_col=np.array(draw_obj.target_col),
    )
    (out_dir / "meta.json").write_text(json.dumps(draw_obj.meta, indent=2))
    return out_dir


def load_draw(path) -> SimDraw:
    """Inverse of save_draw."""
    path = Path(path)
    npz_path = path / "draw.npz" if path.is_dir() else path
    base = npz_path.parent
    with np.load(npz_path) as d:
        arrays = {k: d[k] for k in d.files}
    meta_path = base / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return SimDraw(
        Z=arrays["Z"], F=arrays["F"], Lambda=arrays["Lambda"], C=arrays["C"],
        idx_x=arrays["idx_x"], idx_y=arrays["idx_y"],
        target_col=int(arrays["target_col"]), meta=meta,
    )
