# theorysim (JBES theory-validation add-on) -- CCA / subspace-recovery tests.
"""Canonical correlations: a rotated copy gives all 1; an independent matrix gives <1;
the linear PCA estimator recovers the true factor subspace at low noise."""

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.theorysim import dgp, estimator, alignment, operators


def test_cca_rotation_is_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))  # random rotation
    cc = alignment.canonical_correlations(X, X @ Q)
    assert np.allclose(cc, 1.0, atol=1e-8)


def test_cca_independent_is_below_one():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 4))
    Y = rng.standard_normal((300, 4))
    cc = alignment.canonical_correlations(X, Y)
    assert np.all(cc < 0.9)  # independent => far from 1


def test_linear_pca_recovers_factor_subspace_low_noise():
    # Low-noise oracle: linear PCA factors should be ~collinear with the true factors.
    d = dgp.draw(seed=2, dims=dict(T=400, N_x=80, N_y=40, k_ys=2, k_R=2, sigma2=0.05))
    N, T = d.Z.shape[1], d.Z.shape[0]
    A_z, B = operators.oracle_operators(N, T)
    Ztilde = estimator.attended_panel(B, d.Z, A_z)
    _, F_hat, _ = estimator.pca_factors(Ztilde, 4)
    cc = alignment.canonical_correlations(F_hat, B @ d.F)
    assert np.all(cc > 0.99)
