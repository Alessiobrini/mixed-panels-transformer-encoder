# theorysim (JBES theory-validation add-on) -- estimator/alignment ground-truth tests.
"""Noiseless oracle recovery: the estimator must return the true factor space exactly."""

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.theorysim import dgp, estimator, alignment, operators


DIMS = dict(T=150, N_x=40, N_y=20, k_ys=2, k_R=2, sigma2=0.0)  # noiseless
K = DIMS["k_ys"] + DIMS["k_R"]


def _noiseless_oracle():
    d = dgp.draw(seed=0, dims=DIMS)
    N, T = d.Z.shape[1], d.Z.shape[0]
    A_z, B = operators.oracle_operators(N, T)
    Ztilde = estimator.attended_panel(B, d.Z, A_z)
    Lambda_hat, F_hat, C_hat = estimator.pca_factors(Ztilde, K)
    Lambda_A, F_B = estimator.true_attended_targets(A_z, B, d.Lambda, d.F)
    C_true = estimator.true_attended_common(A_z, B, d.C)
    return d, A_z, B, Lambda_hat, F_hat, C_hat, Lambda_A, F_B, C_true


def test_normalization():
    *_, Lambda_hat, _, _, _, _, _ = _noiseless_oracle()
    N = Lambda_hat.shape[0]
    assert np.allclose(Lambda_hat.T @ Lambda_hat / N, np.eye(K), atol=1e-8)


def test_common_component_recovered_exactly():
    _, _, _, _, _, C_hat, _, _, C_true = _noiseless_oracle()
    # Rotation-invariant, so no alignment needed; noiseless => machine precision.
    assert alignment.common_error(C_hat, C_true) < 1e-18


def test_loadings_and_factors_recovered_after_alignment():
    _, _, _, Lambda_hat, F_hat, _, Lambda_A, F_B, _ = _noiseless_oracle()
    assert alignment.loading_error(Lambda_hat, Lambda_A) < 1e-16
    assert alignment.factor_error(F_hat, F_B) < 1e-16


def test_error_decreases_with_sample_size():
    # With noise, the common-component error should fall as N and T grow.
    def e_c(N_x, N_y, T, seed):
        d = dgp.draw(seed=seed, dims=dict(T=T, N_x=N_x, N_y=N_y, k_ys=2, k_R=2, sigma2=2.0))
        A_z, B = operators.oracle_operators(d.Z.shape[1], T)
        Zt = estimator.attended_panel(B, d.Z, A_z)
        _, _, C_hat = estimator.pca_factors(Zt, K)
        C_true = estimator.true_attended_common(A_z, B, d.C)
        return alignment.common_error(C_hat, C_true)

    small = np.mean([e_c(20, 10, 50, s) for s in range(10)])
    large = np.mean([e_c(160, 80, 400, s) for s in range(10)])
    assert large < small
