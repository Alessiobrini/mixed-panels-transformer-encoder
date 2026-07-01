# theorysim (JBES theory-validation add-on) -- least-squares rotation alignment + errors.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Alignment and estimation-error metrics for E1.

PCA recovers the factor space only up to a general invertible rotation H, so before
comparing estimated to true loadings/factors we estimate H by least squares (NOT
orthogonal Procrustes: Theorem 1's H is general invertible). The common component
C = Lambda^T F is rotation-invariant, so its error needs no alignment.
"""

from __future__ import annotations

import numpy as np


def ls_align(est, truth):
    """Least-squares rotation H such that est ~ truth @ H^T.

    H_hat = est^T truth (truth^T truth)^{-1}, the multivariate regression coefficient of
    estimated on true (design E1, "Aligning is just a regression").
    """
    return est.T @ truth @ np.linalg.inv(truth.T @ truth)


def loading_error(Lambda_hat, Lambda_A):
    """e_Lambda = (1/N) || Lambda_hat - Lambda_A H_hat^T ||_F^2 after LS alignment."""
    H = ls_align(Lambda_hat, Lambda_A)
    resid = Lambda_hat - Lambda_A @ H.T
    return float((resid ** 2).sum() / Lambda_hat.shape[0])


def factor_error(F_hat, F_B):
    """e_F = (1/T) || F_hat - F_B G_hat^T ||_F^2 after LS alignment."""
    G = ls_align(F_hat, F_B)
    resid = F_hat - F_B @ G.T
    return float((resid ** 2).sum() / F_hat.shape[0])


def loading_error_relative(Lambda_hat, Lambda_A):
    """Scale-invariant loading error after LS alignment: ||resid||_F^2 / ||Lambda_hat||_F^2."""
    H = ls_align(Lambda_hat, Lambda_A)
    resid = Lambda_hat - Lambda_A @ H.T
    return float((resid ** 2).sum() / (Lambda_hat ** 2).sum())


def factor_error_relative(F_hat, F_B):
    """Scale-invariant factor error after LS alignment: ||resid||_F^2 / ||F_hat||_F^2."""
    G = ls_align(F_hat, F_B)
    resid = F_hat - F_B @ G.T
    return float((resid ** 2).sum() / (F_hat ** 2).sum())


def common_error(C_hat, C_true):
    """e_C = (1/(N T)) || C_hat - C_true ||_F^2 (alignment-free, rotation-invariant).

    NOTE: this ABSOLUTE error is not comparable across operator arms, because the target
    C_true = B C A_z carries an operator-dependent scale (a concentrated A_z inflates it,
    and a growing op-norm inflates it with N). Use common_error_relative for cross-arm and
    cross-N comparison.
    """
    return float(((C_hat - C_true) ** 2).mean())


def common_error_relative(C_hat, C_true):
    """Scale-invariant common-component error || C_hat - C_true ||_F^2 / || C_true ||_F^2.

    Removes the operator-dependent scale of C_true, so the consistency check (does it go to
    0 as N,T grow?) and the cross-arm comparison are valid. This is the primary E1 metric.
    """
    return float(((C_hat - C_true) ** 2).sum() / (C_true ** 2).sum())


def canonical_correlations(X, Y):
    """Canonical correlations between the column spaces of X (T x p) and Y (T x q).

    Returns min(p, q) values in [0, 1], descending. Used by E4 to compare the nonlinear
    encoder's latent subspace to the true (attended) factor subspace: all ~1 means the
    same factor space is recovered, regardless of rotation/scale.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return np.clip(np.sort(s)[::-1], 0.0, 1.0)


def procrustes_orthogonal(est, truth):
    """Orthogonal Procrustes rotation R minimizing || est - truth R^T ||_F.

    NOTE: correct ONLY for the Y-strong block of Theorems 2-3, where the rotation is
    orthogonal by construction. Do NOT use for the general E1 alignment (use ls_align).
    """
    u, _, vt = np.linalg.svd(truth.T @ est)
    return u @ vt
