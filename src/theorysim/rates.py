# theorysim (JBES theory-validation add-on) -- theoretical rate + assumption diagnostics.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Theorem-1 rate alpha-bar and assumption diagnostics (A.1, A.7, B.1, B.3).

alpha_bar is computed directly from the operators (A_z, B) and (N, T), with no free
constant, so a unit slope of log(error) vs log(alpha_bar) is unambiguous. At A_z=I_N,
B=I_T it reduces to the classical PCA rate 1/T + 2/N.
"""

from __future__ import annotations

import numpy as np


def alpha_bar(A_z, B, N, T):
    """Full three-term Theorem-1 rate.

    alpha_bar = tr(A_z^T A_z)/(N T)
              + (||A_z^T A_z||_F^2 / N^2)(||B||_F^4 / T^2)
              + ||B||_F^2/(N T).
    """
    AtA = A_z.T @ A_z
    tr_AtA = np.trace(AtA)
    fro_AtA_sq = float((AtA ** 2).sum())          # ||A_z^T A_z||_F^2
    fro_B_sq = float((B ** 2).sum())              # ||B||_F^2
    t1 = tr_AtA / (N * T)
    t2 = (fro_AtA_sq / N ** 2) * (fro_B_sq ** 2 / T ** 2)
    t3 = fro_B_sq / (N * T)
    return float(t1 + t2 + t3)


def is_row_stochastic(M, tol=1e-6):
    """True if every row of M is nonnegative and sums to 1 (within tol)."""
    M = np.asarray(M)
    return bool(np.all(M >= -tol) and np.allclose(M.sum(axis=1), 1.0, atol=tol))


def diagnostics(A_z, B, F=None, Lambda=None, idx_y=None, k_ys=2):
    """Assumption diagnostics on the realized operators.

    Returns op-norm and effective-dimension trace of A_z (A.7); if F/Lambda provided,
    the spectrum of T^{-1} F^T B^T B F (A.1) and the attended Y-block Gram eigen-count
    and cross-Gram rank (B.1/B.3).
    """
    out = {}
    out["Az_op_norm"] = float(np.linalg.norm(A_z, 2))
    out["Az_eff_dim_tr"] = float(np.trace(A_z.T @ A_z))   # tr(A_z^T A_z) = N_eff
    out["B_fro_sq_over_T"] = float((B ** 2).sum() / B.shape[0])

    if F is not None:
        T = F.shape[0]
        BF = B @ F
        M = (BF.T @ BF) / T                                # T^{-1} F^T B^T B F
        out["A1_spectrum"] = np.linalg.eigvalsh(M).tolist()

    if Lambda is not None and idx_y is not None:
        Lambda_A = A_z.T @ Lambda
        Ly_s = Lambda_A[np.asarray(idx_y)][:, :k_ys]       # attended Y-block, Y-strong
        gram_y = Ly_s.T @ Ly_s
        out["B1_Yblock_eigs"] = np.linalg.eigvalsh(gram_y).tolist()
    return out
