# theorysim (JBES theory-validation add-on) -- the validated two-step PCA estimator.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""The estimator the inferential theory describes: PCA on the attention-weighted panel.

Z-tilde = B Z A_z, then PCA: top-k eigenvectors of Z-tilde^T Z-tilde give Lambda-hat
(normalized Lambda-hat^T Lambda-hat / N = I_k), F-hat = Z-tilde Lambda-hat / N, and the
common component C-hat = F-hat Lambda-hat^T. This is NOT the transformer's forecast; the
transformer only supplies the frozen attention matrices A_z, B (see operators.py).
"""

from __future__ import annotations

import numpy as np


def attended_panel(B, Z, A_z):
    """Form Z-tilde = B Z A_z."""
    return B @ Z @ A_z


def pca_factors(Ztilde, k):
    """Two-step PCA on the attended panel.

    Returns (Lambda_hat (N x k), F_hat (T x k), C_hat (T x N)) with the identifying
    normalization Lambda_hat^T Lambda_hat / N = I_k.
    """
    N = Ztilde.shape[1]
    gram = Ztilde.T @ Ztilde                      # (N, N)
    # eigh returns ascending eigenvalues; take the top-k, ordered descending.
    eigvals, eigvecs = np.linalg.eigh(gram)
    top = eigvecs[:, -k:][:, ::-1]                # (N, k), orthonormal columns
    Lambda_hat = np.sqrt(N) * top                 # Lambda_hat^T Lambda_hat = N I_k
    F_hat = Ztilde @ Lambda_hat / N               # (T, k)
    C_hat = F_hat @ Lambda_hat.T                   # (T, N)
    return Lambda_hat, F_hat, C_hat


def true_attended_targets(A_z, B, Lambda, F):
    """The population objects the PCA estimates, in attended space.

    Z-tilde = B (F Lambda^T + e) A_z = (B F)(A_z^T Lambda)^T + B e A_z, so the attended
    loadings are Lambda^(A) = A_z^T Lambda and the attended factors are F^(B) = B F.
    """
    Lambda_A = A_z.T @ Lambda                      # (N, k)
    F_B = B @ F                                     # (T, k)
    return Lambda_A, F_B


def true_attended_common(A_z, B, C):
    """Attended common component B C A_z that C-hat estimates."""
    return B @ C @ A_z
