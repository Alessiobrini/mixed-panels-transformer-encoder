# theorysim (JBES theory-validation add-on) -- operator scaling / restriction / IO.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Frozen-operator utilities for the validated estimator.

The deterministic, closed-form-testable pieces live here: the c_N / c_T scaling that
puts the operators on the factor-model scale, the within-block restriction used by E2/E3,
the oracle (population) operators, and disk IO. The learned-operator path (train the
axial model, average and freeze A_z, B) is added with the model in the next build step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def scale_operators(S_z, S_tau, N, T):
    """Apply the global scalars c_N, c_T so the operators match the effective dimension.

    c_N = sqrt(N / ||S_z||_F^2), c_T = sqrt(T / ||S_tau||_F^2). After scaling,
    tr(A_z^T A_z) = N and tr(B^T B) = T exactly (design Sec 4.2).
    """
    c_N = np.sqrt(N / float((S_z ** 2).sum()))
    c_T = np.sqrt(T / float((S_tau ** 2).sum()))
    return c_N * S_z, c_T * S_tau


def within_block_restrict(A_z, idx_x, idx_y):
    """Zero the X<->Y cross terms of A_z (block-diagonal restriction for E2/E3)."""
    A = A_z.copy()
    idx_x = np.asarray(idx_x)
    idx_y = np.asarray(idx_y)
    A[np.ix_(idx_x, idx_y)] = 0.0
    A[np.ix_(idx_y, idx_x)] = 0.0
    return A


def clip_operator(A, kappa, dim, iters=2):
    """Project a frozen operator so both halves of A.7 hold: clip its singular values at
    kappa (bounded op-norm, first half of A.7) and re-apply the trace scaling so
    tr(A^T A) = dim (finite effective dimension, second half). Alternate clip/rescale
    `iters` times; the fixed point has op-norm ~ kappa and trace = dim. Deterministic
    function of A, so the Remark-1 freeze convention is intact. This is the attention
    analogue of eigenvalue clipping for covariance matrices. Same kappa at every N keeps
    the op-norm bounded as the panel grows, unlike the raw learned operator.
    """
    M = np.asarray(A, float).copy()
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        s = np.minimum(s, float(kappa))
        M = (U * s) @ Vt
        c = np.sqrt(dim / float((M ** 2).sum()))
        M = c * M
    return M


def blend_operator(A, kappa0, delta, dim):
    """Corrected projection into A.7's domain: clip the singular values of A at kappa0, add
    delta*I, then trace-rescale once so tr(.^T .) = dim. A bare clip/rescale has no fixed point
    for an effectively low-rank spectrum (clipping removes top mass but cannot create the tail
    mass the trace half needs), so it lands on the A.7 boundary. The delta*I addition restores
    that tail mass, giving every variable a baseline self-weight; it is a shrinkage of the learned
    attention toward the identity. The result satisfies all three diagnostics: op-norm bounded,
    tr/dim = c_A, and PR/dim >= c_A/kappa0^2 (the two halves of A.7 jointly, i.e. anti-concentration).
    Deterministic in A, so the Remark-1 freeze convention is intact.
    """
    M = np.asarray(A, float)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s = np.minimum(s, float(kappa0))
    M = (U * s) @ Vt + float(delta) * np.eye(dim)
    return M * np.sqrt(dim / float((M ** 2).sum()))


def a7_diagnostics(A):
    """Return (op_norm, tr/N, PR/N) for A: the A.7 quantities. PR = tr(A^T A)^2 / ||A^T A||_F^2
    is the participation ratio (effective dimension); PR/N in (0,1], ~1 for the identity."""
    AtA = A.T @ A
    tr = float(np.trace(AtA))
    fro2 = float((AtA ** 2).sum())
    N = A.shape[0]
    return float(np.linalg.norm(A, 2)), tr / N, (tr ** 2 / fro2) / N


def oracle_operators(N, T):
    """Population/oracle operators: identity on both axes (robustness arm 1)."""
    return np.eye(N), np.eye(T)


def parameter_free_operators(oos_draws, N, T, temp=0.1):
    """Paper's simplest-case attention (Sec 3.3, Q=K=V=Z): no learned projection.

    A_z = softmax(Z^T Z / (T*temp)) (N x N), B = softmax(Z Z^T / (N*temp)) (T x T), averaged
    over the OOS panels (same Lambda) then scaled. The Gram is normalized by the token
    dimension (T for A_z, N for B) to keep the logits O(1); temp<1 sharpens toward identity.
    Deterministic function of the data, so independent of the estimation panel's (F,e).
    """
    from scipy.special import softmax

    S_z = np.zeros((N, N))
    S_tau = np.zeros((T, T))
    for d in oos_draws:
        Z = d.Z
        S_z += softmax((Z.T @ Z / T) / temp, axis=1)
        S_tau += softmax((Z @ Z.T / N) / temp, axis=1)
    n = len(oos_draws)
    return scale_operators(S_z / n, S_tau / n, N, T)


def freeze_to_disk(A_z, B, out_dir):
    """Persist frozen operators to out_dir as Az.npy and B.npy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "Az.npy", A_z)
    np.save(out_dir / "B.npy", B)
    return out_dir


def load_operators(path):
    """Load frozen operators (A_z, B) from a directory."""
    path = Path(path)
    return np.load(path / "Az.npy"), np.load(path / "B.npy")


# --- Learned-operator path (axial model). torch imported lazily. ---------------

def train_operators(train_draw, k=4, d_model=16, use_nonlinearity=False,
                    epochs=200, lr=1e-3, seed=0, device="cpu", verbose=False,
                    objective="forecast", tie_qk=False, attn_temp=1.0,
                    target_mode="mask"):
    """Train the axial model on one training panel; return the trained model.

    objective="forecast": predict the target column (the masked-target objective, which
        tends to concentrate attention on a few hub variables).
    objective="reconstruct": linear autoencoder on the whole panel (Option B), which
        pushes the attention toward a well-spread, A.7-satisfying operator.
    The trained softmax attentions are the learned operators (extract_and_average freezes them).
    """
    import torch
    from src.theorysim.axial_model import AxialAttentionMPTE, fit_forecast, fit_reconstruct

    torch.manual_seed(seed)
    T, N = train_draw.Z.shape
    Z = torch.as_tensor(train_draw.Z, dtype=torch.float32, device=device)
    # For reconstruction we keep every column (no target mask); for forecast we mask it.
    tcol = None if objective == "reconstruct" else int(train_draw.target_col)
    model = AxialAttentionMPTE(
        N=N, T=T, k=k, d_model=d_model, nhead=1, num_layers=1,
        use_nonlinearity=use_nonlinearity, target_col=tcol,
        tie_qk=tie_qk, attn_temp=attn_temp, target_mode=target_mode,
    ).to(device)
    if objective == "reconstruct":
        model, _ = fit_reconstruct(model, Z, epochs=epochs, lr=lr, verbose=verbose)
    else:
        target = torch.as_tensor(train_draw.Z[:, train_draw.target_col],
                                 dtype=torch.float32, device=device)
        model, _ = fit_forecast(model, Z, target, epochs=epochs, lr=lr, verbose=verbose)
    return model


def extract_and_average(model, oos_draws, device="cpu"):
    """Average the model's softmax attentions over OOS draws -> (S_z, S_tau) (unscaled).

    Both are row-stochastic (averages of row-stochastic softmax matrices stay so).
    """
    import torch

    Az_sum = None
    B_sum = None
    n = 0
    for d in oos_draws:
        Z = torch.as_tensor(d.Z, dtype=torch.float32, device=device)
        A_z, B = model.attention_matrices(Z)
        A_z = A_z.cpu().numpy()
        B = B.cpu().numpy()
        Az_sum = A_z if Az_sum is None else Az_sum + A_z
        B_sum = B if B_sum is None else B_sum + B
        n += 1
    return Az_sum / n, B_sum / n
