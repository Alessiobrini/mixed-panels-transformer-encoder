# theorysim (JBES theory-validation add-on) -- axial model sanity tests.
"""The model's attentions must be row-stochastic (softmax) and correctly shaped, and
training on the target column must reduce the loss."""

import os
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.theorysim import dgp, operators
from src.theorysim.axial_model import AxialAttentionMPTE, fit_forecast


def test_attention_shapes_and_row_stochastic():
    d = dgp.draw(seed=0, dims=dict(T=30, N_x=12, N_y=6, k_ys=2, k_R=2, sigma2=2.0))
    N, T = d.Z.shape[1], d.Z.shape[0]
    model = AxialAttentionMPTE(N=N, T=T, k=4, target_col=int(d.target_col))
    A_z, B = model.attention_matrices(torch.as_tensor(d.Z, dtype=torch.float32))
    assert A_z.shape == (N, N)
    assert B.shape == (T, T)
    assert torch.allclose(A_z.sum(dim=1), torch.ones(N), atol=1e-5)
    assert torch.allclose(B.sum(dim=1), torch.ones(T), atol=1e-5)
    assert torch.all(A_z >= 0) and torch.all(B >= 0)


def test_forward_shape_and_training_reduces_loss():
    d = dgp.draw(seed=1, dims=dict(T=40, N_x=12, N_y=6, k_ys=2, k_R=2, sigma2=2.0))
    N, T = d.Z.shape[1], d.Z.shape[0]
    torch.manual_seed(0)
    model = AxialAttentionMPTE(N=N, T=T, k=4, target_col=int(d.target_col))
    Z = torch.as_tensor(d.Z, dtype=torch.float32)
    target = torch.as_tensor(d.Z[:, d.target_col], dtype=torch.float32)
    pred0 = model(Z)
    assert pred0.shape == (T,)
    loss0 = float(((pred0 - target) ** 2).mean())
    model, last = fit_forecast(model, Z, target, epochs=150, lr=1e-2)
    assert last < loss0  # training reduced the MSE


def test_extract_and_average_is_row_stochastic():
    dims = dict(T=30, N_x=12, N_y=6, k_ys=2, k_R=2, sigma2=2.0)
    train = dgp.draw(seed=2, dims=dims)
    model = operators.train_operators(train, k=4, d_model=16, epochs=30, lr=1e-2, seed=2)
    oos = [dgp.draw(seed=100 + j, dims=dims) for j in range(4)]
    S_z, S_tau = operators.extract_and_average(model, oos)
    assert np.allclose(S_z.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(S_tau.sum(axis=1), 1.0, atol=1e-5)
    # After scaling, the effective-dimension identities hold.
    N, T = S_z.shape[0], S_tau.shape[0]
    A_z, B = operators.scale_operators(S_z, S_tau, N, T)
    assert np.isclose(np.trace(A_z.T @ A_z), N)
    assert np.isclose(np.trace(B.T @ B), T)
