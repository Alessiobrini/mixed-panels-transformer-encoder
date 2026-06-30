# theorysim (JBES theory-validation add-on) -- operator scaling/restriction tests.
"""Closed-form checks on the deterministic operator utilities."""

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.theorysim import operators, rates


def _random_row_stochastic(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.random((n, n))
    return M / M.sum(axis=1, keepdims=True)


def test_row_stochastic_helper():
    S = _random_row_stochastic(12, 0)
    assert rates.is_row_stochastic(S)
    assert not rates.is_row_stochastic(2 * S)  # rows now sum to 2


def test_scaling_hits_effective_dimension():
    N, T = 30, 25
    S_z = _random_row_stochastic(N, 1)
    S_tau = _random_row_stochastic(T, 2)
    A_z, B = operators.scale_operators(S_z, S_tau, N, T)
    assert np.isclose(np.trace(A_z.T @ A_z), N)
    assert np.isclose(np.trace(B.T @ B), T)


def test_oracle_scaling_is_identity_rate():
    # Oracle operators already satisfy tr = N and tr = T (no rescaling needed).
    N, T = 20, 15
    A_z, B = operators.oracle_operators(N, T)
    assert np.isclose(np.trace(A_z.T @ A_z), N)
    assert np.isclose(np.trace(B.T @ B), T)


def test_within_block_restrict_zeros_cross_terms():
    N_x, N_y = 6, 4
    N = N_x + N_y
    A = _random_row_stochastic(N, 3)
    idx_x = np.arange(N_x)
    idx_y = np.arange(N_x, N)
    Aw = operators.within_block_restrict(A, idx_x, idx_y)
    assert np.allclose(Aw[np.ix_(idx_x, idx_y)], 0.0)
    assert np.allclose(Aw[np.ix_(idx_y, idx_x)], 0.0)
    # Within-block entries are untouched.
    assert np.allclose(Aw[np.ix_(idx_x, idx_x)], A[np.ix_(idx_x, idx_x)])
    assert np.allclose(Aw[np.ix_(idx_y, idx_y)], A[np.ix_(idx_y, idx_y)])


def test_freeze_load_roundtrip(tmp_path):
    A_z, B = operators.oracle_operators(8, 6)
    operators.freeze_to_disk(A_z, B, tmp_path)
    A2, B2 = operators.load_operators(tmp_path)
    assert np.array_equal(A_z, A2)
    assert np.array_equal(B, B2)
