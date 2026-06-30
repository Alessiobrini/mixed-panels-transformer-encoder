# theorysim (JBES theory-validation add-on) -- DGP ground-truth tests.
"""Tests that the union-factor DGP satisfies its design identities by construction."""

import os
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.theorysim import dgp


DIMS = dict(T=120, N_x=40, N_y=20, k_ys=2, k_R=2, sigma2=2.0)


def test_common_component_identity():
    d = dgp.draw(seed=0, dims=DIMS)
    # C == F Lambda^T exactly.
    assert np.allclose(d.C, d.F @ d.Lambda.T, atol=0, rtol=0)
    assert d.Z.shape == (DIMS["T"], DIMS["N_x"] + DIMS["N_y"])
    assert d.F.shape == (DIMS["T"], DIMS["k_ys"] + DIMS["k_R"])
    assert d.Lambda.shape == (DIMS["N_x"] + DIMS["N_y"], DIMS["k_ys"] + DIMS["k_R"])


def test_loading_block_structure():
    d = dgp.draw(seed=1, dims=DIMS)
    k_ys = DIMS["k_ys"]
    # X loads on Y-strong factor 1 only: columns 1..k_ys-1 are zero on X rows.
    assert np.allclose(d.Lambda[np.ix_(d.idx_x, np.arange(1, k_ys))], 0.0)
    # Baseline Lambda_y,R = 0: Y rows, rest columns are zero.
    assert np.allclose(d.Lambda[np.ix_(d.idx_y, np.arange(k_ys, d.Lambda.shape[1]))], 0.0)
    # Cross signal Sigma_{ys,x} has rank 1 < k_ys (Assumption B.3).
    Lx_ys = d.Lambda[np.ix_(d.idx_x, np.arange(0, k_ys))]
    assert np.linalg.matrix_rank(Lx_ys.T @ Lx_ys) == 1
    # The designated target is a Y column.
    assert d.target_col in set(d.idx_y.tolist())


def test_weak_leakage_variant_turns_on_yR():
    d = dgp.draw(seed=2, dims=DIMS, variant="weak_leakage")
    k_ys = DIMS["k_ys"]
    block = d.Lambda[np.ix_(d.idx_y, np.arange(k_ys, d.Lambda.shape[1]))]
    assert not np.allclose(block, 0.0)  # leakage is now nonzero


def test_factor_unconditional_variance_near_one():
    # Long panel: each factor's sample variance should be ~1 (0.75 / (1 - 0.5^2)).
    d = dgp.draw(seed=3, dims=dict(T=20000, N_x=10, N_y=10, k_ys=2, k_R=2, sigma2=2.0))
    v = d.F.var(axis=0)
    assert np.all(np.abs(v - 1.0) < 0.1)


def test_reproducibility():
    d1 = dgp.draw(seed=7, dims=DIMS)
    d2 = dgp.draw(seed=7, dims=DIMS)
    assert np.array_equal(d1.Z, d2.Z)
    assert np.array_equal(d1.F, d2.F)
    assert np.array_equal(d1.Lambda, d2.Lambda)


def test_save_load_roundtrip(tmp_path):
    d = dgp.draw(seed=9, dims=DIMS)
    dgp.save_draw(d, tmp_path)
    r = dgp.load_draw(tmp_path)
    for name in ("Z", "F", "Lambda", "C", "idx_x", "idx_y"):
        assert np.array_equal(getattr(d, name), getattr(r, name))
    assert d.target_col == r.target_col
    assert r.meta["variant"] == "baseline"
