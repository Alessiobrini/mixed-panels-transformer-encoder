# theorysim (JBES theory-validation add-on) -- rate formula tests.
"""alpha_bar must reduce to the classical PCA rate 1/T + 2/N at A_z=I, B=I."""

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.theorysim import rates


def test_alpha_bar_reduces_to_classical_rate():
    for N, T in [(50, 50), (100, 200), (400, 50), (200, 200)]:
        A_z, B = np.eye(N), np.eye(T)
        expected = 1.0 / T + 2.0 / N
        assert np.isclose(rates.alpha_bar(A_z, B, N, T), expected, rtol=1e-12)


def test_alpha_bar_positive_and_monotone_in_dimensions():
    # Larger N, T => smaller rate (identity operators).
    a_small = rates.alpha_bar(np.eye(50), np.eye(50), 50, 50)
    a_large = rates.alpha_bar(np.eye(400), np.eye(400), 400, 400)
    assert a_large < a_small
    assert a_small > 0
