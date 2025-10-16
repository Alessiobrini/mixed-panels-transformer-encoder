import os
import time
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.data import simulate_to_long as sim


@pytest.fixture
def small_config():
    return {
        "T": 200,
        "k": 3,
        "p_x": 4,
        "p_y": 2,
        "r": 3,
        "Lx": 1,
        "Ly": 1,
        "burn_in": 60,
        "link_out": 3,
    }


def run_simulation(
    cfg,
    *,
    seed=123,
    q_fx=0,
    q_fy=0,
    noise_kind="t",
    nonlinearity="identity",
    output_std_match=False,
    intensity=None,
):
    rng = np.random.RandomState(seed)
    link = sim.make_link(
        nonlinearity,
        q=cfg["k"],
        out_dim=cfg.get("link_out", cfg["k"]),
        rng=rng,
        output_std_match=output_std_match,
        intensity=intensity,
    )

    total_steps = cfg["T"] + cfg["burn_in"]
    F_full, _ = sim.simulate_latent_VAR2(total_steps, cfg["k"], rng=rng, burn_in=cfg["burn_in"])
    gF_full = link(F_full)

    X_full = sim.simulate_hf_block(
        F_full,
        p_x=cfg["p_x"],
        Lx=cfg["Lx"],
        link=link,
        rng=rng,
        q_fx=q_fx,
        noise_kind=noise_kind,
        gF=gF_full,
    )

    idx_q_full, Y_full = sim.simulate_lf_block(
        F_full,
        r=cfg["r"],
        p_y=cfg["p_y"],
        Ly=cfg["Ly"],
        link=link,
        rng=rng,
        q_fy=q_fy,
        gF=gF_full,
    )

    burn = cfg["burn_in"]
    mask_q = idx_q_full >= burn

    return {
        "F": F_full[burn:],
        "gF": gF_full[burn:],
        "X": X_full[burn:],
        "idx_q": idx_q_full[mask_q] - burn,
        "Y": Y_full[mask_q],
        "F_full": F_full,
        "gF_full": gF_full,
        "X_full": X_full,
        "idx_q_full": idx_q_full,
        "Y_full": Y_full,
    }


def spectral_radius(mats):
    if not mats:
        return 0.0
    p = mats[0].shape[0]
    L = len(mats)
    companion = np.zeros((p * L, p * L))
    companion[:p, : p * L] = np.hstack(mats)
    if L > 1:
        companion[p:, :-p] = np.eye(p * (L - 1))
    vals = np.linalg.eigvals(companion)
    return float(np.max(np.abs(vals)))


def compute_factor_terms_X(gF, Lambda):
    if Lambda.size == 0:
        return np.zeros((len(gF), 0))
    T = gF.shape[0]
    out_dim = Lambda.shape[1]
    out = np.zeros((T, out_dim))
    for t in range(T):
        term = np.zeros(out_dim)
        for lag in range(Lambda.shape[0]):
            if t - lag >= 0:
                term += Lambda[lag] @ gF[t - lag]
        out[t] = term
    return out


def compute_factor_terms_Y(gF, idx_q, Lambda):
    if Lambda.size == 0:
        return np.zeros((len(idx_q), 0))
    out_dim = Lambda.shape[1]
    out = np.zeros((len(idx_q), out_dim))
    for i, t in enumerate(idx_q):
        term = np.zeros(out_dim)
        for lag in range(Lambda.shape[0]):
            if t - lag >= 0:
                term += Lambda[lag] @ gF[t - lag]
        out[i] = term
    return out


def compute_residuals_X(X, gF, A, Lambda):
    Lx = len(A)
    residuals = []
    for t in range(X.shape[0]):
        ar = np.zeros(X.shape[1])
        for lag in range(Lx):
            idx = t - (lag + 1)
            if idx >= 0:
                ar += A[lag] @ X[idx]
        fac = np.zeros(X.shape[1])
        for lag in range(Lambda.shape[0]):
            idx = t - lag
            if idx >= 0:
                fac += Lambda[lag] @ gF[idx]
        residuals.append(X[t] - ar - fac)
    drop = max(Lx, max(Lambda.shape[0] - 1, 0))
    if drop:
        residuals = residuals[drop:]
    return np.asarray(residuals)


def compute_residuals_Y(Y, gF, idx_q, C, Lambda):
    Ly = len(C)
    residuals = []
    for i in range(Y.shape[0]):
        ar = np.zeros(Y.shape[1])
        for lag in range(Ly):
            idx = i - (lag + 1)
            if idx >= 0:
                ar += C[lag] @ Y[idx]
        fac = np.zeros(Y.shape[1])
        for lag in range(Lambda.shape[0]):
            idx = idx_q[i] - lag
            if idx >= 0:
                fac += Lambda[lag] @ gF[idx]
        residuals.append(Y[i] - ar - fac)
    drop = max(Ly, max(Lambda.shape[0] - 1, 0))
    if drop:
        residuals = residuals[drop:]
    return np.asarray(residuals)


def kurtosis(values):
    centered = values - values.mean(axis=0, keepdims=True)
    m2 = np.mean(centered ** 2, axis=0)
    m4 = np.mean(centered ** 4, axis=0)
    return np.mean(m4 / (m2 ** 2))


def test_shapes_and_alignment(small_config):
    res = run_simulation(small_config, seed=7, q_fx=1, q_fy=1)
    assert res["F"].shape == (small_config["T"], small_config["k"])
    assert res["X"].shape == (small_config["T"], small_config["p_x"])
    expected_len = small_config["T"] // small_config["r"]
    assert res["Y"].shape == (expected_len, small_config["p_y"])
    expected_idx = np.arange(small_config["r"] - 1, small_config["T"], small_config["r"])
    np.testing.assert_array_equal(res["idx_q"], expected_idx)


def test_reproducibility_same_seed(small_config):
    res1 = run_simulation(small_config, seed=11, q_fx=1, q_fy=1)
    res2 = run_simulation(small_config, seed=11, q_fx=1, q_fy=1)
    for key in ("F", "X", "Y"):
        np.testing.assert_array_equal(res1[key], res2[key])

    res3 = run_simulation(small_config, seed=12, q_fx=1, q_fy=1)
    assert not np.allclose(res1["X"], res3["X"])
    assert not np.allclose(res1["Y"], res3["Y"])


def test_ar_structure_and_stability(small_config, monkeypatch):
    captured = {}
    original_rescale = sim._rescale_var_mats

    def record_rescale(mats, target=0.99):
        scaled, rho_before, rho_after = original_rescale(mats, target)
        if mats:
            key = "A" if mats[0].shape[0] == small_config["p_x"] else "C"
            captured[key] = scaled
            captured[f"{key}_rho"] = (rho_before, rho_after)
        return scaled, rho_before, rho_after

    monkeypatch.setattr(sim, "_rescale_var_mats", record_rescale)

    res = run_simulation(small_config, seed=17, q_fx=1, q_fy=1)
    assert len(captured["A"]) == small_config["Lx"] == 1
    assert len(captured["C"]) == small_config["Ly"] == 1
    assert spectral_radius(captured["A"]) <= 0.99 + 1e-6
    assert spectral_radius(captured["C"]) <= 0.99 + 1e-6


def test_factor_lag_contributions_change_output(small_config):
    base = run_simulation(small_config, seed=23, q_fx=0, q_fy=0)
    richer = run_simulation(small_config, seed=23, q_fx=2, q_fy=2)
    assert not np.allclose(base["X"], richer["X"])
    assert not np.allclose(base["Y"], richer["Y"])


def test_zero_factor_loadings_reduce_to_ar(monkeypatch, small_config):
    captured = {}
    original_rescale = sim._rescale_var_mats
    original_almon = sim.make_almon_lag_matrices

    def record_rescale(mats, target=0.99):
        scaled, rho_before, rho_after = original_rescale(mats, target)
        if mats:
            key = "A" if mats[0].shape[0] == small_config["p_x"] else "C"
            captured[key] = scaled
            captured[f"{key}_rho"] = (rho_before, rho_after)
        return scaled, rho_before, rho_after

    def zero_almon(num_lags, out_dim, factor_dim, rng, degree=2):
        mats = original_almon(num_lags, out_dim, factor_dim, rng, degree=degree)
        key = "Lambda_fx" if out_dim == small_config["p_x"] else "Lambda_fy"
        captured[key] = np.zeros_like(mats)
        return captured[key]

    monkeypatch.setattr(sim, "_rescale_var_mats", record_rescale)
    monkeypatch.setattr(sim, "make_almon_lag_matrices", zero_almon)

    res = run_simulation(small_config, seed=29, q_fx=2, q_fy=2)
    fac_x = compute_factor_terms_X(res["gF"], captured["Lambda_fx"])
    assert fac_x.shape[1] == small_config["p_x"]
    assert np.allclose(fac_x, 0.0)
    fac_y = compute_factor_terms_Y(res["gF"], res["idx_q"], captured["Lambda_fy"])
    assert fac_y.shape[1] == small_config["p_y"]
    assert np.allclose(fac_y, 0.0)

    residuals = compute_residuals_X(res["X"], res["gF"], captured["A"], captured["Lambda_fx"])
    assert residuals.shape[0] > 0


def test_almon_decay(monkeypatch, small_config):
    captured = {}
    original_almon = sim.make_almon_lag_matrices

    def record_almon(num_lags, out_dim, factor_dim, rng, degree=2):
        mats = original_almon(num_lags, out_dim, factor_dim, rng, degree=degree)
        if num_lags >= 3:
            key = "fx" if out_dim == small_config["p_x"] else "fy"
            captured[key] = mats[:3]
        return mats

    monkeypatch.setattr(sim, "make_almon_lag_matrices", record_almon)

    run_simulation(small_config, seed=31, q_fx=2, q_fy=2)
    for key in ("fx", "fy"):
        mats = captured[key]
        means = [np.mean(np.abs(mats[i])) for i in range(3)]
        assert means[0] >= means[1] - 1e-8
        assert means[1] >= means[2] - 1e-8


def test_identity_nonlinearity_matches_inputs():
    rng = np.random.RandomState(41)
    F = rng.normal(size=(200, 3))
    link = sim.make_link("identity", q=3, out_dim=3, rng=rng)
    np.testing.assert_allclose(link(F), F)


def test_rbf_output_std_match_preserves_std():
    rng = np.random.RandomState(43)
    F = rng.normal(size=(200, 3))
    link = sim.make_link("rbf", q=3, out_dim=3, rng=rng, output_std_match=True)
    gF = link(F)
    ratios = gF.std(axis=0) / F.std(axis=0)
    assert np.allclose(ratios, np.ones_like(ratios), atol=0.2)


def test_rbf_column_standardization():
    rng = np.random.RandomState(47)
    F = rng.normal(size=(200, 3))
    link = sim.make_link("rbf", q=3, out_dim=3, rng=rng, output_std_match=False)
    gF = link(F)
    stds = gF.std(axis=0)
    np.testing.assert_allclose(stds, np.ones_like(stds), atol=0.1)


def test_rbf_intensity_controls_centers():
    rng = np.random.RandomState(51)
    F = rng.normal(size=(150, 2))
    link = sim.make_link("rbf", q=2, out_dim=2, rng=rng, intensity=7)
    gF = link(F)
    assert gF.shape[1] == 7


def test_noise_kurtosis_behaviour(monkeypatch, small_config):
    captured = {}
    original_rescale = sim._rescale_var_mats
    original_almon = sim.make_almon_lag_matrices

    def record_rescale(mats, target=0.99):
        scaled, rho_before, rho_after = original_rescale(mats, target)
        if mats:
            key = "A" if mats[0].shape[0] == small_config["p_x"] else "C"
            captured[key] = scaled
            captured[f"{key}_rho"] = (rho_before, rho_after)
        return scaled, rho_before, rho_after

    def record_almon(num_lags, out_dim, factor_dim, rng, degree=2):
        mats = original_almon(num_lags, out_dim, factor_dim, rng, degree=degree)
        key = "Lambda_fx" if out_dim == small_config["p_x"] else "Lambda_fy"
        captured[key] = mats
        return mats

    monkeypatch.setattr(sim, "_rescale_var_mats", record_rescale)
    monkeypatch.setattr(sim, "make_almon_lag_matrices", record_almon)

    res_t = run_simulation(small_config, seed=53, q_fx=1, q_fy=1, noise_kind="t")
    resid_x_t = compute_residuals_X(res_t["X"], res_t["gF"], captured["A"], captured["Lambda_fx"])
    kurt_x_t = kurtosis(resid_x_t)
    assert kurt_x_t > 3.0

    resid_y = compute_residuals_Y(res_t["Y"], res_t["gF"], res_t["idx_q"], captured["C"], captured["Lambda_fy"])
    kurt_y = kurtosis(resid_y)
    assert np.isclose(kurt_y, 3.0, atol=0.7)

    captured_gauss = {}

    def record_rescale_gauss(mats, target=0.99):
        scaled, rho_before, rho_after = original_rescale(mats, target)
        if mats:
            key = "A" if mats[0].shape[0] == small_config["p_x"] else "C"
            captured_gauss[key] = scaled
            captured_gauss[f"{key}_rho"] = (rho_before, rho_after)
        return scaled, rho_before, rho_after

    def record_almon_gauss(num_lags, out_dim, factor_dim, rng, degree=2):
        mats = original_almon(num_lags, out_dim, factor_dim, rng, degree=degree)
        key = "Lambda_fx" if out_dim == small_config["p_x"] else "Lambda_fy"
        captured_gauss[key] = mats
        return mats

    monkeypatch.setattr(sim, "_rescale_var_mats", record_rescale_gauss)
    monkeypatch.setattr(sim, "make_almon_lag_matrices", record_almon_gauss)

    res_gauss = run_simulation(small_config, seed=53, q_fx=1, q_fy=1, noise_kind="gaussian")
    resid_x_gauss = compute_residuals_X(res_gauss["X"], res_gauss["gF"], captured_gauss["A"], captured_gauss["Lambda_fx"])
    kurt_x_gauss = kurtosis(resid_x_gauss)
    assert kurt_x_gauss < kurt_x_t


def test_burn_in_handling(small_config):
    res = run_simulation(small_config, seed=59, q_fx=2, q_fy=2)
    assert res["X"].shape[0] == small_config["T"]
    assert res["F_full"].shape[0] == small_config["T"] + small_config["burn_in"]
    assert res["X_full"].shape[0] == small_config["T"] + small_config["burn_in"]
    assert res["idx_q"].min() == small_config["r"] - 1


def test_backward_compatibility_statistics(small_config):
    expected = {
        "F_mean": np.array([-0.06218596, 0.00201632, -0.03128081]),
        "F_std": np.array([1.1762863, 1.12846694, 1.64384969]),
        "X_mean": np.array([ 0.02784908, -0.07572891,  0.00491393, -0.11608128]),
        "X_std": np.array([33.1091632, 16.29098696, 45.41771867, 45.80333223]),
        "Y_mean": np.array([-0.19104351, -0.11323034]),
        "Y_std": np.array([6.92930838, 8.29280537]),
    }

    res = run_simulation(small_config, seed=123, q_fx=0, q_fy=0)
    np.testing.assert_allclose(res["F"].mean(axis=0), expected["F_mean"], atol=1e-7)
    np.testing.assert_allclose(res["F"].std(axis=0), expected["F_std"], atol=1e-7)
    np.testing.assert_allclose(res["X"].mean(axis=0), expected["X_mean"], atol=1e-6)
    np.testing.assert_allclose(res["X"].std(axis=0), expected["X_std"], atol=1e-6)
    np.testing.assert_allclose(res["Y"].mean(axis=0), expected["Y_mean"], atol=1e-6)
    np.testing.assert_allclose(res["Y"].std(axis=0), expected["Y_std"], atol=1e-6)


def test_configuration_validation_errors(small_config):
    rng = np.random.RandomState(71)
    F = rng.normal(size=(20, small_config["k"]))
    link = sim.make_link("identity", q=small_config["k"], out_dim=small_config["k"], rng=rng)

    with pytest.raises(ValueError):
        sim.simulate_hf_block(F, p_x=small_config["p_x"], Lx=small_config["Lx"], link=link, rng=rng, q_fx=0, noise_kind="t", gF=link(F), cov_scale=1.0, student_df=2)

    with pytest.raises(ValueError):
        sim.make_link("unsupported", q=small_config["k"], out_dim=small_config["k"], rng=rng)

    with pytest.raises(ValueError):
        sim.simulate_hf_block(F, p_x=small_config["p_x"], Lx=small_config["Lx"], link=link, rng=rng, q_fx=0, noise_kind="mystery", gF=link(F))


def test_performance_smoke(small_config):
    start = time.perf_counter()
    for offset in range(5):
        run_simulation(small_config, seed=90 + offset, q_fx=1, q_fy=1)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
