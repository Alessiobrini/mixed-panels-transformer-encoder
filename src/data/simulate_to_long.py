import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pdb

# ensure project root on path so `import src...` works
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.utils.config import Config

# --------- simple, stable latent VAR(2) ----------
def _spectral_radius(A):
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))

def _make_stable_var2(Phi1, Phi2, target=0.9):
    q = Phi1.shape[0]
    C = np.block([[Phi1, Phi2],
                  [np.eye(q), np.zeros((q, q))]])
    rho = _spectral_radius(C)
    if rho >= target:
        Phi1 *= target / rho
        Phi2 *= target / rho
    return Phi1, Phi2


def _draw_signed_uniform(rng, shape, low=1.0, high=2.0):
    """Draw values uniformly from ``(-high, -low) ∪ (low, high)``."""

    if low <= 0 or high <= 0:
        raise ValueError("`low` and `high` must be positive to define |value| bounds.")
    if low >= high:
        raise ValueError("`low` must be strictly less than `high`.")

    draws = rng.uniform(-high, high, size=shape)
    mask = np.abs(draws) < low
    while np.any(mask):
        draws[mask] = rng.uniform(-high, high, size=mask.sum())
        mask = np.abs(draws) < low
    return draws


def _rescale_var_mats(mats, target=0.99):
    """Rescale VAR coefficient matrices so the spectral radius reaches ``target``."""
    if target <= 0 or target >= 1.0:
        raise ValueError("spectral radius target must be in (0, 1).")

    if not mats:
        return mats, 0.0, 0.0

    p = mats[0].shape[0]
    L = len(mats)

    def _companion(matrices):
        companion = np.zeros((p * L, p * L))
        companion[:p, :p * L] = np.hstack(matrices)
        if L > 1:
            companion[p:, :-p] = np.eye(p * (L - 1))
        return companion

    companion = _companion(mats)
    rho_before = _spectral_radius(companion)

    if rho_before == 0.0:
        return mats, rho_before, rho_before

    scale = target / max(rho_before, np.finfo(float).eps)
    mats = [A * scale for A in mats]

    companion_scaled = _companion(mats)
    rho_after = _spectral_radius(companion_scaled)

    if rho_after > target:
        correction = target / max(rho_after, np.finfo(float).eps)
        mats = [A * correction for A in mats]
        companion_scaled = _companion(mats)
        rho_after = _spectral_radius(companion_scaled)

    return mats, rho_before, rho_after

def simulate_latent_VAR2(T, q, rng=None, burn_in=300):
    if rng is None:
        rng = np.random.RandomState(123)
    Phi1 = rng.uniform(-0.6, 0.6, size=(q, q))
    Phi2 = rng.uniform(-0.6, 0.6, size=(q, q))
    Phi1, Phi2 = _make_stable_var2(Phi1, Phi2, target=0.9)
    Sigma = 0.5 * np.eye(q)
    TT = T + burn_in
    F = np.zeros((TT, q))
    eps = rng.multivariate_normal(np.zeros(q), Sigma, size=TT)
    for t in range(2, TT):
        F[t] = Phi1 @ F[t-1] + Phi2 @ F[t-2] + eps[t]
    return F[burn_in:], {"Phi1": Phi1, "Phi2": Phi2, "Sigma": Sigma}

# --------- link functions: identity vs RBF ----------
def id_features(F):
    return F

def rbf_features(F, rng, n_centers, output_std_match=False):
    """Return RBF features with ``n_centers`` radial basis functions."""
    # Higher intensity => more RBF centers => richer nonlinear behavior.
    # centers ~ N(0,1), median distance gamma heuristic
    K, q = n_centers, F.shape[1]
    C = rng.normal(0, 1, size=(K, q))
    if K > 1:
        d2 = ((C[:, None, :] - C[None, :, :])**2).sum(-1)
        med = np.median(d2[np.triu_indices(K, 1)])
        gamma = 1.0 / max(med, 1e-6)
    else:
        gamma = 0.5
    d2 = ((F[:, None, :] - C[None, :, :])**2).sum(-1)
    Phi = np.exp(-gamma * d2)
    std = Phi.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Phi = Phi / std
    if output_std_match:
        target_std = F.std(axis=0, keepdims=True)
        target_std[target_std == 0] = 1.0
        if Phi.shape[1] != target_std.shape[1]:
            raise ValueError(
                "RBF output_std_match requires out_dim to equal the latent dimension."
            )
        Phi = Phi * target_std
    return Phi

def make_link(kind, q, out_dim, rng, *, output_std_match=False, intensity=None):
    kind = kind.lower()
    if kind in ("linear", "identity", "id"):
        return lambda F: id_features(F)
    if kind == "rbf":
        if intensity is not None:
            n_centers = max(1, int(intensity))
        else:
            n_centers = out_dim
        return lambda F: rbf_features(
            F,
            rng,
            n_centers,
            output_std_match=output_std_match,
        )
    raise ValueError(f"Unsupported nonlinearity type: {kind}")

# --------- noise helpers ----------
def _student_t_noise(rng, size, Sigma, df):
    if df <= 2:
        raise ValueError("Student-t noise requires df > 2 for finite variance.")

    dim = Sigma.shape[0]
    base = rng.multivariate_normal(np.zeros(dim), Sigma, size=size)
    chi = rng.chisquare(df, size=size)
    scale = np.sqrt(chi / df)
    scale = np.maximum(scale, np.sqrt(np.finfo(float).tiny))
    adjust = np.sqrt((df - 2) / df)
    return (base / scale[..., None]) * adjust

# --------- simulate HF (monthly) and LF (quarterly) blocks ----------
def _almon_polynomial_weights(num_lags: int, rng, degree: int = 2):
    """Return Almon polynomial weights that decay with the lag index."""
    if num_lags <= 0:
        return np.zeros(0, dtype=float)

    idx = np.arange(num_lags, dtype=float)
    X = np.vstack([idx ** d for d in range(degree + 1)]).T
    coeffs = rng.normal(0.0, 0.5, size=(degree + 1,))
    # ensure decay by keeping intercept positive and higher-order terms non-positive
    coeffs[0] = np.abs(coeffs[0]) + 0.5
    coeffs[1:] = -np.abs(coeffs[1:])
    weights = X @ coeffs
    weights = np.maximum(weights, 1e-6)
    weights /= weights.sum()
    return weights


def make_almon_lag_matrices(num_lags: int, out_dim: int, factor_dim: int, rng, degree: int = 2):
    if num_lags <= 0:
        return np.zeros((0, out_dim, factor_dim))

    weights = _almon_polynomial_weights(num_lags, rng, degree=degree)
    base_loadings = rng.uniform(0.1, 0.6, size=(out_dim, factor_dim))
    Lambda = np.zeros((num_lags, out_dim, factor_dim))
    for lag in range(num_lags):
        signs = rng.choice([-1.0, 1.0], size=(out_dim, factor_dim))
        Lambda[lag] = base_loadings * weights[lag] * signs
    return Lambda


def simulate_hf_block(
    F,
    p_x,
    Lx,
    link,
    rng,
    q_fx,
    noise_kind="student_t",
    cov_scale=1.0,
    noise_rescale=1.0,
    student_df=8,
    gF=None,
    spectral_target=0.99,
):
    T, _ = F.shape
    if gF is None:
        gF = link(F)                               # [T, d_g]
    elif gF.shape[0] != T:
        raise ValueError("Length mismatch between provided factors and F in HF block.")
    d_g = gF.shape[1]
    A = [_draw_signed_uniform(rng, (p_x, p_x), low=1.0, high=2.0) for _ in range(Lx)]
    A, rho_before, rho_after = _rescale_var_mats(A, target=spectral_target) if A else (A, 0.0, 0.0)
    if A:
        print(
            f"[simulate_hf_block] spectral radius {rho_before:.4f} -> {rho_after:.4f} "
            f"(target={spectral_target:.4f})"
        )
    Lambda_fx = make_almon_lag_matrices(q_fx + 1, p_x, d_g, rng)
    Sigma = cov_scale * np.eye(p_x)

    X = np.zeros((T, p_x))
    noise_kind = noise_kind.lower()
    if noise_kind in {"student_t", "student-t", "studentt", "t"}:
        eps = _student_t_noise(rng, size=T, Sigma=Sigma, df=student_df)
    elif noise_kind in {"gaussian", "normal"}:
        eps = rng.multivariate_normal(np.zeros(p_x), Sigma, size=T)
    else:
        raise ValueError(f"Unsupported noise kind for X block: {noise_kind}")
    eps *= noise_rescale
    for t in range(T):
        max_l = min(Lx, t)
        ar = sum(A[l] @ X[t - (l + 1)] for l in range(max_l))

        factor_terms = sum(
            Lambda_fx[lag] @ gF[t - lag]
            for lag in range(q_fx + 1)
            if t - lag >= 0
        )
        X[t] = ar + factor_terms + eps[t]
    return X

def simulate_lf_block(
    F,
    r,
    p_y,
    Ly,
    link,
    rng,
    q_fy,
    cov_scale=1.0,
    noise_rescale=1.0,
    gF=None,
    spectral_target=0.99,
):
    T, _ = F.shape
    idx_q = np.arange(r - 1, T, r)  # e.g., r=3 -> 2,5,8,... align LF at every r-th HF step
    if gF is None:
        gF = link(F)
    elif gF.shape[0] != T:
        raise ValueError("Length mismatch between provided factors and F in LF block.")
    d_g = gF.shape[1]
    C = [_draw_signed_uniform(rng, (p_y, p_y), low=1.0, high=2.0) for _ in range(Ly)]
    C, rho_before, rho_after = _rescale_var_mats(C, target=spectral_target) if C else (C, 0.0, 0.0)
    if C:
        print(
            f"[simulate_lf_block] spectral radius {rho_before:.4f} -> {rho_after:.4f} "
            f"(target={spectral_target:.4f})"
        )
    Lambda_fy = make_almon_lag_matrices(q_fy + 1, p_y, d_g, rng)
    Sigma = cov_scale * np.eye(p_y)

    Y = np.zeros((len(idx_q), p_y))
    eps = rng.multivariate_normal(np.zeros(p_y), Sigma, size=len(idx_q))
    eps *= noise_rescale
    for i, t in enumerate(idx_q):
        # Convert quarterly index to the corresponding monthly step (zero-based)
        expected_idx = (i + 1) * r - 1
        if expected_idx != t:
            raise ValueError("Low-frequency index misaligned with aggregation ratio.")
        quarter_month_idx = expected_idx

        # AR(Ly): use lags Y[i-1], Y[i-2], ..., Y[i-Ly]
        max_l = min(Ly, i)
        ar = sum(C[l - 1] @ Y[i - l] for l in range(1, max_l + 1))
        factor_terms = sum(
            Lambda_fy[lag] @ gF[quarter_month_idx - lag]
            for lag in range(q_fy + 1)
            if quarter_month_idx - lag >= 0
        )
        Y[i] = ar + factor_terms + eps[i]
    return idx_q, Y

# --------- long format identical to fred md converter ----------

def build_int_indices(T_M, idx_q):
    """Return integer monthly and quarterly indices (no dates)."""
    months = np.arange(T_M, dtype=int)    # 0,1,2,...,T_M-1
    quarters = idx_q.astype(int)          # e.g., 2,5,8,... for r=3
    return months, quarters

def to_long(df_wide: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    out = (
        df_wide
        .stack()
        .reset_index()
        .rename(columns={"level_0": "Timestamp", "level_1": "Variable", 0: "Value"})
        .assign(Frequency=freq_label)
    )
    return out

if __name__ == "__main__":
    # --- Read config and match your existing I/O contract ---
    cfg_path = project_root / "src" / "config" / "cfg.yaml"
    config = Config(cfg_path)

    # knobs (choose defaults that mirror your real pipeline)
    simulate         = getattr(config.simulation, "simulate", True)
    seed         = getattr(config.simulation, "seed", 123)
    T_months     = getattr(config.simulation, "T_months", 360)    # e.g., 30 years
    r            = getattr(config.simulation, "ratio", 3)         # 3 months per quarter
    q            = getattr(config.simulation, "latent_dim", 3)
    nonlinear_cfg = getattr(config.simulation, "nonlinearity", "identity")
    nonlinear_intensity = None
    if isinstance(nonlinear_cfg, str):
        nonlinear_type = nonlinear_cfg
        nonlinear_out_dim = getattr(config.simulation, "nonlinearity_out_dim", q)
        nonlinear_std_match = getattr(config.simulation, "nonlinearity_std_match", False)
        nonlinear_intensity = getattr(
            config.simulation, "nonlinearity_intensity", None
        )
    elif isinstance(nonlinear_cfg, dict):
        nonlinear_type = nonlinear_cfg.get("type", "identity")
        nonlinear_out_dim = nonlinear_cfg.get(
            "out_dim", getattr(config.simulation, "nonlinearity_out_dim", q)
        )
        nonlinear_std_match = nonlinear_cfg.get(
            "output_std_match", getattr(config.simulation, "nonlinearity_std_match", False)
        )
        nonlinear_intensity = nonlinear_cfg.get(
            "intensity", getattr(config.simulation, "nonlinearity_intensity", None)
        )
    else:
        nonlinear_type = getattr(nonlinear_cfg, "type", "identity")
        nonlinear_out_dim = getattr(
            nonlinear_cfg,
            "out_dim",
            getattr(config.simulation, "nonlinearity_out_dim", q),
        )
        nonlinear_std_match = getattr(
            nonlinear_cfg,
            "output_std_match",
            getattr(config.simulation, "nonlinearity_std_match", False),
        )
        nonlinear_intensity = getattr(
            nonlinear_cfg,
            "intensity",
            getattr(config.simulation, "nonlinearity_intensity", None),
        )

    if isinstance(nonlinear_type, str) and nonlinear_type.lower() == "rbf":
        if nonlinear_intensity is not None:
            try:
                nonlinear_intensity = int(nonlinear_intensity)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "simulation.nonlinearity_intensity must be an integer"
                ) from exc
            if nonlinear_intensity < 1:
                raise ValueError(
                    "simulation.nonlinearity_intensity must be >= 1 for RBF features"
                )
            nonlinear_out_dim = nonlinear_intensity
    Lx           = getattr(config.simulation, "Lx", 1)
    Ly           = getattr(config.simulation, "Ly", 1)

    # ----- variable names come from counts only: X1..Xp_x, Y1..Yp_y -----
    # Use p_x / p_y from the simulation section (recommended). If not present,
    # fall back to old num_monthly/num_quarterly for backward-compat.
    p_x = getattr(config.simulation, "p_x", getattr(config.simulation, "num_monthly", 8))
    p_y = getattr(config.simulation, "p_y", getattr(config.simulation, "num_quarterly", 1))
    monthly_vars   = [f"X{i+1}" for i in range(p_x)]
    quarterly_vars = [f"Y{j+1}" for j in range(p_y)]

    # --- Simulate on the monthly clock ---
    rng = np.random.RandomState(seed)
    factor_lags = getattr(config.simulation, "factor_lags", None)
    if factor_lags is None:
        q_fx = q_fy = 0
    else:
        q_fx = getattr(factor_lags, "q_fx", 0)
        q_fy = getattr(factor_lags, "q_fy", 0)

    default_burn_in = max(50, q_fx + q_fy + 5)
    burn_in = getattr(config.simulation, "burn_in", default_burn_in)

    link = make_link(
        nonlinear_type,
        q=q,
        out_dim=nonlinear_out_dim,
        rng=rng,
        output_std_match=nonlinear_std_match,
        intensity=nonlinear_intensity,
    )

    F_full, _ = simulate_latent_VAR2(T_months, q, rng=rng, burn_in=burn_in)
    gF_full = link(F_full)

    noise_x_distribution = getattr(config.simulation, "noise_x_distribution", "student_t")
    cov_scale_x = getattr(config.simulation, "cov_scale_x", 1.0)
    cov_scale_y = getattr(config.simulation, "cov_scale_y", 1.0)
    noise_rescale_x = getattr(config.simulation, "noise_rescale_x", 1.0)
    noise_rescale_y = getattr(config.simulation, "noise_rescale_y", 1.0)
    spectral_target_default = getattr(config.simulation, "spectral_target", 0.99)
    spectral_target_x = getattr(config.simulation, "spectral_target_x", spectral_target_default)
    spectral_target_y = getattr(config.simulation, "spectral_target_y", spectral_target_default)

    X_full = simulate_hf_block(
        F_full,
        p_x=p_x,
        Lx=Lx,
        link=link,
        rng=rng,
        q_fx=q_fx,
        noise_kind=noise_x_distribution,
        cov_scale=cov_scale_x,
        noise_rescale=noise_rescale_x,
        gF=gF_full,
        spectral_target=spectral_target_x,
    )         # [T_total, p_x]
    idx_q_full, Y_full = simulate_lf_block(
        F_full,
        r=r,
        p_y=p_y,
        Ly=Ly,
        link=link,
        rng=rng,
        q_fy=q_fy,
        cov_scale=cov_scale_y,
        noise_rescale=noise_rescale_y,
        gF=gF_full,
        spectral_target=spectral_target_y,
    )  # [T_Q_full, p_y]

    # Discard burn-in observations across all simulated series
    X = X_full
    idx_q = idx_q_full
    Y = Y_full

    # --- Build integer indices and assemble long format ---
    idx_M, idx_Q = build_int_indices(T_M=X.shape[0], idx_q=idx_q)
    dfM = pd.DataFrame(X, index=idx_M, columns=monthly_vars)
    dfQ = pd.DataFrame(Y, index=idx_Q, columns=quarterly_vars)

    md_long = to_long(dfM, freq_label="M")
    qd_long = to_long(dfQ, freq_label="Q")
    long_df = pd.concat([md_long, qd_long], ignore_index=True)

    # Match your converter’s sort behavior:
    #   - sort by Timestamp
    #   - within date: monthly first, then quarterly
    #   - within freq: preserve (monthly_vars + quarterly_vars) order
    freq_order = {"M": 0, "Q": 1}
    long_df["FreqSort"] = long_df["Frequency"].map(freq_order)
    var_order = {v: i for i, v in enumerate(monthly_vars + quarterly_vars)}
    long_df["VarOrder"] = long_df["Variable"].map(var_order)
    long_df = (
        long_df
        .sort_values(["Timestamp", "FreqSort", "VarOrder"])
        .drop(columns=["FreqSort", "VarOrder"])
        .reset_index(drop=True)
    )

    # File naming convention identical to your converter
    suffix = f"{len(monthly_vars)}M_{len(quarterly_vars)}Q"
    output_path = project_root / config.paths.data_processed_template_simulation.format(suffix=suffix)
    long_df.to_csv(output_path, index=False)
    print(long_df.head())
    print(f"\nSaved simulated long-format data to: {output_path.resolve()}")