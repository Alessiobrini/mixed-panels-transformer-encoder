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

def _make_stable_ar(mats, target=0.9):
    if not mats:
        return mats

    p = mats[0].shape[0]
    L = len(mats)

    companion = np.zeros((p * L, p * L))
    companion[:p, :] = np.hstack(mats)
    if L > 1:
        companion[p:, :-p] = np.eye(p * (L - 1))

    rho = _spectral_radius(companion)
    if rho >= target and rho > 0:
        scale = target / rho
        mats = [A * scale for A in mats]
    return mats

def simulate_latent_VAR2(T, q, seed=123, burn_in=300):
    rng = np.random.RandomState(seed)
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

def rbf_features(F, rng, out_dim):
    # centers ~ N(0,1), median distance gamma heuristic
    K, q = out_dim, F.shape[1]
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
    return Phi / std

def make_link(kind, q, out_dim, rng):
    if kind.lower() in ("linear", "identity", "id"):
        return lambda F: id_features(F)
    return lambda F: rbf_features(F, rng, out_dim)

# --------- simulate HF (monthly) and LF (quarterly) blocks ----------
def simulate_hf_block(F, p_x, Lx, link, rng):
    T, _ = F.shape
    gF = link(F)                               # [T, d_g]
    d_g = gF.shape[1]
    A = [rng.uniform(-0.3, 0.3, size=(p_x, p_x)) for _ in range(Lx)]
    A = _make_stable_ar(A, target=0.9)
    B = rng.uniform(-0.6, 0.6, size=(p_x, d_g))
    Sigma = 0.5 * np.eye(p_x)

    X = np.zeros((T, p_x))
    eps = rng.multivariate_normal(np.zeros(p_x), Sigma, size=T)
    for t in range(Lx, T):
        ar = sum(A[l] @ X[t - (l + 1)] for l in range(Lx))
        X[t] = ar + B @ gF[t] + eps[t]
    return X

def simulate_lf_block(F, r, p_y, Ly, link, rng):
    T, _ = F.shape
    idx_q = np.arange(r-1, T, r)  # e.g., r=3 -> 2,5,8,... align LF at every r-th HF step
    gF = link(F)                               
    d_g = gF.shape[1]
    C = [rng.uniform(-0.4, 0.4, size=(p_y, p_y)) for _ in range(Ly)]
    C = _make_stable_ar(C, target=0.9)
    D = rng.uniform(-0.6, 0.6, size=(p_y, d_g))
    Sigma = 0.5 * np.eye(p_y)

    Y = np.zeros((len(idx_q), p_y))
    eps = rng.multivariate_normal(np.zeros(p_y), Sigma, size=len(idx_q))
    for i, t in enumerate(idx_q):
        # AR(Ly): use lags Y[i-1], Y[i-2], ..., Y[i-Ly]
        max_l = min(Ly, i)
        ar = sum(C[l - 1] @ Y[i - l] for l in range(1, max_l + 1))
        Y[i] = ar + D @ gF[t] + eps[i]
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
    nonlinear    = getattr(config.simulation, "nonlinearity", "identity")  # or "rbf"
    q            = getattr(config.simulation, "latent_dim", 3)
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
    F, _ = simulate_latent_VAR2(T_months, q, seed=seed, burn_in=300)
    link = make_link(nonlinear, q=q, out_dim=q, rng=rng)
    X = simulate_hf_block(F, p_x=p_x, Lx=Lx, link=link, rng=rng)         # [T_M, p_x]
    idx_q, Y = simulate_lf_block(F, r=r, p_y=p_y, Ly=Ly, link=link, rng=rng)  # [T_Q, p_y]

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