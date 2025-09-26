import sys
import numpy as np
import pandas as pd
from pathlib import Path

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
    B = rng.uniform(-0.6, 0.6, size=(p_x, d_g))
    Sigma = 0.5 * np.eye(p_x)

    X = np.zeros((T, p_x))
    eps = rng.multivariate_normal(np.zeros(p_x), Sigma, size=T)
    for t in range(max(1, Lx), T):
        ar = sum(A[ℓ] @ X[t-ℓ-1] for ℓ in range(Lx))
        X[t] = ar + B @ gF[t] + eps[t]
    return X

def simulate_lf_block(F, r, p_y, Ly, link, rng):
    T, _ = F.shape
    idx_q = np.arange(r-1, T, r)               # align LF at every r-th HF step
    gF = link(F)                               
    d_g = gF.shape[1]
    C = [rng.uniform(-0.4, 0.4, size=(p_y, p_y)) for _ in range(Ly)]
    D = rng.uniform(-0.6, 0.6, size=(p_y, d_g))
    Sigma = 0.5 * np.eye(p_y)

    Y = np.zeros((len(idx_q), p_y))
    eps = rng.multivariate_normal(np.zeros(p_y), Sigma, size=len(idx_q))
    for i, t in enumerate(idx_q):
        ar = sum(C[ℓ] @ Y[i-ℓ-1] for ℓ in range(1, min(Ly, i)+1))
        Y[i] = ar + D @ gF[t] + eps[i]
    return idx_q, Y

# --------- build dates + long format identical to your converter ----------
def build_dates(T_M, r, start="2000-01-01", place_quarter_at="end"):
    # Monthly index (your converter keeps a single 'Timestamp' column downstream)
    dates_M = pd.date_range(start=start, periods=T_M, freq="MS")
    # Often you display EoM; switch if you prefer:
    dates_M = dates_M + pd.offsets.MonthEnd(0)

    # Quarterly index aligned to monthly per r
    T_Q = T_M // r
    if place_quarter_at == "start":
        dates_Q = pd.period_range(dates_M[0], periods=T_Q, freq="Q").to_timestamp(how="start")
    else:
        dates_Q = pd.period_range(dates_M[r-1], periods=T_Q, freq="Q").to_timestamp(how="end")
    return dates_M, dates_Q

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
    cfg_path = project_root / "src" / "config" / "cfg_sim.yaml"
    config = Config(cfg_path)

    # knobs (choose defaults that mirror your real pipeline)
    seed         = getattr(config.simulation, "seed", 123)
    T_months     = getattr(config.simulation, "T_months", 360)    # e.g., 30 years
    r            = getattr(config.simulation, "ratio", 3)         # 3 months per quarter
    nonlinear    = getattr(config.simulation, "nonlinearity", "identity")  # or "rbf"
    q            = getattr(config.simulation, "latent_dim", 3)
    Lx           = getattr(config.simulation, "Lx", 1)
    Ly           = getattr(config.simulation, "Ly", 1)
    start_date   = getattr(config.simulation, "start_date", "1990-01-01")
    place_q_at   = getattr(config.simulation, "place_quarter_at", "end")  # "start" | "end"

    # variable lists: we only need counts and names for output
    # Your converter builds monthly_vars/quarterly_vars from config; do the same here:
    if config.features.all_monthly:
        # In your converter, you discover columns from a CSV; here we synthesize names
        # Keep target in quarterly_vars like your current logic.
        target = config.features.target
        num_m  = getattr(config.simulation, "num_monthly", 8)
        num_q  = getattr(config.simulation, "num_quarterly", 1)
        monthly_vars   = [f"SIM_X{i+1}" for i in range(num_m)]
        quarterly_vars = [target] if target else [f"SIM_Y1"]
    else:
        monthly_vars   = list(config.features.monthly_vars)
        quarterly_vars = list(config.features.quarterly_vars)
        # If user wants to keep target as quarterly as in converter, ensure it's present:
        if config.features.target and config.features.target not in quarterly_vars:
            quarterly_vars = [config.features.target] + quarterly_vars

    p_x = len(monthly_vars)
    p_y = len(quarterly_vars)

    # --- Simulate on the monthly clock ---
    rng = np.random.RandomState(seed)
    F, _ = simulate_latent_VAR2(T_months, q, seed=seed, burn_in=300)
    link = make_link(nonlinear, q=q, out_dim=q, rng=rng)
    X = simulate_hf_block(F, p_x=p_x, Lx=Lx, link=link, rng=rng)         # [T_M, p_x]
    idx_q, Y = simulate_lf_block(F, r=r, p_y=p_y, Ly=Ly, link=link, rng=rng)  # [T_Q, p_y]

    # --- Build dates and assemble long format identical to your converter ---
    dates_M, dates_Q = build_dates(T_M=X.shape[0], r=r, start=start_date, place_quarter_at=place_q_at)
    dfM = pd.DataFrame(X, index=dates_M, columns=monthly_vars)
    dfQ = pd.DataFrame(Y, index=dates_Q, columns=quarterly_vars)

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
    output_path = project_root / config.paths.data_processed_template.format(suffix=suffix)
    long_df.to_csv(output_path, index=False)
    print(long_df.head())
    print(f"\nSaved simulated long-format data to: {output_path.resolve()}")