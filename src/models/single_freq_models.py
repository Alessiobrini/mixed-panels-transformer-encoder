import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import yaml
import random

SEED = 42

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


# ---------------------------------------------------------------------------
# NN architecture (unchanged)
# ---------------------------------------------------------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, h1=64, h2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Callable entry point
# ---------------------------------------------------------------------------
def run_single_freq_baselines(
    target: str,
    data: pd.DataFrame,
    train_ratio: float,
    exp_path: Path,
    suffix: str,
    n_lags: int = 4,
    optimize: bool = True,
    val_ratio: float = 0.1,
    seed: int = SEED,
):
    """Run OLS, XGBoost, and NN baselines on lagged features.

    Args:
        target: column name of the target variable in *data*.
        data: wide-format DataFrame (DatetimeIndex + value columns).
              Must include the target column and predictor columns.
        train_ratio: fraction of rows for training.
        exp_path: directory to write prediction CSVs.
        suffix: file suffix for output names (e.g., ``7D_43M_14Q``).
        n_lags: number of lags to create for each variable.
        optimize: whether to run validation-based hyperparameter search.
        val_ratio: fraction of training set to hold out for validation.
        seed: random seed.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    exp_path = Path(exp_path)
    exp_path.mkdir(parents=True, exist_ok=True)

    # --- Build lagged features ---
    all_vars = [c for c in data.columns]
    lagged_dict = {}
    for var in all_vars:
        for lag in range(1, n_lags + 1):
            lagged_dict[f"{var}_lag{lag}"] = data[var].shift(lag)

    lagged_df = pd.DataFrame(lagged_dict, index=data.index)
    model_data = pd.concat([data[[target]], lagged_df], axis=1).dropna()

    lagged_feats = list(lagged_dict.keys())

    # --- Train / (val) / test split ---
    n_total = len(model_data)
    n_train = int(n_total * train_ratio)
    df_train = model_data.iloc[:n_train]
    df_test = model_data.iloc[n_train:]

    if optimize:
        n_val = max(1, int(len(df_train) * val_ratio))
        df_val = df_train.iloc[-n_val:]
        df_train = df_train.iloc[:-n_val]

    feature_cols = [c for c in model_data.columns if c != target]
    X_train = df_train[feature_cols].values
    y_train = df_train[target].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target].values
    test_dates = df_test.index

    if optimize:
        X_val = df_val[feature_cols].values
        y_val = df_val[target].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if optimize:
        X_val_scaled = scaler.transform(X_val)

    # --- 1. OLS ---
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    ols_preds = ols.predict(X_test_scaled)
    pd.DataFrame({"date": test_dates, "target": y_test, "predicted": ols_preds}).to_csv(
        exp_path / f"ols_preds_{suffix}.csv", index=False
    )

    # --- 2. XGBoost ---
    if optimize:
        xgb_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }
        best_rmse, best_params = float("inf"), None
        for params in ParameterGrid(xgb_grid):
            tmp = XGBRegressor(**params, random_state=seed)
            tmp.fit(X_train, y_train)
            rmse = np.sqrt(mean_squared_error(y_val, tmp.predict(X_val)))
            if rmse < best_rmse:
                best_rmse, best_params = rmse, params

        xgb_final = XGBRegressor(**best_params, random_state=seed)
        xgb_final.fit(
            np.vstack([X_train, X_val]),
            np.concatenate([y_train, y_val]),
        )
        xgb_preds = xgb_final.predict(X_test)
    else:
        xgb = XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed
        )
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)

    pd.DataFrame({"date": test_dates, "target": y_test, "predicted": xgb_preds}).to_csv(
        exp_path / f"xgb_preds_{suffix}.csv", index=False
    )

    # --- 3. Feedforward NN ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_scaled.shape[1]

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    crit = nn.MSELoss()

    if optimize:
        nn_grid = [
            {"h1": 64, "h2": 32, "lr": 1e-3},
            {"h1": 128, "h2": 64, "lr": 1e-4},
        ]
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

        best_loss, best_model = float("inf"), None
        for cfg in nn_grid:
            net = SimpleNN(input_dim, cfg["h1"], cfg["h2"]).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
            for _ in range(100):
                net.train()
                loss = crit(net(X_train_t), y_train_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = crit(net(X_val_t), y_val_t).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = net
        final_model = best_model
    else:
        final_model = SimpleNN(input_dim).to(device)
        opt = torch.optim.Adam(final_model.parameters(), lr=1e-4)
        for _ in range(100):
            final_model.train()
            loss = crit(final_model(X_train_t), y_train_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

    final_model.eval()
    with torch.no_grad():
        nn_preds = final_model(X_test_t).cpu().numpy().flatten()

    pd.DataFrame({"date": test_dates, "target": y_test, "predicted": nn_preds}).to_csv(
        exp_path / f"nn_preds_{suffix}.csv", index=False
    )

    print(f"  Single-freq baselines saved to {exp_path}")


# ---------------------------------------------------------------------------
# Concatenated (cross-sectional) baselines
# ---------------------------------------------------------------------------
def _build_quarterly_wide(csv_path, target_var):
    """Pivot long-format CSV quarterly rows into wide format."""
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    q = df[df["Frequency"] == "Q"].copy()
    wide = q.pivot_table(index="Timestamp", columns="Variable", values="Value")
    wide = wide.sort_index().ffill()
    wide = wide.dropna(subset=[target_var])
    wide.index.name = "date"
    return wide


def run_concatenated_single_freq_baselines(
    ticker_csv_paths: dict,
    target_template: str,
    train_ratio: float,
    exp_path,
    suffix: str,
    n_lags: int = 2,
    optimize: bool = True,
    val_ratio: float = 0.1,
    seed: int = SEED,
):
    """Pooled OLS / XGBoost / NN across all stocks.

    Each stock's quarterly-wide data has ticker prefixes stripped so all
    share the same column names. The pooled model is trained on stacked
    data and predictions are saved per ticker.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    exp_path = Path(exp_path)

    # --- Stack all stocks ---
    generic_target = target_template.split("_", 1)[1] if "_" in target_template else target_template
    # The target template is e.g. "{TKR}_eps_yoy", generic is "eps_yoy"
    generic_target = target_template.replace("{TKR}_", "")

    all_train, all_test = [], []
    ticker_test_meta = []  # (ticker, test_dates, y_test)

    for tkr, csv_path in ticker_csv_paths.items():
        target_var = target_template.replace("{TKR}", tkr)
        try:
            wide = _build_quarterly_wide(csv_path, target_var)
        except Exception:
            continue
        # Strip ticker prefix from column names
        wide = wide.rename(columns=lambda c: c.replace(f"{tkr}_", ""))
        n_total = len(wide)
        if n_total < 4:
            continue
        n_train = int(n_total * train_ratio)
        df_train = wide.iloc[:n_train]
        df_test = wide.iloc[n_train:]
        all_train.append(df_train)
        all_test.append(df_test)
        ticker_test_meta.append({
            "ticker": tkr,
            "n_test": len(df_test),
        })

    if not all_train:
        print("  No stocks with sufficient data for single-freq baselines")
        return

    stacked_train = pd.concat(all_train, axis=0)
    stacked_test = pd.concat(all_test, axis=0)

    # --- Lag features ---
    all_vars = list(stacked_train.columns)
    def _add_lags(df):
        lagged = {}
        for var in all_vars:
            for lag in range(1, n_lags + 1):
                lagged[f"{var}_lag{lag}"] = df[var].shift(lag)
        return pd.concat([df[[generic_target]], pd.DataFrame(lagged, index=df.index)], axis=1).dropna()

    # Build lags per stock block (to avoid cross-stock contamination)
    train_blocks, test_blocks = [], []
    train_offset = 0
    test_offset = 0
    for i, meta in enumerate(ticker_test_meta):
        n_tr = len(all_train[i])
        n_te = len(all_test[i])
        # Combine train+test for this stock to compute lags correctly
        stock_df = pd.concat([all_train[i], all_test[i]], axis=0)
        stock_lagged = _add_lags(stock_df)
        # Split back — rows from train portion vs test portion
        # After dropna from lag creation, we need to re-identify train vs test
        stock_dates = stock_df.index
        train_dates = set(all_train[i].index)
        tr_mask = stock_lagged.index.isin(train_dates)
        train_blocks.append(stock_lagged[tr_mask])
        test_blocks.append(stock_lagged[~tr_mask])
        train_offset += n_tr
        test_offset += n_te

    model_train = pd.concat(train_blocks, axis=0)
    model_test_list = test_blocks  # keep separate for per-stock prediction saving

    feature_cols = [c for c in model_train.columns if c != generic_target]
    X_train = model_train[feature_cols].values
    y_train = model_train[generic_target].values

    # Validation split from training data
    if optimize:
        n_val = max(1, int(len(X_train) * val_ratio))
        X_val = X_train[-n_val:]
        y_val = y_train[-n_val:]
        X_train_fit = X_train[:-n_val]
        y_train_fit = y_train[:-n_val]
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fit)
    if optimize:
        X_val_scaled = scaler.transform(X_val)

    # --- Helper to save per-ticker predictions ---
    def _save_per_ticker(model_name, all_preds):
        offset = 0
        for i, meta in enumerate(ticker_test_meta):
            block = model_test_list[i]
            n = len(block)
            if n == 0:
                offset += 0
                continue
            ticker_preds = all_preds[offset:offset + n]
            ticker_exp = exp_path / meta["ticker"]
            ticker_exp.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "date": block.index,
                "target": block[generic_target].values,
                "predicted": ticker_preds,
            }).to_csv(ticker_exp / f"{model_name}_preds_{suffix}.csv", index=False)
            offset += n

    # Combine all test blocks for model prediction
    model_test = pd.concat(model_test_list, axis=0)
    X_test = model_test[feature_cols].values
    y_test = model_test[generic_target].values
    X_test_scaled = scaler.transform(X_test)

    # --- 1. OLS ---
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train_fit)
    ols_preds = ols.predict(X_test_scaled)
    _save_per_ticker("ols", ols_preds)

    # --- 2. XGBoost ---
    if optimize:
        xgb_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }
        best_rmse, best_params = float("inf"), None
        for params in ParameterGrid(xgb_grid):
            tmp = XGBRegressor(**params, random_state=seed)
            tmp.fit(X_train_fit, y_train_fit)
            rmse = np.sqrt(mean_squared_error(y_val, tmp.predict(X_val)))
            if rmse < best_rmse:
                best_rmse, best_params = rmse, params
        xgb_final = XGBRegressor(**best_params, random_state=seed)
        xgb_final.fit(X_train, y_train)  # retrain on full train (incl val)
    else:
        xgb_final = XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed
        )
        xgb_final.fit(X_train_fit, y_train_fit)
    xgb_preds = xgb_final.predict(X_test)
    _save_per_ticker("xgb", xgb_preds)

    # --- 3. NN ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_scaled.shape[1]
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_fit, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    crit = nn.MSELoss()

    if optimize:
        nn_grid = [
            {"h1": 64, "h2": 32, "lr": 1e-3},
            {"h1": 128, "h2": 64, "lr": 1e-4},
        ]
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        best_loss, best_model = float("inf"), None
        for cfg in nn_grid:
            net = SimpleNN(input_dim, cfg["h1"], cfg["h2"]).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
            for _ in range(100):
                net.train()
                loss = crit(net(X_train_t), y_train_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = crit(net(X_val_t), y_val_t).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = net
        final_model = best_model
    else:
        final_model = SimpleNN(input_dim).to(device)
        opt = torch.optim.Adam(final_model.parameters(), lr=1e-4)
        for _ in range(100):
            final_model.train()
            loss = crit(final_model(X_train_t), y_train_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

    final_model.eval()
    with torch.no_grad():
        nn_preds = final_model(X_test_t).cpu().numpy().flatten()
    _save_per_ticker("nn", nn_preds)

    print(f"  Concatenated single-freq baselines saved for {len(ticker_test_meta)} stocks")


# ---------------------------------------------------------------------------
# Original __main__ for FRED experiments (unchanged behavior)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open(project_root / "src" / "config" / "cfg.yaml", "r") as f:
        config = yaml.safe_load(f)

    EXPERIMENT_DATE = "2025-07-24"
    N_LAGS = 4
    OPTIMIZE = True
    VAL_RATIO = 0.1

    quarterly_path = project_root / config["paths"]["data_raw_fred_quarterly"]
    monthly_path = project_root / config["paths"]["data_raw_fred_monthly"]
    experiment_dir = project_root / "outputs" / "experiments"
    quarterly_vars = config["features"]["quarterly_vars"]
    monthly_vars = config["features"]["monthly_vars"]
    train_ratio = config["data"]["train_ratio"]

    qd = pd.read_csv(quarterly_path, parse_dates=["date"]).sort_values("date")
    md = pd.read_csv(monthly_path, parse_dates=["date"]).sort_values("date")

    md_q = md.set_index("date").resample("QE").last()
    md_q.index = md_q.index.to_period("M").to_timestamp()

    overlap = set(qd.columns) & set(md_q.columns) - {"date"}
    md_q = md_q.drop(columns=list(overlap))
    data = pd.merge(qd, md_q, on="date", how="inner").set_index("date")

    for target in quarterly_vars:
        print(f"\n--- Processing target: {target} ---")
        target_dir = experiment_dir / f"{target}_{EXPERIMENT_DATE}"
        target_dir.mkdir(parents=True, exist_ok=True)

        sample = next(target_dir.glob("*_preds_*.csv"), None)
        suffix = sample.stem.split("preds_")[1] if sample else EXPERIMENT_DATE

        run_single_freq_baselines(
            target=target,
            data=data,
            train_ratio=train_ratio,
            exp_path=target_dir,
            suffix=suffix,
            n_lags=N_LAGS,
            optimize=OPTIMIZE,
            val_ratio=VAL_RATIO,
        )

    print("\nAll models finished and predictions saved.")
