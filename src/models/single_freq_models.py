import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import yaml, pdb

# --- Load config ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
with open(project_root / "src" / "config" / "cfg.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Parameters ---
EXPERIMENT_DATE = "2025-07-24"
N_LAGS = 4
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3

# --- Paths ---
# Use project_root to resolve relative paths from config
quarterly_path = project_root / config["paths"]["data_raw_fred_quarterly"]
monthly_path = project_root / config["paths"]["data_raw_fred_monthly"]
experiment_dir = project_root / "outputs" / "experiments"
quarterly_vars = config["features"]["quarterly_vars"]
monthly_vars = config["features"]["monthly_vars"]
train_ratio = config["data"]["train_ratio"]

# --- Load and prepare data ---
if not quarterly_path.exists():
    raise FileNotFoundError(f"Quarterly data not found at {quarterly_path}")
qd = pd.read_csv(quarterly_path, parse_dates=["date"]).sort_values("date")

if not monthly_path.exists():
    raise FileNotFoundError(f"Monthly data not found at {monthly_path}")
md = pd.read_csv(monthly_path, parse_dates=["date"]).sort_values("date")

# Resample monthly data to last value of each quarter
# 1) Resample at quarter‐end exactly as before
md_q = (
    md.set_index("date")
      .resample("QE")   # quarter‐end frequency
      .last()
)
# 2) Convert those quarter‐end timestamps to the quarter‐start (e.g., 1959-03-31 → 1959-03-01)
md_q.index = md_q.index.to_period("M").to_timestamp()

# Merge quarterly and resampled monthly data
# Ensures only dates present in both
# Identify overlapping columns (excluding the key “date”)
overlap = set(qd.columns) & set(md_q.columns) - {'date'}
# Drop them from md_q so qd’s UNRATE (etc.) remains
md_q = md_q.drop(columns=list(overlap))
data = pd.merge(qd, md_q, on="date", how="inner").set_index("date")

# --- Loop through each target ---
for target in quarterly_vars[:1]:
    print(f"\n--- Processing target: {target} ---")
    target_dir = experiment_dir / f"{target}_{EXPERIMENT_DATE}"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    sample = next(target_dir.glob("*_preds_*.csv"), None)
    if sample is not None:
        # split on underscore, take the last piece before “.csv”
        suffix = sample.stem.split("preds_")[1]
    else:
        # fallback if folder is empty
        suffix = EXPERIMENT_DATE

    # Define predictors: other quarterly vars + monthly vars
    other_qd_vars = [v for v in quarterly_vars if v != target]
    predictors = other_qd_vars + monthly_vars
    
    # Add lags of the target
    # Create lag features for all vars (quarterly + monthly + target)
    # Build all lagged columns in a separate DataFrame, then concat once
    lagged_dict = {}
    vars_to_lag = [target] + other_qd_vars + monthly_vars
    for var in vars_to_lag:
        for lag in range(1, N_LAGS + 1):
            col = f"{var}_lag{lag}"
            lagged_dict[col] = data[var].shift(lag)
    
    lagged_df = pd.DataFrame(lagged_dict, index=data.index)
    data = pd.concat([data, lagged_df], axis=1)
    
    # Now collect the feature names
    lagged_feats = list(lagged_dict.keys())
    
        
    df_model = (
        data[lagged_feats + [target]]
        .dropna()
        .copy()
    )


    # Train-test split (temporal)
    n_total = len(df_model)
    n_train = int(n_total * train_ratio)
    df_train = df_model.iloc[:n_train]
    df_test = df_model.iloc[n_train:]

    # Prepare arrays
    feature_cols = df_train.columns.drop(target)
    
    X_train = df_train[feature_cols].values
    y_train = df_train[target].values
    
    X_test  = df_test[feature_cols].values
    y_test  = df_test[target].values
    test_dates = df_test.index

    # Standardize predictors
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. OLS
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    ols_preds = ols.predict(X_test_scaled)
    pd.DataFrame({"date": test_dates, "target": y_test, "predicted": ols_preds})\
      .to_csv(target_dir / f"ols_preds_{suffix}.csv", index=False)

    # 2. XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    pd.DataFrame({"date": test_dates, "target": y_test, "predicted": xgb_preds})\
      .to_csv(target_dir / f"xgb_preds_{suffix}.csv", index=False)

    # 3. Feedforward NN (PyTorch)
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_ds = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        nn_preds = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device))\
            .cpu().numpy().flatten()
    pd.DataFrame({"date": test_dates, "target": y_test, "predicted": nn_preds})\
      .to_csv(target_dir / f"nn_preds_{suffix}.csv", index=False)

print("\n All models finished and predictions saved.")
