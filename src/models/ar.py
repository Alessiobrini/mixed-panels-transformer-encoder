import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.utils.config import Config

# ------------------------
# Setup project root & load config
# ------------------------
cfg_path = project_root / "src" / "config" / "cfg.yaml"
config = Config(cfg_path)

raw_md_path   = project_root / config.paths.data_raw_fred_monthly
md_cols       = pd.read_csv(raw_md_path, nrows=0).columns.tolist()
if config.features.all_monthly:
    monthly_vars = [c for c in md_cols if c != 'date']
    target_var   = config.features.target
    quarterly_vars = [target_var]  # always include target as quarterly
    # if target_var in monthly_vars:
    #     monthly_vars.remove(target_var)
else:
    monthly_vars   = config.features.monthly_vars
    quarterly_vars = config.features.quarterly_vars
suffix = f"{len(monthly_vars)}M_{len(quarterly_vars)}Q"

# Paths driven by suffix
DATA_PATH   = project_root / config.paths.data_processed_template.format(suffix=suffix)
OUTPUT_FILE = project_root / config.paths.outputs.ar_preds.format(suffix=suffix)

TARGET_VAR  = config.features.target
TRAIN_SPLIT = config.data.train_ratio
MAX_LAG     = config.model.ar.max_lag

# ------------------------
# Load and preprocess data
# ------------------------
def load_target_series(df, target_var):
    """Extracts the quarterly target series sorted by time."""
    target_df = df[
        (df["Variable"] == target_var) &
        (df["Frequency"] == "Q")
    ].sort_values("Timestamp")
    return target_df[["Timestamp", "Value"]].reset_index(drop=True)


# ------------------------
# Fit AR model with optimal lag (based on BIC)
# ------------------------
def fit_ar_with_optimal_lag(y_train, y_test, max_lag):
    selection = ar_select_order(y_train, maxlag=max_lag, ic="bic", old_names=False)
    selected_lag = selection.ar_lags[-1]
    print(f"Selected lag order (BIC): {selected_lag}")

    model = AutoReg(y_train, lags=selected_lag).fit()
    preds = model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
    return preds, selected_lag, selection, model

# ------------------------
# Main script
# ------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, parse_dates=["Timestamp"])
    target_df = load_target_series(df, TARGET_VAR)

    n = len(target_df)
    split_idx = int(n * TRAIN_SPLIT)
    y_train = target_df["Value"][:split_idx]
    y_test = target_df["Value"][split_idx:]
    test_dates = target_df["Timestamp"][split_idx:]

    ar_preds, selected_lag, selection, model = fit_ar_with_optimal_lag(y_train, y_test, MAX_LAG)

    # Write predictions CSV
    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({
        "date": test_dates,
        "target": y_test.values,
        "predicted": ar_preds.values
    }).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved AR forecasts to: {OUTPUT_FILE.resolve()}")

    # # Plot predictions vs. true values
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_test.values, label="True", marker='o')
    # plt.plot(ar_preds.values, label="Predicted", marker='x')
    # plt.legend()
    # plt.title(f"AR({selected_lag}) Forecasts vs True Targets (Out-of-Sample)")
    # plt.xlabel("Sample")
    # plt.ylabel("Target Value")
    # plt.tight_layout()
    # plt.show()

    # # Plot ACF and PACF of training series
    # fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    # plot_acf(y_train, lags=MAX_LAG, ax=axes[0])
    # axes[0].set_title("ACF of Training Target")
    # plot_pacf(y_train, lags=MAX_LAG, ax=axes[1])
    # axes[1].set_title("PACF of Training Target")
    # plt.tight_layout()
    # plt.show()
