import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ------------------------
# Setup paths and configs
# ------------------------
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

DATA_PATH = project_root / "data" / "processed" / "long_format_fred.csv"
OUTPUT_PATH = project_root / "outputs"
TARGET_VAR = "INDPRO"
TRAIN_SPLIT = 0.8
MAX_LAG = 40  # for BIC-based lag selection

# ------------------------
# Load and preprocess data
# ------------------------
def load_target_series(df, target_var):
    """Extracts the target series sorted by time."""
    target_df = df[df["Variable"] == target_var].sort_values("Timestamp")
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

    OUTPUT_PATH.mkdir(exist_ok=True)
    pd.DataFrame({
        "date": test_dates,
        "target": y_test.values,
        "predicted": ar_preds.values
    }).to_csv(OUTPUT_PATH / "ar_preds.csv", index=False)

    # Plot predictions vs. true values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="True", marker='o')
    plt.plot(ar_preds.values, label="Predicted", marker='x')
    plt.legend()
    plt.title(f"AR({selected_lag}) Forecasts vs True Targets (Out-of-Sample)")
    plt.xlabel("Sample")
    plt.ylabel("Target Value")
    plt.tight_layout()
    plt.show()

    # Plot ACF and PACF of training series
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(y_train, lags=MAX_LAG, ax=axes[0])
    axes[0].set_title("ACF of Training Target")
    plot_pacf(y_train, lags=MAX_LAG, ax=axes[1])
    axes[1].set_title("PACF of Training Target")
    plt.tight_layout()
    plt.show()
