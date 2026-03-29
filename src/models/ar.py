import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ------------------------
# Load and preprocess data
# ------------------------
def load_target_series(df, target_var):
    """Extracts the quarterly target series sorted by time when appropriate."""
    target_df = df[
        (df["Variable"] == target_var) &
        (df["Frequency"] == "Q")
    ]

    # Simulated datasets may contain integer-based timestamps that are already
    # ordered. In those cases, preserve the original ordering to avoid pandas
    # attempting to sort non-datetime values as dates.
    if is_datetime64_any_dtype(target_df["Timestamp"]):
        target_df = target_df.sort_values("Timestamp")

    return target_df[["Timestamp", "Value"]].reset_index(drop=True)


# ------------------------
# Fit AR model with optimal lag (based on BIC)
# ------------------------
def fit_ar_with_optimal_lag(y_train, y_test, max_lag):
    selection = ar_select_order(y_train, maxlag=max_lag, ic="bic", old_names=False)
    if not selection.ar_lags:
        print("[INFO] No lag order selected — falling back to AR(0) model (mean predictor).")
        mean_val = y_train.mean()
        preds = pd.Series([mean_val] * len(y_test), index=y_test.index)
        return preds, 0, selection, None
    else:
        selected_lag = selection.ar_lags[-1]
        print(f"Selected lag order (BIC): {selected_lag}")
        model = AutoReg(y_train, lags=selected_lag).fit()
        preds = model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
        return preds, selected_lag, selection, model


# ------------------------
# Callable entry point
# ------------------------
def run_ar_baseline(csv_path, target_var, train_ratio, max_lag, output_path):
    """Run AR baseline on a long-format CSV and save predictions.

    This function is importable from other pipelines (e.g., equity).
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    target_df = load_target_series(df, target_var)

    n = len(target_df)
    split_idx = int(n * train_ratio)
    y_train = target_df["Value"][:split_idx]
    y_test = target_df["Value"][split_idx:]
    test_dates = target_df["Timestamp"][split_idx:]

    ar_preds, selected_lag, selection, model = fit_ar_with_optimal_lag(y_train, y_test, max_lag)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({
        "date": test_dates,
        "target": y_test.values,
        "predicted": ar_preds.values,
    }).to_csv(output_path, index=False)
    print(f"Saved AR forecasts to: {output_path.resolve()}")


# ------------------------
# Concatenated (cross-sectional) AR
# ------------------------
def run_concatenated_ar_baseline(
    ticker_csv_paths, target_template, train_ratio, max_lag, exp_path, suffix
):
    """Fit a single pooled AR on demeaned quarterly targets across all stocks.

    Each stock's series is demeaned by its training-set mean before pooling.
    At prediction time the per-stock mean is added back.
    """
    import numpy as np

    stock_data = []
    for tkr, csv_path in ticker_csv_paths.items():
        target_var = target_template.replace("{TKR}", tkr)
        df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
        ts = load_target_series(df, target_var)
        n = len(ts)
        if n < 4:
            continue
        split = int(n * train_ratio)
        train_vals = ts["Value"][:split]
        test_vals = ts["Value"][split:]
        test_dates = ts["Timestamp"][split:]
        mu = train_vals.mean()
        stock_data.append({
            "ticker": tkr,
            "train_dm": (train_vals - mu).reset_index(drop=True),
            "test_vals": test_vals.reset_index(drop=True),
            "test_dates": test_dates.reset_index(drop=True),
            "mu": mu,
            "n_test": len(test_vals),
        })

    # Pool demeaned training series
    pooled_train = pd.concat(
        [s["train_dm"] for s in stock_data], ignore_index=True
    )

    # Fit single AR
    ar_preds_raw, selected_lag, _, _ = fit_ar_with_optimal_lag(
        pooled_train, pd.Series(np.zeros(1)), max_lag
    )

    # For per-stock forecasting: use the pooled AR coefficients but
    # predict from each stock's own recent history
    from statsmodels.tsa.ar_model import AutoReg

    if selected_lag == 0:
        # Mean predictor — just use per-stock training mean
        for s in stock_data:
            tkr = s["ticker"]
            n_test = s["n_test"]
            preds = pd.Series([s["mu"]] * n_test)
            ticker_exp = Path(exp_path) / tkr
            ticker_exp.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "date": s["test_dates"],
                "target": s["test_vals"].values,
                "predicted": preds.values,
            }).to_csv(ticker_exp / f"ar_preds_{suffix}.csv", index=False)
        print(f"  Concatenated AR(0) — mean predictor for {len(stock_data)} stocks")
        return

    # Refit on pooled with selected lag to get parameters
    pooled_model = AutoReg(pooled_train, lags=selected_lag).fit()

    for s in stock_data:
        tkr = s["ticker"]
        full_dm = s["train_dm"]
        n_train = len(full_dm)
        n_test = s["n_test"]

        # Predict out-of-sample from this stock's demeaned series
        preds_dm = pooled_model.predict(start=n_train, end=n_train + n_test - 1)
        # If predict returns fewer values (edge case), pad with last known
        if len(preds_dm) < n_test:
            last = preds_dm.iloc[-1] if len(preds_dm) > 0 else 0.0
            preds_dm = pd.concat([preds_dm, pd.Series([last] * (n_test - len(preds_dm)))])
        preds = preds_dm.values[:n_test] + s["mu"]

        ticker_exp = Path(exp_path) / tkr
        ticker_exp.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "date": s["test_dates"],
            "target": s["test_vals"].values,
            "predicted": preds,
        }).to_csv(ticker_exp / f"ar_preds_{suffix}.csv", index=False)

    print(f"  Concatenated AR({selected_lag}) saved for {len(stock_data)} stocks")


# ------------------------
# Main script
# ------------------------
if __name__ == "__main__":
    from src.utils.config import Config
    from src.utils.data_paths import (
        get_output_path,
        resolve_data_paths,
        resolve_target_variable,
    )

    cfg_path = project_root / "src" / "config" / "cfg.yaml"
    config = Config(cfg_path)

    DATA_PATH, suffix, _, _ = resolve_data_paths(config, project_root)
    OUTPUT_FILE = get_output_path(config, project_root, "ar_preds", suffix)
    TARGET_VAR = resolve_target_variable(config)
    TRAIN_SPLIT = config.data.train_ratio
    MAX_LAG = config.model.ar.max_lag

    run_ar_baseline(DATA_PATH, TARGET_VAR, TRAIN_SPLIT, MAX_LAG, OUTPUT_FILE)

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
