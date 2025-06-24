import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from dieboldmariano import dm_test
import matplotlib.dates as mdates
import pdb

# ------------------------
# Config
# ------------------------
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
OUTPUT_DIR = project_root / "outputs"
FORECAST_FILES = ["ar_preds.csv", "midas_preds_3vars.csv", "transformer_preds.csv"]
TRUE_COL = "target"
PRED_COL = "predicted"

# ------------------------
# Functions
# ------------------------
def load_forecasts(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df[[TRUE_COL, PRED_COL]].rename(columns={PRED_COL: file_path.stem})

def compute_rmse(df, true_vals):
    rmse = ((df.sub(true_vals, axis=0)) ** 2).mean().pow(0.5)
    return rmse.to_frame(name="RMSE")


def run_dm_tests(df, true_vals):
    results = {}
    for i, col1 in enumerate(df.columns):
        for col2 in list(df.columns)[i+1:]:
            series_length = len(df[col1])
            if series_length <= 1:
                results[f"{col1} vs {col2}"] = {"DM_stat": np.nan, "p_value": np.nan}
                continue
            try:
                stat, pval = dm_test(true_vals, df[col1], df[col2], h=1, one_sided=False)
                results[f"{col1} vs {col2}"] = {"DM_stat": stat, "p_value": pval}
            except Exception as e:
                results[f"{col1} vs {col2}"] = {"DM_stat": np.nan, "p_value": np.nan}
                print(f"DM test failed for {col1} vs {col2}: {e}")
    return pd.DataFrame(results).T




def plot_rmse(rmse_series):
    rmse_series.plot(kind="bar", figsize=(8, 5), ylabel="RMSE", title="Forecast RMSE Comparison")
    plt.tight_layout()
    plt.show()

def plot_forecasts_with_target(true_vals, preds_df):
    plt.figure(figsize=(10, 6))

    # Plot true values: solid black with transparency
    plt.plot(true_vals.index, true_vals, label="Actual", color="black", linewidth=2, alpha=0.7)

    # Use distinct markers for each prediction series
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    for i, col in enumerate(preds_df.columns):
        plt.plot(preds_df.index, preds_df[col], label=col,
                 marker=markers[i % len(markers)], linestyle="solid", markersize=5)

    # Format x-axis: fewer ticks and rotation
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')

    plt.title("Actual vs Forecasts")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)




if __name__ == "__main__":

    dfs = [load_forecasts(OUTPUT_DIR / f) for f in FORECAST_FILES]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df.drop(columns=TRUE_COL), how="inner")

    true = merged[TRUE_COL]
    preds = merged.drop(columns=TRUE_COL)
    # pdb.set_trace()
    rmse_df = compute_rmse(preds, true)
    dm_results = run_dm_tests(preds, true)

    print("RMSE Summary:")
    print(rmse_df)
    print("\nDiebold-Mariano Tests:")
    print(dm_results)

    plot_rmse(rmse_df["RMSE"])
    true.index = pd.to_datetime(true.index)
    preds.index = pd.to_datetime(preds.index)
    plot_forecasts_with_target(true, preds)
