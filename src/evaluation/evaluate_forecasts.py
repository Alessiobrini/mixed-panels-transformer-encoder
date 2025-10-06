import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dieboldmariano import dm_test
from src.utils.config import Config
from src.utils.data_paths import get_output_path, resolve_data_paths
import pdb
import shutil


# ------------------------
# Functions
# ------------------------
def load_forecasts(file_path, true_col, pred_col):
    df = pd.read_csv(file_path, index_col=0)
    return df[[true_col, pred_col]].rename(columns={pred_col: file_path.stem})

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
    rmse_series.plot(
        kind="bar",
        figsize=(8, 5),
        ylabel="RMSE",
        title="Forecast RMSE Comparison"
    )
    plt.tight_layout()
    plt.show()

def plot_forecasts_with_target(true_vals, preds_df):
    plt.figure(figsize=(10, 6))
    plt.plot(true_vals.index, true_vals,
             label="Actual", color="black", linewidth=2, alpha=0.7)

    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    for i, col in enumerate(preds_df.columns):
        plt.plot(
            preds_df.index,
            preds_df[col],
            label=col,
            marker=markers[i % len(markers)],
            linestyle="solid",
            markersize=5
        )

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
    

    # ------------------------
    # Config
    # ------------------------
    cfg_path = project_root / "src" / "config" / "cfg.yaml"
    config = Config(cfg_path)
    
    # Determine suffix
    _, suffix, _, _ = resolve_data_paths(config, project_root)
    exp_name = getattr(config.training, "experiment_name", None)
    

    if exp_name:
        exp_path = project_root / "outputs" / "experiments" / exp_name
        exp_path.mkdir(parents=True, exist_ok=True)
    
        for key in config.evaluation.forecast_files:
            src_path = get_output_path(config, project_root, key, suffix)

            # Only copy AR and MIDAS predictions; transformer already lives there
            if key != "transformer_preds":
                dst_path = exp_path / src_path.name
                shutil.copy(src_path, dst_path)
    
    # Build list of forecast-CSV paths
    forecast_dir = exp_path if exp_name else project_root / "outputs"
    forecast_filenames = [
        Path(getattr(config.paths.outputs, key).format(suffix=suffix)).name
        for key in config.evaluation.forecast_files
    ]
    FORECAST_PATHS = [forecast_dir / name for name in forecast_filenames]

    
    TRUE_COL = config.evaluation.true_col
    PRED_COL = config.evaluation.pred_col

    # Load and merge predictions
    dfs = []
    for path in FORECAST_PATHS:
        df = load_forecasts(path, TRUE_COL, PRED_COL)
        
        # Trim the first row if the model is AR or MIDAS
        filename = path.stem.lower()
        if "ar" in filename or "midas" in filename:
            df = df.iloc[1:]
        
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df.drop(columns=TRUE_COL), how="inner")

    true = merged[TRUE_COL]
    preds = merged.drop(columns=TRUE_COL)

    # Compute metrics
    rmse_df = compute_rmse(preds, true)
    dm_results = run_dm_tests(preds, true)

    print("RMSE Summary:")
    print(rmse_df)
    print("\nDiebold-Mariano Tests:")
    print(dm_results)
    
    # Save results if experiment folder is defined
    if exp_name:
        exp_path = project_root / "outputs" / "experiments" / exp_name
        exp_path.mkdir(parents=True, exist_ok=True)
        rmse_df.to_csv(exp_path / "rmse_summary.csv")
        dm_results.to_csv(exp_path / "dm_test_results.csv")
        print(f"\nSaved RMSE and DM test results to: {exp_path}")
    else:
        # Plot results
        plot_rmse(rmse_df["RMSE"])
        true.index = pd.to_datetime(true.index)
        preds.index = pd.to_datetime(preds.index)
        plot_forecasts_with_target(true, preds)
