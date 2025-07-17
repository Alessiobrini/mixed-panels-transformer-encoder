import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Config ---
EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "experiments"
TARGETS = [
    "GDPC1", "GPDIC1", "PCECC96", "DPIC96", "OUTNFB", "UNRATE",
    "PCECTPI", "PCEPILFE", "CPIAUCSL", "CPILFESL", "FPIx", "EXPGSC1", "IMPGSC1"
]
PRED_FILES = {
    "transformer": "transformer_preds",
    "ar": "ar_preds",
    "midas": "midas_preds"
}
DM_FILENAME = "dm_test_results.csv"

# --- Metrics ---
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_errors(df, period_end=None):
    if period_end:
        df = df[df['date'] <= period_end]
    y_true = df.iloc[:, 1].values
    y_pred = df.iloc[:, 2].values
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),  # manual RMSE
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred)
    }

# --- Storage ---
dm_results = []
prediction_metrics = []

# --- Walk folders ---
for target in TARGETS:
    for folder in sorted(EXPERIMENT_DIR.glob(f"{target}_*")):
        suffix = folder.name.split("_", 1)[-1]  # e.g., 2025-07-16

        # DM test parsing
        dm_path = folder / DM_FILENAME
        if dm_path.exists():
            df_dm = pd.read_csv(dm_path, sep=None, engine="python")
            for _, row in df_dm.iterrows():
                dm_results.append({
                    "target": target,
                    "date": suffix,
                    "comparison": row.iloc[0],
                    "DM_stat": row.iloc[1],
                    "p_value": row.iloc[2]
                })

        # Prediction error metrics
        for model, filename_start in PRED_FILES.items():
            pred_file = next(folder.glob(f"{filename_start}_*.csv"), None)
            if pred_file is None:
                continue

            df = pd.read_csv(pred_file, parse_dates=['date'])
            # Compute metrics
            full = compute_errors(df)
            early = compute_errors(df, period_end=pd.Timestamp("2019-12-31"))

            for period, metrics in zip(["full", "pre_2020"], [full, early]):
                prediction_metrics.append({
                    "target": target,
                    "date": suffix,
                    "model": model,
                    "period": period,
                    **metrics
                })

# --- Build DataFrames ---
df_dm = pd.DataFrame(dm_results)
df_dm.set_index(["target", "date", "comparison"], inplace=True)

df_metrics = pd.DataFrame(prediction_metrics)
df_metrics.set_index(["target", "date", "model", "period"], inplace=True)


