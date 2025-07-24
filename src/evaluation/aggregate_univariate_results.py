import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import yaml


def get_best_targets(df_metrics, model_name, period, metric):
    df_flat = df_metrics.reset_index()

    # Filter by period and exclude the 'ar' model
    df_period = df_flat[(df_flat["period"] == period) & (df_flat["model"] != "ar")]

    # Find best model per target (among non-AR models)
    best_models = (
        df_period.groupby("target")
        .apply(lambda g: g.loc[g[metric].idxmin(), "model"], include_groups=False)
    )

    # Return targets where the selected model is best
    return best_models[best_models == model_name].index.tolist()


# --- Config ---
EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "experiments"
EXPERIMENT_DATE = "2025-07-18"  # <-- Set this to the date you want to analyze
PLOT_PRE_COVID_ONLY = False      # Set to False to plot full range

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
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred)
    }

# --- Storage ---
dm_results = []
prediction_metrics = []

# --- Walk through each target ---
for target in TARGETS:
    folder = EXPERIMENT_DIR / f"{target}_{EXPERIMENT_DATE}"
    if not folder.exists():
        print(f"Skipping {target}: folder for date {EXPERIMENT_DATE} not found.")
        continue

    suffix = EXPERIMENT_DATE

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
        early = compute_errors(df, period_end=pd.Timestamp("2019-06-30"))

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

# --- Analyze model performance: count how often each model is best ---
print("\n=== Model performance summary: number of times each model is best per metric and period ===")

df_flat = df_metrics.reset_index()
df_flat = df_flat[df_flat["model"] != "ar"]
metric_names = ["RMSE", "MAE", "MAPE"]
win_counts = {metric: {} for metric in metric_names}

for metric in metric_names:
    grouped = df_flat.groupby(["target", "date", "period"])
    best_models = grouped.apply(lambda g: g.loc[g[metric].idxmin(), "model"], include_groups=False)
    counts = best_models.groupby([best_models.index.get_level_values("period"), best_models]).size()

    for (period, model), count in counts.items():
        if model not in win_counts[metric]:
            win_counts[metric][model] = {}
        win_counts[metric][model][period] = count

# --- Print formatted results ---
for metric, data in win_counts.items():
    df_result = pd.DataFrame(data).fillna(0).astype(int).T
    df_result.columns.name = "Period"
    print(f"\n=== Best model counts by {metric} ===")
    print(df_result)

# --- Plotting predictions per target ---
print("\n=== Plotting predictions per target ===")
for target in TARGETS:
    folder = EXPERIMENT_DIR / f"{target}_{EXPERIMENT_DATE}"
    if not folder.exists():
        print(f"Skipping plot for {target}: folder for date {EXPERIMENT_DATE} not found.")
        continue

    fig, ax = plt.subplots(figsize=(10, 4))
    true_df = None

    for model, filename_start in PRED_FILES.items():
        pred_file = next(folder.glob(f"{filename_start}_*.csv"), None)
        if pred_file is None:
            continue

        df = pd.read_csv(pred_file, parse_dates=['date'])

        if PLOT_PRE_COVID_ONLY:
            df = df[df['date'] <= pd.Timestamp("2019-06-30")]

        if true_df is None:
            true_df = df.iloc[:, :2]
            ax.plot(true_df['date'], true_df.iloc[:, 1], 'k--', label='True')

        ax.plot(df['date'], df.iloc[:, 2], label=model.capitalize())

    ax.set_title(f"{target} — Model Predictions{' (pre-COVID)' if PLOT_PRE_COVID_ONLY else ''}")
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.show()

# best_rmse_targets = get_best_targets(df_metrics, model_name="transformer", period="full", metric="RMSE")
# print("Transformer is best (lowest RMSE) for these targets in 'full':", best_rmse_targets)



records = []

for target in TARGETS:
    folder = EXPERIMENT_DIR / f"{target}_{EXPERIMENT_DATE}"
    param_file = folder / "best_params.yaml"
    if not param_file.exists():
        print(f"Missing best_params.yaml for {target}, skipping.")
        continue

    with open(param_file, "r") as f:
        params = yaml.safe_load(f)
        params["target"] = target
        records.append(params)

# Convert to DataFrame
df_params = pd.DataFrame(records).set_index("target")
