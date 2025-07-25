import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import yaml

# --- Config ---
EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "experiments"
EXPERIMENT_DATE = "2025-07-24"
PLOT_PRE_COVID_ONLY = True

TARGETS = [
    "GDPC1", "GPDIC1", "PCECC96", "DPIC96", "OUTNFB", "UNRATE",
    "PCECTPI", "PCEPILFE", "CPIAUCSL", "CPILFESL", "FPIx", "EXPGSC1", "IMPGSC1"
]
PRED_FILES = {
    "transformer": "transformer_preds",
    "ar": "ar_preds",
    "midas": "midas_preds",
    "ols": "ols_preds",
    "xgb": "xgb_preds",
    "nn": "nn_preds",
    
}
DM_FILENAME = "dm_test_results.csv"

# --- Metrics ---
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def directional_accuracy(y_true, y_pred):
    # compute the sign of the change from t–1 to t
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    # hit rate: fraction of times the directions match
    return np.mean(true_dir == pred_dir)


def compute_errors(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "DA": directional_accuracy(y_true, y_pred)
    }

# --- Storage ---
dm_results = []
prediction_metrics = []
plot_data = {}  # store merged DataFrames for plotting later

# --- Walk through each target ---
for target in TARGETS:
    folder = EXPERIMENT_DIR / f"{target}_{EXPERIMENT_DATE}"
    if not folder.exists():
        print(f"Skipping {target}: folder not found.")
        continue

    suffix = EXPERIMENT_DATE

    # Load DM results if available
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

    # Load predictions and track true values for consistency check
    raw_dfs = []
    pred_dfs = []
    for model, prefix in PRED_FILES.items():
        file = next(folder.glob(f"{prefix}_*.csv"), None)
        if file is None:
            continue
        df = pd.read_csv(file, parse_dates=['date'])
        df = df.rename(columns={df.columns[1]: "true", df.columns[2]: model})
        df["target"] = target
        raw_dfs.append(df[['date', 'true']].copy())
        pred_dfs.append(df.set_index(['date', 'target']))

    if not pred_dfs:
        continue

    # --- Target consistency check ---
    # Build a DataFrame of true values from each model, rounded for stability
    true_series = []
    for idx, df_raw in enumerate(raw_dfs):
        series = df_raw.set_index('date')['true'].round(6)
        true_series.append(series.rename(f"true_{idx}"))
    df_true_compare = pd.concat(true_series, axis=1, join='inner')
    unequal = df_true_compare.nunique(axis=1) != 1
    if unequal.any():
        print(f"Inconsistent target values for {target} on dates:")
        print(df_true_compare[unequal].head())
        raise ValueError(f"Target mismatch detected for {target}.")

    # --- Merge and drop missing rows ---
    merged = pred_dfs[0].copy()
    for df in pred_dfs[1:]:
        merged = merged.join(df.drop(columns='true'), how='inner')
    merged.reset_index(inplace=True)

    # Store for plotting
    plot_data[target] = merged.copy()

    # --- Compute metrics for full, pre-COVID, post-COVID ---
    for period, mask in {
        "full": slice(None),
        "pre_2020": merged['date'] <= pd.Timestamp("2019-06-30"),
        "post_2020": merged['date'] > pd.Timestamp("2019-06-30")
    }.items():
        subset = merged.loc[mask]
        y_true = subset['true'].values
        for model in PRED_FILES:
            if model not in subset.columns:
                continue
            y_pred = subset[model].values
            metrics = compute_errors(y_true, y_pred)
            prediction_metrics.append({
                "target": target,
                "date": suffix,
                "model": model,
                "period": period,
                **metrics
            })

# --- Build DataFrames ---
df_dm = pd.DataFrame(dm_results).set_index(['target', 'date', 'comparison'])
df_metrics = pd.DataFrame(prediction_metrics).set_index(['target', 'date', 'model', 'period'])

# --- Print performance summary ---
print("\n=== Model performance summary ===")
metric_names = ["RMSE", "MAE", "MAPE", "DA"]
df_flat = df_metrics.reset_index()
df_flat = df_flat[df_flat['model'] != 'ar']

for metric in metric_names:
    # choose idxmin or idxmax depending on the metric
    if metric == "DA":
        best_idx = lambda g: g[metric].idxmax()
    else:
        best_idx = lambda g: g[metric].idxmin()

    best_models = (
        df_flat
        .groupby(['target', 'date', 'period'])
        .apply(lambda g: g.loc[best_idx(g), 'model'], 
               include_groups=False)
    )

    counts = best_models.groupby(
        [best_models.index.get_level_values('period'), best_models]
    ).size()

    print(f"\n=== Best model counts by {metric} ===")
    print(counts.unstack().fillna(0).astype(int))

# --- Plot using merged data ---
PLOT_MODELS = ['transformer', 'midas']
print("\n=== Plotting predictions per target ===")
for target, merged in plot_data.items():
    fig, ax = plt.subplots(figsize=(10, 4))
    df_plot = merged.copy()
    if PLOT_PRE_COVID_ONLY:
        df_plot = df_plot[df_plot['date'] <= pd.Timestamp("2019-06-30")]
    ax.plot(df_plot['date'], df_plot['true'], 'k--', label='True')
    for model in PLOT_MODELS:
        if model in df_plot.columns:
            ax.plot(df_plot['date'], df_plot[model], label=model.capitalize())
    ax.set_title(f"{target} Predictions")
    ax.legend()
    plt.tight_layout()
    plt.show()
