import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch.bootstrap import MCS
from typing import Optional

import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# --- Config ---
EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "experiments"
EXPERIMENT_DATE = "2025-10-20" #"2025-10-17"
SCENARIO_PREFIX = "synth_"
TIME_COLUMN = "date"

PRED_FILES = {
    "transformer": "transformer_preds",
    "midas": "midas_preds",
    "ar": "ar_preds",
}

DM_FILENAME = "dm_test_results.csv"


# --- Metrics ---
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true, y_pred):
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return np.mean(true_dir == pred_dir)


def compute_errors(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "DA": directional_accuracy(y_true, y_pred),
    }


# --- Helpers ---
def parse_scenario_and_date(folder_name: str):
    if not folder_name.startswith(SCENARIO_PREFIX):
        raise ValueError(f"Unexpected folder name: {folder_name}")
    remainder = folder_name[len(SCENARIO_PREFIX):]
    if "_" not in remainder:
        raise ValueError(
            f"Synthetic experiment folders must follow '{SCENARIO_PREFIX}<scenario>_<date>' naming; got {folder_name}"
        )
    scenario, date_suffix = remainder.rsplit("_", 1)
    return scenario, date_suffix


def discover_experiment_folders(date_filter: Optional[str]):
    if date_filter:
        pattern = f"{SCENARIO_PREFIX}*_{date_filter}"
    else:
        pattern = f"{SCENARIO_PREFIX}*"
    return sorted([p for p in EXPERIMENT_DIR.glob(pattern) if p.is_dir()])


def run_mcs(y_true, preds_dict, size=0.10, reps=1000, block_size=None, method="R", bootstrap="stationary", seed=None):
    model_names = list(preds_dict.keys())
    losses = np.column_stack([(y_true - preds_dict[m]) ** 2 for m in model_names])
    mcs = MCS(losses, size=size, reps=reps, block_size=block_size, method=method, bootstrap=bootstrap, seed=seed)
    mcs.compute()
    included = [model_names[i] for i in mcs.included]
    pvalues = mcs.pvalues
    return included, pvalues


# --- Storage ---
dm_results = []
prediction_metrics = []
plot_data = {}

scenario_folders = discover_experiment_folders(EXPERIMENT_DATE)

if not scenario_folders:
    print("No synthetic experiment folders were found. Check EXPERIMENT_DATE and EXPERIMENT_DIR.")

for folder in scenario_folders:
    scenario, suffix = parse_scenario_and_date(folder.name)

    # Load DM results
    dm_path = folder / DM_FILENAME
    if dm_path.exists():
        df_dm = pd.read_csv(dm_path, sep=None, engine="python")
        for _, row in df_dm.iterrows():
            dm_results.append({
                "scenario": scenario,
                "date": suffix,
                "comparison": row.iloc[0],
                "DM_stat": row.iloc[1],
                "p_value": row.iloc[2],
            })

    # Load predictions
    raw_dfs = []
    pred_dfs = []
    for model, prefix in PRED_FILES.items():
        file = next(folder.glob(f"{prefix}_*.csv"), None)
        if file is None:
            continue
        df = pd.read_csv(file)
        time_col = df.columns[0]
        df = df.rename(columns={time_col: TIME_COLUMN, df.columns[1]: "true", df.columns[2]: model})
        raw_dfs.append(df[[TIME_COLUMN, "true"]].copy())
        pred_dfs.append(df.set_index(TIME_COLUMN))

    if not pred_dfs:
        print(f"Skipping {scenario}: no prediction files found.")
        continue

    # Target consistency check
    true_series = []
    for idx, df_raw in enumerate(raw_dfs):
        series = df_raw.set_index(TIME_COLUMN)["true"].round(6)
        true_series.append(series.rename(f"true_{idx}"))
    df_true_compare = pd.concat(true_series, axis=1, join="inner")
    row_diff = df_true_compare.max(axis=1) - df_true_compare.min(axis=1)
    tolerance = 1e-5
    unequal = row_diff > tolerance

    if unequal.any():
        print(f"Inconsistent target values for {scenario} at indices (tol={tolerance}):")
        print(df_true_compare[unequal].head())
        raise ValueError(f"Target mismatch detected for {scenario}.")

    # Merge predictions
    merged = pred_dfs[0].copy()
    for df in pred_dfs[1:]:
        merged = merged.join(df.drop(columns="true", errors="ignore"), how="inner")
    merged.reset_index(inplace=True)
    merged["scenario"] = scenario

    plot_data[scenario] = merged.copy()

    y_true_full = merged["true"].values
    for model in PRED_FILES:
        if model not in merged.columns:
            continue
        y_pred = merged[model].values
        metrics = compute_errors(y_true_full, y_pred)
        prediction_metrics.append({
            "scenario": scenario,
            "date": suffix,
            "model": model,
            **metrics,
        })

# --- Build DataFrames ---
df_dm = pd.DataFrame(dm_results).set_index(["scenario", "date", "comparison"]) if dm_results else pd.DataFrame()
df_metrics = pd.DataFrame(prediction_metrics).set_index(["scenario", "date", "model"]) if prediction_metrics else pd.DataFrame()

# --- Summary statistics ---
if not df_metrics.empty:
    print("\n=== Model performance summary ===")
    metric_names = ["RMSE", "MAE", "MAPE", "DA"]
    df_flat = df_metrics.reset_index()
    df_flat = df_flat[df_flat["model"] != "ar"]

    for metric in metric_names:
        if metric == "DA":
            def best_idx(g):
                return g[metric].idxmax()
        else:
            def best_idx(g):
                return g[metric].idxmin()

        best_models = (
            df_flat
            .groupby(["scenario", "date"])
            .apply(lambda g: g.loc[best_idx(g), "model"], include_groups=False)
        )

        counts = best_models.groupby(best_models).size().sort_values(ascending=False)

        print(f"\n=== Best model counts by {metric} ===")
        print(counts)
else:
    print("No prediction metrics available.")

# --- Plotting ---
PLOT_MODELS = ["transformer", "midas", "ar"]
print("\n=== Plotting predictions per scenario ===")
for scenario, merged in plot_data.items():
    fig, ax = plt.subplots(figsize=(10, 4))
    df_plot = merged.copy()
    ax.plot(df_plot[TIME_COLUMN], df_plot["true"], "k--", label="True")
    for model in PLOT_MODELS:
        if model in df_plot.columns:
            ax.plot(df_plot[TIME_COLUMN], df_plot[model], label=model.capitalize())
    ax.set_title(f"{scenario} Predictions")
    ax.set_xlabel(TIME_COLUMN)
    ax.legend()
    plt.tight_layout()
    plt.show()

# --- MCS ---
print("\n=== Model Confidence Set (MCS) Results ===")
mcs_results = []

for scenario, merged in plot_data.items():
    y_true = merged["true"].values
    preds = {
        model: merged[model].values
        for model in PRED_FILES
        if model in merged.columns and model != "ar"
    }
    if len(preds) < 2:
        continue

    try:
        included_models, pvals = run_mcs(y_true, preds, size=0.10, reps=1000, seed=42)
        mcs_results.append({
            "scenario": scenario,
            "included_models": included_models,
            "pvalues": pvals,
        })
    except Exception as exc:
        print(f"Skipping MCS for {scenario} due to error: {exc}")

print("\n=== MCS Summary by Scenario ===")
for row in mcs_results:
    included = ", ".join(row["included_models"]) if row["included_models"] else "<none>"
    print(f"{row['scenario']}: included = {included}")
    print("p-values:")
    print(row["pvalues"])
    print()

model_counter = Counter()
model_pvalues = defaultdict(list)

num_scenarios = len(mcs_results)

for row in mcs_results:
    included = row["included_models"]
    pvals = row["pvalues"]

    for model in included:
        model_counter[model] += 1
        if hasattr(pvals, "index") and model in pvals.index:
            model_pvalues[model].append(pvals.loc[model, "Pvalue"])
        elif hasattr(pvals, "columns") and model in pvals.columns:
            model_pvalues[model].append(pvals.loc[:, model].mean())
        else:
            model_pvalues[model].append(np.nan)

print("\n=== MCS Inclusion Frequency and Avg p-value ===")
print(f"Total scenarios evaluated: {num_scenarios}\n")

for model in sorted(model_counter):
    count = model_counter[model]
    freq = count / num_scenarios if num_scenarios else np.nan
    pvals = [p for p in model_pvalues[model] if not np.isnan(p)]
    avg_p = np.mean(pvals) if pvals else np.nan
    freq_str = f"{freq:.1%}" if num_scenarios else "nan"
    avg_p_str = f"{avg_p:.3f}" if not np.isnan(avg_p) else "nan"
    print(f"{model:<12} included in {count:2d} / {num_scenarios} scenarios ({freq_str}), avg p-value = {avg_p_str}")



dft =df_metrics.xs('transformer',level=2)
dftnss = dft[dft.index.get_level_values("scenario").str.contains("nss")]
dftlss = dft[dft.index.get_level_values("scenario").str.contains("lss")]

dfm = df_metrics.xs('midas',level=2)
dfmnss = dfm[dfm.index.get_level_values("scenario").str.contains("nss")]
dfmlss = dfm[dfm.index.get_level_values("scenario").str.contains("lss")]