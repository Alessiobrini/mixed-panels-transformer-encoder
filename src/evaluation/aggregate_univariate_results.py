import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch.bootstrap import MCS

import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 14          # base font
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14

import matplotlib.pyplot as plt
import yaml, sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from src.evaluation.evaluate_forecasts import run_dm_tests

# --- Config ---
INCLUDE_ABLATIONS = True
EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "experiments"
EXPERIMENT_DATE = "2025-09-26"
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


# --- MCS helper ---
def run_mcs(y_true, preds_dict, size=0.10, reps=5000, block_size=None, method="R", bootstrap="stationary", seed=None):
    """
    Compute the Model Confidence Set using arch.bootstrap.MCS.
    """
    model_names = list(preds_dict.keys())
    # build loss matrix as squared errors (T×k)
    losses = np.column_stack([(y_true - preds_dict[m])**2 for m in model_names])

    # instantiate and run MCS with no 'loss' argument
    mcs = MCS(losses, size=size, reps=reps, block_size=block_size, method=method, bootstrap=bootstrap, seed=seed)
    mcs.compute()

    included = [model_names[i] for i in mcs.included]
    pvalues = mcs.pvalues  # DataFrame indexed by model index or name
    return included, pvalues


# --- Storage ---
dm_results = []
dm_full_results = []
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

    def load_prediction(file_path, model_name):
        df_loaded = pd.read_csv(file_path, parse_dates=['date'])
        df_loaded = df_loaded.rename(columns={df_loaded.columns[1]: "true", df_loaded.columns[2]: model_name})
        df_loaded["target"] = target
        raw_dfs.append(df_loaded[['date', 'true']].copy())
        pred_dfs.append(df_loaded.set_index(['date', 'target']))

    for model, prefix in PRED_FILES.items():
        file = next(folder.glob(f"{prefix}_*.csv"), None)
        if file is None:
            continue
        load_prediction(file, model)

    # Load ablation transformer predictions if requested
    if INCLUDE_ABLATIONS:
        ablation_prefix = f"{target}_{EXPERIMENT_DATE}_"
        ablation_dirs = [p for p in sorted(EXPERIMENT_DIR.glob(f"{ablation_prefix}*")) if p.is_dir()]

        for ab_dir in ablation_dirs:
            ablation_name = ab_dir.name[len(ablation_prefix):]
            if not ablation_name:
                continue
            short_code = ablation_name.split("_")[0]
            model_name = f"transformer_A{short_code}"
            ab_file = next(ab_dir.glob(f"{PRED_FILES['transformer']}_*.csv"), None)
            if ab_file is None:
                continue
            load_prediction(ab_file, model_name)

    if not pred_dfs:
        continue

    # --- Target consistency check ---
    # Build a DataFrame of true values from each model, rounded for stability
    true_series = []
    for idx, df_raw in enumerate(raw_dfs):
        series = df_raw.set_index('date')['true'].round(6)
        true_series.append(series.rename(f"true_{idx}"))
    df_true_compare = pd.concat(true_series, axis=1, join='inner')
    # compute max difference between min and max per row
    row_diff = df_true_compare.max(axis=1) - df_true_compare.min(axis=1)
    tolerance = 1e-5
    unequal = row_diff > tolerance
    
    if unequal.any():
        print(f"Inconsistent target values for {target} on dates (tol={tolerance}):")
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
        model_columns = [col for col in subset.columns if col not in {"date", "target", "true"}]
        for model in model_columns:
            y_pred = subset[model].values
            metrics = compute_errors(y_true, y_pred)
            prediction_metrics.append({
                "target": target,
                "date": suffix,
                "model": model,
                "period": period,
                **metrics
            })

        # --- Diebold-Mariano tests (baseline transformer vs others) ---
        baseline = "transformer"
        if baseline in model_columns and len(subset) > 1:
            others = [m for m in model_columns if m != baseline]
            if others:
                preds_df = subset[[baseline] + others].copy()
                dm_period = run_dm_tests(preds_df.drop(columns=["date", "target", "true"], errors="ignore"), subset['true'])

                for other in others:
                    pair_key = f"{baseline} vs {other}"
                    reverse_key = f"{other} vs {baseline}"
                    if pair_key in dm_period.index:
                        row_dm = dm_period.loc[pair_key]
                    elif reverse_key in dm_period.index:
                        row_dm = dm_period.loc[reverse_key]
                    else:
                        row_dm = pd.Series({"DM_stat": np.nan, "p_value": np.nan})

                    dm_full_results.append({
                        "target": target,
                        "date": suffix,
                        "period": period,
                        "model_1": baseline,
                        "model_2": other,
                        "DM_stat": row_dm.get("DM_stat", np.nan),
                        "p_value": row_dm.get("p_value", np.nan)
                    })

# --- Build DataFrames ---
df_dm = pd.DataFrame(dm_results).set_index(['target', 'date', 'comparison'])
df_metrics = pd.DataFrame(prediction_metrics).set_index(['target', 'date', 'model', 'period'])
df_dm_full = pd.DataFrame(dm_full_results)

# --- Validate new DM results against loaded CSVs (full period only) ---
if not df_dm.empty and not df_dm_full.empty:
    for _, row in df_dm_full[df_dm_full["period"] == "full"].iterrows():
        comp = f"{row['model_1']} vs {row['model_2']}"
        rev = f"{row['model_2']} vs {row['model_1']}"
        idx = (row['target'], row['date'], comp)
        rev_idx = (row['target'], row['date'], rev)

        if idx in df_dm.index:
            existing = df_dm.loc[idx]
        elif rev_idx in df_dm.index:
            existing = df_dm.loc[rev_idx]
        else:
            continue

        if pd.notna(existing['DM_stat']) and pd.notna(row['DM_stat']):
            if abs(existing['DM_stat'] - row['DM_stat']) > 1e-3:
                print(
                    f"Warning: DM discrepancy for {row['target']} ({row['period']})"
                    f" between stored and recomputed values: "
                    f"stored={existing['DM_stat']}, new={row['DM_stat']}"
                )

# --- LaTeX helper ---
def make_latex_table_for_target(df_metrics, target, experiment_date, outfile=None):
    """Generate a LaTeX table for transformer and ablation models for a single target.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Metrics with MultiIndex ['target', 'date', 'model', 'period'] and columns
        including ['RMSE', 'MAE', 'MAPE', 'DA'].
    target : str
        Target series name (e.g., "GDPC1").
    experiment_date : str
        Experiment date string matching the 'date' level in df_metrics.
    outfile : str or Path, optional
        If provided, write the LaTeX output to this path.

    Returns
    -------
    str
        The LaTeX table as a string.
    """

    df_filtered = (
        df_metrics
        .reset_index()
        .query("target == @target and date == @experiment_date")
    )

    df_filtered = df_filtered[
        (df_filtered["model"] == "transformer")
        | (df_filtered["model"].str.startswith("transformer_AB"))
    ]

    model_order = [
        "transformer",
        "transformer_AB1",
        "transformer_AB2",
        "transformer_AB3",
        "transformer_AB5",
        "transformer_AB6",
    ]
    
    # remap models so AB5 → AB4 and AB6 → AB5
    rename_map = {
        "transformer_AB5": "transformer_AB4",
        "transformer_AB6": "transformer_AB5",
    }
    
    df_filtered["model"] = df_filtered["model"].replace(rename_map)
    model_order = [rename_map.get(m, m) for m in model_order]

    periods = ["full", "pre_2020", "post_2020"]
    metrics = ["RMSE", "MAE", "DA"]

    def fmt(value):
        return f"{value:.4f}" if pd.notna(value) else ""

    lines = [
        "\\begin{table}[]",
        f"\\caption{{Metrics for {target}}}",
        "\\begin{tabular}{@{}cccccccccc@{}}",
        "\\toprule",
        "                  & \\multicolumn{3}{c}{\\textbf{Full}} & \\multicolumn{3}{c}{\\textbf{Pre-2020}} & \\multicolumn{3}{c}{\\textbf{Post-2020}} \\\\ \\midrule",
        "                  & RMSE      & MAE       & DA        & RMSE        & MAE        & DA         & RMSE        & MAE         & DA         \\\\ \\midrule",
        f"\\textbf{{{target}}} &           &           &           &             &            &            &             &             &             \\\\ \\midrule",
    ]

    for model in model_order:
        df_model = df_filtered[df_filtered["model"] == model]
        if df_model.empty:
            continue

        # rename model for display
        display_name = (
            "transformer" if model == "transformer"
            else model.replace("transformer_AB", "AB")
        )
        row_entries = [display_name]

        for period in periods:
            df_period = df_model[df_model["period"] == period]
            if df_period.empty:
                row_entries.extend([""] * len(metrics))
                continue

            values = df_period.iloc[0]
            row_entries.extend([fmt(values[m]) for m in metrics])

        line = "   ".join(row_entries[0:1]) + "   & " + " & ".join(row_entries[1:]) + " \\\\"  # noqa: E501
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_output = "\n".join(lines)

    if outfile is not None:
        Path(outfile).write_text(latex_output)

    return latex_output


def make_dm_latex_tables(df_dm_full, experiment_date, precision=2):
    """Return LaTeX tables (as a string) for DM t-stats using df_dm_full.

    This consumes already-computed results and only uses the full-period rows
    for the provided experiment date. DM t-statistics are formatted with a
    fixed precision and missing comparisons are shown as ``--``.
    """

    df_filtered = df_dm_full[
        (df_dm_full["period"] == "full") & (df_dm_full["date"] == experiment_date)
    ]

    def fmt(val):
        return "--" if pd.isna(val) else f"{val:.{precision}f}"

    targets = [t for t in TARGETS if t in df_filtered["target"].unique()]

    # Table 1: Transformer vs competing models
    comp_models = ["midas", "ar", "ols", "xgb", "nn"]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Diebold--Mariano tests: Transformer vs competing models}",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Target & MIDAS & AR & OLS & XGB & NN \\\\",
        "\\midrule",
    ]

    for target in targets:
        row_vals = [target]
        for model in comp_models:
            match = df_filtered[
                (df_filtered["target"] == target)
                & (df_filtered["model_1"] == "transformer")
                & (df_filtered["model_2"] == model)
            ]
            val = match["DM_stat"].iloc[0] if not match.empty else np.nan
            row_vals.append(fmt(val))
        lines.append(" & ".join(row_vals) + " \\\\")


    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    # Table 2: Transformer vs ablations
    ablation_map = {
        "transformer_AB1": "AB1",
        "transformer_AB2": "AB2",
        "transformer_AB3": "AB3",
        "transformer_AB5": "AB4",
        "transformer_AB6": "AB5",
    }
    ablation_order = [ablation_map[k] for k in sorted(ablation_map)]
    ablation_lookup = {v: k for k, v in ablation_map.items()}

    lines.extend([
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Diebold--Mariano tests: Transformer vs ablations}",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Target & AB1 & AB2 & AB3 & AB4 & AB5 \\\\",
        "\\midrule",
    ])

    for target in targets:
        row_vals = [target]
        for ab_short in ablation_order:
            model_name = ablation_lookup[ab_short]
            match = df_filtered[
                (df_filtered["target"] == target)
                & (df_filtered["model_1"] == "transformer")
                & (df_filtered["model_2"] == model_name)
            ]
            val = match["DM_stat"].iloc[0] if not match.empty else np.nan
        lines.append(" & ".join(row_vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)

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
PLOT_MODELS = ['transformer', 'midas', 'xgb']
# Optional legend placement per target; use None or omit for default placement
# Example: {"GDPC1": "upper left", "UNRATE": {"loc": "lower right", "bbox_to_anchor": (1, 0.5)}}
PLOT_LEGEND_ARGS = {"GDPC1": "lower left" , "OUTNFB": "lower left"}
print("\n=== Plotting predictions per target ===")
for target, merged in plot_data.items():
    fig, ax = plt.subplots(figsize=(10, 4))
    df_plot = merged.copy()
    if PLOT_PRE_COVID_ONLY:
        df_plot = df_plot[df_plot['date'] <= pd.Timestamp("2019-06-30")]
    ax.plot(df_plot['date'], df_plot['true'], 'k--', label='Actual')

    label_map = {
        'transformer': 'MPTE',
        'midas': 'MIDAS',
        'xgb': 'XGB',
        'xgboost': 'XGB',
    }

    style_map = {
        'transformer': {"color": "#003f5c", "linewidth": 2.2, "alpha": 1.0},
        'midas': {"color": "#bc6c25", "linewidth": 1.4, "alpha": 0.85},
        'xgb': {"color": "#4c956c", "linewidth": 1.4, "alpha": 0.85},
        'xgboost': {"color": "#4c956c", "linewidth": 1.4, "alpha": 0.85},
    }

    for model in PLOT_MODELS:
        if model in df_plot.columns:
            style_kwargs = style_map.get(model, {})
            ax.plot(
                df_plot['date'],
                df_plot[model],
                label=label_map.get(model, model),
                linestyle='-',
                **style_kwargs,
            )
    ax.set_title(f"{target}")
    legend_cfg = PLOT_LEGEND_ARGS.get(target)
    if legend_cfg is None:
        ax.legend()
    elif isinstance(legend_cfg, dict):
        ax.legend(**legend_cfg)
    else:
        ax.legend(loc=legend_cfg)
    plt.tight_layout()
    plt.show()

# --- MCS ---
print("\n=== Model Confidence Set (MCS) Results ===")
mcs_results = []

for target, merged in plot_data.items():
    for period, mask in {
        "full": slice(None),
        "pre_2020": merged['date'] <= pd.Timestamp("2019-06-30"),
        "post_2020": merged['date'] > pd.Timestamp("2019-06-30")
    }.items():
        df_sub = merged.loc[mask]
        y_true = df_sub['true'].values

        # Collect predictions (skip AR if needed)
        preds = {
            model: df_sub[model].values
            for model in df_sub.columns
            if model not in {"date", "target", "true", "ar"}
        }

        if len(preds) < 2:
            continue  # MCS needs at least 2 models

        try:
            included_models, pvals = run_mcs(y_true, preds, size=0.10, reps=1000, seed=42)
            mcs_results.append({
                "target": target,
                "period": period,
                "included_models": included_models,
                "pvalues": pvals
            })
        except Exception as e:
            print(f"Skipping MCS for {target} ({period}) due to error: {e}")

print("\n=== MCS Summary by Target (Full Period Only) ===")
for row in mcs_results:
    if row['period'] != "full":
        continue  # skip pre/post 2020

    print(f"{row['target']} ({row['period']}): included = {', '.join(row['included_models'])}")
    print("p-values:")
    print(row["pvalues"])
    print()

# --- MCS model frequency summary (full period only) ---
from collections import Counter, defaultdict

# --- Track frequency and p-values per model ---
model_counter = Counter()
model_pvalues = defaultdict(list)

for row in mcs_results:
    if row["period"] != "full":
        continue
    included = row["included_models"]
    pvals = row["pvalues"]

    for model in included:
        model_counter[model] += 1
        # Get index of this model in pval DataFrame
        for idx, name in enumerate(pvals.index):
            if model_counter.keys() == pvals.index.tolist():
                model_pvalues[model].append(pvals.loc[name, "Pvalue"])
                break
        else:
            # fallback: match model by name if available
            model_pvalues[model].append(pvals["Pvalue"].mean())

# --- Print table ---
num_targets = sum(1 for row in mcs_results if row["period"] == "full")

print("\n=== MCS Inclusion Frequency and Avg p-value (Full Period) ===")
print(f"Total targets evaluated: {num_targets}\n")

for model in sorted(model_counter):
    count = model_counter[model]
    freq = count / num_targets
    avg_p = np.mean(model_pvalues[model])
    print(f"{model:<12} included in {count:2d} / {num_targets} targets ({freq:.1%}), avg p-value = {avg_p:.3f}")


def build_mcs_latex_tables(mcs_results):
    """Return LaTeX tables summarizing MCS inclusion for the full period.

    Parameters
    ----------
    mcs_results : list of dict
        Output rows produced earlier in this module. Each row contains keys
        ``target``, ``period``, ``included_models``, and ``pvalues``.

    Returns
    -------
    str
        Combined LaTeX tables using booktabs. An empty string is returned if
        there are no full-period results.
    """

    df_mcs = pd.DataFrame(mcs_results)
    if df_mcs.empty:
        return ""

    df_mcs = df_mcs[df_mcs["period"] == "full"].copy()
    if df_mcs.empty:
        return ""

    def _format_latex(df, columns, caption):
        col_spec = "l" + "c" * (len(columns) - 1)
        tabular = df.to_latex(
            index=False,
            escape=False,
            column_format=col_spec,
        ).strip()
        return "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                rf"\caption{{{caption}}}",
                tabular,
                r"\end{table}",
            ]
        )

    # --- Table 1: baseline models (no ablations) ---
    model_columns = ["transformer", "midas", "ar", "ols", "xgb", "nn"]
    rows = []
    for _, row in df_mcs.iterrows():
        included = set(row["included_models"])
        values = ["\\checkmark" if m in included else "--" for m in model_columns]
        rows.append([row["target"], *values])

    df_table1 = pd.DataFrame(rows, columns=["target", *model_columns])

    # --- Table 2: transformer ablations ---
    ablation_columns = ["transformer", "AB1", "AB2", "AB3", "AB4", "AB5"]

    def _ablation_label(name):
        if name == "transformer":
            return "transformer"
        for prefix in ("transformer_AB", "transformer_A"):
            if name.startswith(prefix):
                label = name[len(prefix) :]
                return label if label.startswith("AB") else f"AB{label}"
        return None

    rows_ab = []
    for _, row in df_mcs.iterrows():
        included_labels = { _ablation_label(m) for m in row["included_models"] }
        values = [
            "\\checkmark" if col in included_labels else "--"
            for col in ablation_columns
        ]
        rows_ab.append([row["target"], *values])

    df_table2 = pd.DataFrame(rows_ab, columns=["target", *ablation_columns])

    table1 = _format_latex(
        df_table1, df_table1.columns, "MCS inclusion for baseline models"
    )
    table2 = _format_latex(
        df_table2, df_table2.columns, "MCS inclusion for transformer ablations"
    )

    return f"{table1}\n\n{table2}"
