"""Reconcile and regenerate the simulation Table 1 (``Tab:evals_simulation``).

The paper's simulation table was originally assembled by hand from the per-scenario
outputs of ``aggregate_univariate_results_synth.py``. This script makes that table
reproducible end to end: it pins each (regime x model) cell to a specific experiment
folder, recomputes RMSE/MAE/DA with the *same* definitions used in the original
aggregation, reconciles the result against the numbers currently in ``main.tex``, and
emits the combined three-regime LaTeX table with best/second highlighting computed
programmatically.

Canonical regime -> experiment mapping (verified 2026-06-08 against rmse_summary.csv):

    Linear            -> synth_*_lss_2025-10-17   (g = identity)
    Mildly Nonlinear  -> synth_*_nss_2025-10-17   (RBF, K = 6)
    Highly Nonlinear  -> synth_*_nss_2025-10-20   (RBF, K = 12)

Note: AR/MIDAS are identical across the two linear-scenario dates (deterministic on the
same DGP); only MPTE retraining differs, and the paper used the 10-17 MPTE (1.2990).

DA definition: directional accuracy is the fraction of one-step *changes* whose sign is
predicted correctly (sign of np.diff), matching the original aggregation script. (The
manuscript prose describes it as "same sign of the level"; that wording does not match
the implementation that produced the table -- noted for the R2 revision, not changed here.)

Run:
    python src/evaluation/build_simulation_table.py
Outputs:
    outputs/tables/synthetic_table1_combined.tex
    outputs/tables/synthetic_table1_reconciliation.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "experiments"
TABLE_DIR = EXPERIMENT_DIR.parent / "tables"
TIME_COLUMN = "date"
RECONCILE_TOL = 1e-3  # paper prints 4 decimals; tolerance comfortably below rounding

# regime label -> (scenario tag, experiment date)
REGIMES = [
    ("Linear", "lss", "2025-10-17"),
    ("Mildly Nonlinear", "nss", "2025-10-17"),
    ("Highly Nonlinear", "nss", "2025-10-20"),
]

# table row order (display) -> internal model key
BASELINE_MODELS = [("MPTE", "transformer"), ("AR", "ar"), ("MIDAS", "midas")]
# ablation folder prefix -> table label (matches ABLATION_RENAME in the original script)
ABLATIONS = [
    ("synth_B1_no_nonlinearity", "AB1"),
    ("synth_B2_no_attention", "AB2"),
    ("synth_B3_no_attention_no_nonlinearity", "AB3"),
    ("synth_B5_no_positional_encoding", "AB4"),
    ("synth_B6_y_only", "AB5"),
]
ROW_ORDER = ["MPTE", "AR", "MIDAS", "AB1", "AB2", "AB3", "AB4", "AB5"]
METRICS = ["RMSE", "MAE", "DA"]

PRED_PREFIXES = {"transformer": "transformer_preds", "midas": "midas_preds", "ar": "ar_preds"}

# Numbers currently in main.tex (Tab:evals_simulation), for reconciliation.
# regime -> model -> (RMSE, MAE, DA)
PAPER_TABLE = {
    "Linear": {
        "MPTE": (1.2990, 1.0231, 0.6355), "AR": (1.3832, 1.0930, 0.3283),
        "MIDAS": (1.2631, 0.9733, 0.6807), "AB1": (1.2872, 1.0099, 0.6476),
        "AB2": (1.2967, 1.0312, 0.6506), "AB3": (1.2679, 0.9933, 0.6747),
        "AB4": (1.3634, 1.0707, 0.6687), "AB5": (1.3488, 1.0415, 0.6596),
    },
    "Mildly Nonlinear": {
        "MPTE": (1.3995, 1.1134, 0.6777), "AR": (1.7798, 1.4361, 0.1235),
        "MIDAS": (1.5162, 1.2112, 0.6355), "AB1": (1.4063, 1.1268, 0.6657),
        "AB2": (1.4261, 1.1472, 0.6596), "AB3": (1.4365, 1.1422, 0.6566),
        "AB4": (1.7394, 1.4103, 0.5843), "AB5": (1.4149, 1.1414, 0.6536),
    },
    "Highly Nonlinear": {
        "MPTE": (1.1579, 0.9345, 0.5964), "AR": (1.2906, 1.0234, 0.0301),
        "MIDAS": (1.2857, 1.0527, 0.5873), "AB1": (1.2157, 0.9773, 0.5392),
        "AB2": (1.3084, 1.0407, 0.5090), "AB3": (1.1965, 0.9606, 0.5693),
        "AB4": (1.3066, 1.0413, 0.5693), "AB5": (1.2018, 0.9706, 0.5994),
    },
}


# --------------------------------------------------------------------------------------
# Metric helpers (identical definitions to aggregate_univariate_results_synth.py)
# --------------------------------------------------------------------------------------
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))


def compute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "DA": directional_accuracy(y_true, y_pred),
    }


def load_folder_metrics(folder: Path) -> dict:
    """Replicate the per-folder merge in the original aggregator and return
    {model_key -> metrics dict} for whichever of transformer/ar/midas are present."""
    pred_dfs = {}
    for model, prefix in PRED_PREFIXES.items():
        file = next(folder.glob(f"{prefix}_*.csv"), None)
        if file is None:
            continue
        df = pd.read_csv(file)
        df = df.rename(columns={df.columns[0]: TIME_COLUMN, df.columns[1]: "true", df.columns[2]: model})
        pred_dfs[model] = df.set_index(TIME_COLUMN)

    if not pred_dfs:
        return {}

    keys = list(pred_dfs)
    merged = pred_dfs[keys[0]].copy()
    for k in keys[1:]:
        merged = merged.join(pred_dfs[k].drop(columns="true", errors="ignore"), how="inner")

    # Drop rows with NaN (a diverged training run can emit NaN preds); skip the folder entirely
    # if nothing valid remains, so one bad replication seed cannot break the aggregation.
    merged = merged.dropna(subset=["true"] + keys)
    if len(merged) == 0:
        return {}

    y_true = merged["true"].values
    return {model: compute_errors(y_true, merged[model].values) for model in keys}


# --------------------------------------------------------------------------------------
# Build the table
# --------------------------------------------------------------------------------------
def build() -> tuple[dict, list]:
    """Returns (computed[regime][label] -> metrics dict, list of missing-folder warnings)."""
    computed: dict = {}
    warnings: list = []

    for regime, tag, date in REGIMES:
        computed[regime] = {}

        base = EXPERIMENT_DIR / f"synth_mixed_frequency_transformer_{tag}_{date}"
        base_metrics = load_folder_metrics(base) if base.is_dir() else {}
        if not base_metrics:
            warnings.append(f"[{regime}] missing baseline folder/preds: {base.name}")
        for label, key in BASELINE_MODELS:
            if key in base_metrics:
                computed[regime][label] = base_metrics[key]

        for prefix, label in ABLATIONS:
            folder = EXPERIMENT_DIR / f"{prefix}_{tag}_{date}"
            if not folder.is_dir():
                warnings.append(f"[{regime}] missing ablation folder: {folder.name}")
                continue
            m = load_folder_metrics(folder)
            if "transformer" in m:
                computed[regime][label] = m["transformer"]
            else:
                warnings.append(f"[{regime}] no transformer preds in {folder.name}")

    return computed, warnings


def reconcile(computed: dict) -> pd.DataFrame:
    rows = []
    for regime in PAPER_TABLE:
        for label in ROW_ORDER:
            paper = PAPER_TABLE[regime].get(label)
            comp = computed.get(regime, {}).get(label)
            for i, metric in enumerate(METRICS):
                p = paper[i] if paper else np.nan
                c = comp[metric] if comp else np.nan
                diff = abs(p - c) if (paper and comp) else np.nan
                rows.append({
                    "regime": regime, "model": label, "metric": metric,
                    "paper": p, "computed": round(c, 4) if comp else np.nan,
                    "abs_diff": round(diff, 6) if not np.isnan(diff) else np.nan,
                    "status": "MISSING" if (np.isnan(diff)) else ("PASS" if diff < RECONCILE_TOL else "FAIL"),
                })
    return pd.DataFrame(rows)


def highlight_map(computed: dict) -> dict:
    """For each (regime, metric) find best and second across all models present.
    Returns {(regime, metric, label) -> 'best'|'second'}."""
    hl = {}
    for regime, _, _ in REGIMES:
        for metric in METRICS:
            vals = [(label, computed[regime][label][metric])
                    for label in ROW_ORDER if label in computed.get(regime, {})]
            if len(vals) < 2:
                continue
            reverse = (metric == "DA")  # DA: higher is better
            ordered = sorted(vals, key=lambda x: x[1], reverse=reverse)
            hl[(regime, metric, ordered[0][0])] = "best"
            hl[(regime, metric, ordered[1][0])] = "second"
    return hl


def fmt(value: float, regime: str, metric: str, label: str, hl: dict) -> str:
    s = f"{value:.4f}"
    tag = hl.get((regime, metric, label))
    if tag == "best":
        return f"\\best{{{s}}}"
    if tag == "second":
        return f"\\second{{{s}}}"
    return s


def make_latex(computed: dict, hl: dict) -> str:
    L = []
    L.append("% Auto-generated by src/evaluation/build_simulation_table.py -- do not edit by hand.")
    L.append("\\begin{table}[h!]")
    L.append("\\centering")
    L.append("\\begin{tabular}{lccccccccc}")
    L.append("\\toprule")
    L.append("& \\multicolumn{3}{c}{\\textbf{Linear}}")
    L.append("& \\multicolumn{3}{c}{\\textbf{Mildly Nonlinear}}")
    L.append("& \\multicolumn{3}{c}{\\textbf{Highly Nonlinear}} \\\\")
    L.append("\\midrule")
    L.append("& RMSE & MAE & DA & RMSE & MAE & DA & RMSE & MAE & DA \\\\")
    L.append("\\midrule")
    for label in ROW_ORDER:
        cells = []
        for regime, _, _ in REGIMES:
            m = computed.get(regime, {}).get(label)
            if not m:
                cells += ["--", "--", "--"]
            else:
                cells += [fmt(m[metric], regime, metric, label, hl) for metric in METRICS]
        L.append(f"{label}\n& " + " & ".join(cells) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\caption{Forecasting accuracy for the first low-frequency target series $Y_1$ "
             "across linear, mildly nonlinear, and highly nonlinear simulation designs. "
             "Dark green indicates the best-performing method and light green the second best.}")
    L.append("\\label{Tab:evals_simulation}")
    L.append("\\end{table}")
    return "\n".join(L)


def main() -> None:
    computed, warnings = build()
    for w in warnings:
        print("WARN:", w)

    recon = reconcile(computed)
    n_fail = int((recon["status"] == "FAIL").sum())
    n_missing = int((recon["status"] == "MISSING").sum())
    n_pass = int((recon["status"] == "PASS").sum())

    print("\n=== Reconciliation vs main.tex (Tab:evals_simulation) ===")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(recon.to_string(index=False))
    print(f"\nPASS={n_pass}  FAIL={n_fail}  MISSING={n_missing}  (tol={RECONCILE_TOL})")

    TABLE_DIR.mkdir(exist_ok=True)
    recon_path = TABLE_DIR / "synthetic_table1_reconciliation.csv"
    recon.to_csv(recon_path, index=False)
    print(f"Saved reconciliation -> {recon_path}")

    hl = highlight_map(computed)
    latex = make_latex(computed, hl)
    tex_path = TABLE_DIR / "synthetic_table1_combined.tex"
    tex_path.write_text(latex + "\n")
    print(f"Saved combined LaTeX table -> {tex_path}")


if __name__ == "__main__":
    main()
