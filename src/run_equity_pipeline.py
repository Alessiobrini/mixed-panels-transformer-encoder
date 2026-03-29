"""
Equity experiment pipeline.

Runs MPTE + all baselines (AR, MIDAS, OLS, XGBoost, NN) per ticker,
then evaluates with RMSE and Diebold-Mariano tests.

Usage:
    python src/run_equity_pipeline.py                          # all 40 tickers
    python src/run_equity_pipeline.py --ticker AAPL             # single ticker
    python src/run_equity_pipeline.py --config path/to.yaml     # custom config
"""

import argparse
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.data_paths import (
    resolve_all_equity_csv_paths,
    resolve_equity_data_paths,
    resolve_equity_tickers,
)
from src.models.ar import run_ar_baseline, run_concatenated_ar_baseline
from src.models.single_freq_models import (
    run_single_freq_baselines,
    run_concatenated_single_freq_baselines,
)
from src.train import (
    run_standard_training,
    run_concatenated_training,
    run_optuna,
)
from src.evaluation.evaluate_forecasts import (
    load_forecasts,
    compute_rmse,
    run_dm_tests,
)

if platform.system() == "Windows":
    RSCRIPT = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
else:
    RSCRIPT = "/hpc/group/darec/ab978/miniconda3/envs/tsa-dev/bin/Rscript"

MIDAS_SCRIPT = project_root / "src" / "models" / "midas.R"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolve_midas_monthly_vars(config, ticker: str) -> list[str]:
    """Expand {TKR} placeholders in midas_monthly_vars_template."""
    templates = list(config.equity.midas_monthly_vars_template)
    return [v.replace("{TKR}", ticker) for v in templates]


def _write_midas_data_and_config(
    config, ticker: str, csv_path: Path, target_var: str,
    suffix: str, exp_path: Path,
) -> Path:
    """Write a MIDAS-ready long-format CSV and temp config for the R script.

    The R MIDAS script expects quarterly timestamps on regular quarter
    boundaries (via ``as.yearqtr``). EPS report dates are asynchronous,
    so we floor them to the quarter-start before writing.
    """
    midas_vars = _resolve_midas_monthly_vars(config, ticker)
    quarterly_vars = list(config.features.quarterly_vars) + [target_var]
    seen = set()
    quarterly_vars = [v for v in quarterly_vars if not (v in seen or seen.add(v))]
    all_vars = set(midas_vars) | set(quarterly_vars)

    # Read the full long-format CSV and filter to MIDAS-relevant vars
    full = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    midas_df = full[full["Variable"].isin(all_vars)].copy()

    # Align timestamps to regular calendar boundaries so R's ts() works:
    # - Monthly: floor to month-start (e.g., 2015-01-30 → 2015-01-01)
    # - Quarterly: floor to quarter-start (e.g., 2015-04-27 → 2015-04-01)
    m_mask = midas_df["Frequency"] == "M"
    midas_df.loc[m_mask, "Timestamp"] = (
        midas_df.loc[m_mask, "Timestamp"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    q_mask = midas_df["Frequency"] == "Q"
    midas_df.loc[q_mask, "Timestamp"] = (
        midas_df.loc[q_mask, "Timestamp"]
        .dt.to_period("Q")
        .dt.to_timestamp()
    )


    # Write temp CSV
    tmp_csv = Path(tempfile.mktemp(suffix=".csv", prefix=f"midas_data_{ticker}_"))
    midas_df.to_csv(tmp_csv, index=False)

    # Build temp YAML config
    tmp_csv_path = str(tmp_csv).replace("\\", "/")
    output_path = str(exp_path / f"midas_preds_{suffix}.csv").replace("\\", "/")

    temp_cfg = {
        "simulation": {"simulate": False},
        "features": {
            "target": target_var,
            "monthly_vars": midas_vars,
            "quarterly_vars": quarterly_vars,
        },
        "paths": {
            "data_processed_template": tmp_csv_path,
            "outputs": {"midas_preds": output_path},
        },
        "data": {"train_ratio": config.data.train_ratio},
        "model": {
            "midas": {
                "lags": list(config.model.midas.lags),
                "ar_lags": config.model.midas.ar_lags,
                "use_y_lags": config.model.midas.use_y_lags,
            }
        },
        "training": {"experiment_name": None},
    }

    tmp_yaml = Path(tempfile.mktemp(suffix=".yaml", prefix=f"cfg_midas_{ticker}_"))
    with open(tmp_yaml, "w") as f:
        yaml.dump(temp_cfg, f, default_flow_style=False)
    return tmp_yaml, tmp_csv


def _build_quarterly_wide(csv_path: Path, target_var: str) -> pd.DataFrame:
    """Pivot long-format CSV quarterly rows into wide format for single-freq baselines.

    EPS report dates and FRED macro quarterly dates don't align, so we
    forward-fill the pivot table and keep only rows where the target is
    observed. This ensures macro values are the most recent available at
    each EPS report date (no look-ahead bias).
    """
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    q = df[df["Frequency"] == "Q"].copy()
    wide = q.pivot_table(index="Timestamp", columns="Variable", values="Value")
    wide = wide.sort_index().ffill()
    # Keep only rows where the target variable is non-null
    wide = wide.dropna(subset=[target_var])
    wide.index.name = "date"
    return wide


# ---------------------------------------------------------------------------
# Per-ticker experiment
# ---------------------------------------------------------------------------
def run_ticker(config, ticker: str, today: str, cfg_path: Path):
    """Run all models + evaluation for one ticker."""
    config.equity.active_ticker = ticker
    csv_path, suffix, target_var = resolve_equity_data_paths(config, project_root)
    config.features.target = target_var

    exp_name = f"{ticker}_{today}"
    config.training.experiment_name = exp_name
    exp_path = project_root / "outputs" / "experiments" / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility
    shutil.copy(cfg_path, exp_path / "used_config.yaml")

    print(f"\n{'=' * 60}")
    print(f"  Ticker: {ticker}  |  Target: {target_var}  |  Exp: {exp_name}")
    print(f"{'=' * 60}")

    # --- 1. MPTE Transformer ---
    print("\n--- MPTE Transformer ---")
    try:
        if config.training.optimize:
            run_optuna(config, csv_path, exp_path, suffix)
        else:
            run_standard_training(config, csv_path, exp_path, suffix)
    except Exception as e:
        print(f"  ERROR in transformer training: {e}")

    # --- 2. AR Baseline ---
    print("\n--- AR Baseline ---")
    ar_output = exp_path / f"ar_preds_{suffix}.csv"
    try:
        run_ar_baseline(
            csv_path, target_var,
            config.data.train_ratio,
            config.model.ar.max_lag,
            ar_output,
        )
    except Exception as e:
        print(f"  ERROR in AR: {e}")

    # --- 3. MIDAS Baseline ---
    print("\n--- MIDAS Baseline ---")
    tmp_csv = None
    try:
        tmp_cfg, tmp_csv = _write_midas_data_and_config(
            config, ticker, csv_path, target_var, suffix, exp_path
        )
        result = subprocess.run(
            [RSCRIPT, str(MIDAS_SCRIPT), str(tmp_cfg)],
            check=True, text=True, capture_output=True,
            cwd=str(project_root),
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        tmp_cfg.unlink(missing_ok=True)
        if tmp_csv:
            tmp_csv.unlink(missing_ok=True)
    except FileNotFoundError:
        print(f"  SKIP MIDAS: Rscript not found at {RSCRIPT}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR in MIDAS:\n{e.stderr[-500:]}")
        if tmp_csv:
            tmp_csv.unlink(missing_ok=True)

    # --- 4. Single-freq baselines (OLS, XGBoost, NN) ---
    print("\n--- Single-freq baselines (OLS, XGB, NN) ---")
    try:
        wide_q = _build_quarterly_wide(csv_path, target_var)
        sf_cfg = getattr(config, "single_freq", None)
        n_lags = getattr(sf_cfg, "n_lags", 2) if sf_cfg else 2
        sf_optimize = getattr(sf_cfg, "optimize", True) if sf_cfg else True
        sf_val = getattr(sf_cfg, "val_ratio", 0.1) if sf_cfg else 0.1

        run_single_freq_baselines(
            target=target_var,
            data=wide_q,
            train_ratio=config.data.train_ratio,
            exp_path=exp_path,
            suffix=suffix,
            n_lags=n_lags,
            optimize=sf_optimize,
            val_ratio=sf_val,
        )
    except Exception as e:
        print(f"  ERROR in single-freq baselines: {e}")

    # --- 5. Evaluation ---
    print("\n--- Evaluation ---")
    try:
        _evaluate_ticker(config, exp_path, suffix)
    except Exception as e:
        print(f"  ERROR in evaluation: {e}")


def _evaluate_ticker(config, exp_path: Path, suffix: str):
    """Load all available prediction CSVs and compute RMSE + DM tests.

    Different models produce different date formats for the same quarters
    (e.g., report dates vs quarter-start). We normalize all dates to
    quarter period labels before joining so the evaluation aligns correctly.
    """
    true_col = config.evaluation.true_col
    pred_col = config.evaluation.pred_col

    forecast_keys = list(config.evaluation.forecast_files)
    dfs = []
    labels = []
    for key in forecast_keys:
        fname = f"{key}_{suffix}.csv"
        fpath = exp_path / fname
        if not fpath.exists():
            print(f"  SKIP {key}: {fname} not found")
            continue
        df = pd.read_csv(fpath)
        # Trim first row for AR/MIDAS (off-by-one from lagged prediction)
        if "ar" in key or "midas" in key:
            df = df.iloc[1:].reset_index(drop=True)
        # Normalize date to quarter label for alignment
        df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q")
        df = df.drop_duplicates(subset=["quarter"], keep="last")
        df = df.set_index("quarter")
        dfs.append(df[[true_col, pred_col]].rename(columns={pred_col: key}))
        labels.append(key)

    if len(dfs) < 2:
        print("  Not enough forecasts for evaluation")
        return

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df.drop(columns=true_col, errors="ignore"), how="inner")

    true = merged[true_col]
    preds = merged.drop(columns=true_col)

    rmse_df = compute_rmse(preds, true)
    dm_results = run_dm_tests(preds, true)

    rmse_df.to_csv(exp_path / "rmse_summary.csv")
    dm_results.to_csv(exp_path / "dm_test_results.csv")

    print("  RMSE:")
    print(rmse_df.to_string())
    print(f"  Saved to {exp_path}")


# ---------------------------------------------------------------------------
# Concatenated (cross-sectional) experiment
# ---------------------------------------------------------------------------
def run_concatenated(config, today: str, cfg_path: Path):
    """Train one model on all stocks pooled, then evaluate per-ticker."""
    suffix = config.equity.suffix
    ticker_csv = resolve_all_equity_csv_paths(config, project_root)

    exp_name = f"concat_{today}"
    exp_path = project_root / "outputs" / "experiments" / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, exp_path / "used_config.yaml")

    print(f"\n{'=' * 60}")
    print(f"  Concatenated training | {len(ticker_csv)} stocks | Exp: {exp_name}")
    print(f"{'=' * 60}")

    # --- 1. MPTE Transformer ---
    print("\n--- MPTE Transformer (concatenated) ---")
    try:
        run_concatenated_training(config, project_root, exp_path, suffix)
    except Exception as e:
        print(f"  ERROR in concatenated transformer training: {e}")

    # --- 2. AR Baseline (pooled) ---
    print("\n--- AR Baseline (concatenated) ---")
    try:
        run_concatenated_ar_baseline(
            ticker_csv,
            config.equity.target_template,
            config.data.train_ratio,
            config.model.ar.max_lag,
            exp_path,
            suffix,
        )
    except Exception as e:
        print(f"  ERROR in concatenated AR: {e}")

    # --- 3. MIDAS --- skip in concatenated mode
    print("\n--- MIDAS: SKIPPED (not supported in concatenated mode) ---")

    # --- 4. Single-freq baselines (OLS, XGB, NN) ---
    print("\n--- Single-freq baselines (concatenated OLS, XGB, NN) ---")
    try:
        sf_cfg = getattr(config, "single_freq", None)
        n_lags = getattr(sf_cfg, "n_lags", 2) if sf_cfg else 2
        sf_optimize = getattr(sf_cfg, "optimize", True) if sf_cfg else True
        sf_val = getattr(sf_cfg, "val_ratio", 0.1) if sf_cfg else 0.1

        run_concatenated_single_freq_baselines(
            ticker_csv,
            config.equity.target_template,
            config.data.train_ratio,
            exp_path,
            suffix,
            n_lags=n_lags,
            optimize=sf_optimize,
            val_ratio=sf_val,
        )
    except Exception as e:
        print(f"  ERROR in concatenated single-freq baselines: {e}")

    # --- 5. Per-ticker evaluation ---
    print("\n--- Per-ticker evaluation ---")
    tickers_evaluated = 0
    for tkr in ticker_csv:
        ticker_exp = exp_path / tkr
        if not ticker_exp.exists():
            continue
        try:
            _evaluate_ticker(config, ticker_exp, suffix)
            tickers_evaluated += 1
        except Exception as e:
            print(f"  ERROR evaluating {tkr}: {e}")

    # --- 6. Aggregate RMSE across stocks ---
    print("\n--- Aggregate RMSE ---")
    rmse_rows = []
    for tkr in ticker_csv:
        rmse_path = exp_path / tkr / "rmse_summary.csv"
        if rmse_path.exists():
            df = pd.read_csv(rmse_path)
            df.insert(0, "ticker", tkr)
            rmse_rows.append(df)
    if rmse_rows:
        agg = pd.concat(rmse_rows, ignore_index=True)
        agg.to_csv(exp_path / "aggregate_rmse.csv", index=False)
        # Print mean RMSE per model
        print(agg.groupby(agg.columns[1]).mean(numeric_only=True).to_string())
        print(f"\n  Aggregate RMSE saved to {exp_path / 'aggregate_rmse.csv'}")

    print(f"\nConcatenated experiment complete: {tickers_evaluated} tickers evaluated.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Equity experiment pipeline")
    parser.add_argument("--config", default="src/config/cfg_equity.yaml")
    parser.add_argument("--ticker", default=None, help="Run single ticker")
    args = parser.parse_args()

    cfg_path = project_root / args.config
    config = Config(cfg_path)
    set_seeds(config.training.seed)
    today = datetime.now().strftime("%Y-%m-%d")

    training_mode = getattr(config.equity, "training_mode", "per_ticker")

    if training_mode == "concatenated" and not args.ticker:
        print(f"Equity pipeline: CONCATENATED mode, date={today}")
        run_concatenated(config, today, cfg_path)
    else:
        if args.ticker:
            tickers = [args.ticker]
        else:
            tickers = resolve_equity_tickers(config)
        print(f"Equity pipeline: {len(tickers)} ticker(s), date={today}")
        for ticker in tickers:
            run_ticker(config, ticker, today, cfg_path)
        print(f"\nAll {len(tickers)} ticker(s) completed.")


if __name__ == "__main__":
    main()
