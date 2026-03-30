"""Pre-build the concatenated cross-sectional dataset and save to disk.

This avoids the ~5-minute CSV loading overhead each time the pipeline runs.
The cache is a single .pt file containing all materialized samples and
per-ticker metadata needed for training and evaluation.

Usage:
    python src/data/build_concat_cache.py
    python src/data/build_concat_cache.py --config src/config/cfg_equity.yaml
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.mixed_frequency_dataset import MixedFrequencyDataset
from src.utils.config import Config
from src.utils.data_paths import resolve_all_equity_csv_paths


def build_concat_cache(config, project_root):
    """Build and return the cache dict for the concatenated dataset."""
    ticker_csv = resolve_all_equity_csv_paths(config, project_root)
    target_template = config.equity.target_template
    context_days = config.data.context_days
    train_ratio = config.data.train_ratio

    # Filter to existing CSVs
    ticker_csv = {t: p for t, p in ticker_csv.items() if p.exists()}
    print(f"Building cache for {len(ticker_csv)} tickers...")

    # Build unified var/freq maps from first ticker
    first_ticker = next(iter(ticker_csv))
    first_csv = ticker_csv[first_ticker]
    first_target = target_template.replace("{TKR}", first_ticker)
    ref_ds = MixedFrequencyDataset(
        first_csv, context_days=context_days,
        target_variable=first_target, ticker=first_ticker,
    )
    unified_var_map = dict(ref_ds.var_map)
    unified_freq_map = dict(ref_ds.freq_map)

    # Materialize all samples
    all_samples = []       # list of dicts with tensors
    per_ticker_info = []
    global_offset = 0

    for tkr, csv_path in tqdm(ticker_csv.items(), desc="Loading tickers"):
        target_var = target_template.replace("{TKR}", tkr)
        try:
            ds = MixedFrequencyDataset(
                csv_path, context_days=context_days,
                target_variable=target_var, ticker=tkr,
            )
        except Exception as e:
            print(f"  SKIP {tkr}: {e}")
            continue
        ds.set_maps(unified_var_map, unified_freq_map)

        n = len(ds)
        if n == 0:
            continue

        n_train = int(train_ratio * n)
        local_train = list(range(n_train))
        local_test = list(range(n_train, n))

        ds.fit_scalers_from_train_items(local_train)

        # Materialize all items as tensors
        samples = [ds[i] for i in range(n)]
        all_samples.extend(samples)

        # Extract test dates for evaluation
        mask = ds.df['Variable'] == ds.target_variable
        timestamps = ds.df[mask]['Timestamp'].reset_index(drop=True)
        test_offset = local_test[0] + ds.skipped_context if local_test else 0
        test_dates = timestamps.iloc[test_offset:test_offset + len(local_test)].tolist() if local_test else []

        per_ticker_info.append({
            "ticker": tkr,
            "local_train": local_train,
            "local_test": local_test,
            "global_train": [i + global_offset for i in local_train],
            "global_test": [i + global_offset for i in local_test],
            "target_scaler_mean": float(ds.scaler.mean_[0]),
            "target_scaler_scale": float(ds.scaler.scale_[0]),
            "test_dates": test_dates,
        })
        global_offset += n

    cache = {
        "samples": all_samples,
        "per_ticker_info": per_ticker_info,
        "var_map": unified_var_map,
        "freq_map": unified_freq_map,
        "n_total": len(all_samples),
    }

    print(f"Cache built: {len(all_samples)} samples from "
          f"{len(per_ticker_info)} tickers")
    return cache


def build_baseline_cache(config, project_root):
    """Build wide quarterly DataFrames for all tickers (used by OLS/XGB/NN).

    Returns ``{ticker: wide_df}`` where each DataFrame has ~57 columns
    (14 quarterly + 43 monthly collapsed to quarterly).
    """
    from src.run_equity_pipeline import _build_quarterly_wide

    ticker_csv = resolve_all_equity_csv_paths(config, project_root)
    ticker_csv = {t: p for t, p in ticker_csv.items() if p.exists()}
    target_template = config.equity.target_template

    cache = {}
    for tkr, csv_path in tqdm(ticker_csv.items(), desc="Building baseline cache"):
        target_var = target_template.replace("{TKR}", tkr)
        try:
            wide = _build_quarterly_wide(csv_path, target_var)
            if len(wide) >= 4:
                cache[tkr] = wide
        except Exception as e:
            print(f"  SKIP {tkr}: {e}")

    print(f"Baseline cache: {len(cache)} tickers")
    return cache


def main():
    parser = argparse.ArgumentParser(description="Build concatenated dataset cache")
    parser.add_argument("--config", default="src/config/cfg_equity.yaml")
    parser.add_argument("--baseline", action="store_true",
                        help="Build baseline wide-data cache (for OLS/XGB/NN)")
    args = parser.parse_args()

    cfg_path = project_root / args.config
    config = Config(cfg_path)
    suffix = getattr(config.equity, "suffix", "7D_43M_14Q")

    if args.baseline:
        cache = build_baseline_cache(config, project_root)
        cache_path = project_root / "data" / "processed" / f"baseline_wide_cache_{suffix}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(cache, open(cache_path, "wb"))
        print(f"Saved baseline cache to {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")
        return

    cache = build_concat_cache(config, project_root)

    cache_path = project_root / "data" / "processed" / f"concat_cache_{suffix}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(cache, cache_path)
    print(f"Saved cache to {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
