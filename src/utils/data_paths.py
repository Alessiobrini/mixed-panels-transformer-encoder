"""Utilities for resolving data- and output-related paths from the config.

These helpers centralize the logic for deciding whether the pipeline should
operate on real FRED data or on simulated data produced by
``simulate_to_long.py``.  Every stage (training, AR baseline, MIDAS, etc.)
relies on the same suffix-based naming convention, so we expose a couple of
small functions that return the variable lists, suffix, processed-data path,
and output paths derived from the active configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def is_simulation_enabled(config) -> bool:
    """Return ``True`` if the simulation branch is enabled in the config."""

    sim_cfg = getattr(config, "simulation", None)
    if sim_cfg is None:
        return False
    return bool(getattr(sim_cfg, "simulate", False))


def use_quarterly_only_predictors(config) -> bool:
    """Return ``True`` if simulation should ignore monthly predictors."""

    if not is_simulation_enabled(config):
        return False

    sim_cfg = getattr(config, "simulation", None)
    if sim_cfg is None:
        return False

    flag = getattr(sim_cfg, "use_y_only_predictors", None)
    if flag is not None:
        return bool(flag)

    n_monthly, _ = _resolve_sim_counts(config)
    return n_monthly == 0


def _coalesce(*values):
    for value in values:
        if value is None:
            continue
        return value
    return None


def _safe_len(value) -> int | None:
    try:
        return len(value) if value is not None else None
    except TypeError:
        return None


def _resolve_sim_counts(config) -> Tuple[int, int]:
    sim_cfg = getattr(config, "simulation", None)
    features = getattr(config, "features", None)

    fallback_monthly = _safe_len(getattr(features, "monthly_vars", None))
    fallback_quarterly = _safe_len(getattr(features, "quarterly_vars", None))

    n_monthly = _coalesce(
        getattr(sim_cfg, "p_x", None) if sim_cfg else None,
        getattr(sim_cfg, "num_monthly", None) if sim_cfg else None,
        fallback_monthly,
    )
    n_quarterly = _coalesce(
        getattr(sim_cfg, "p_y", None) if sim_cfg else None,
        getattr(sim_cfg, "num_quarterly", None) if sim_cfg else None,
        fallback_quarterly,
    )

    if n_monthly is None:
        n_monthly = 0
    if n_quarterly is None:
        # fall back to at least the target series if defined
        n_quarterly = 1 if getattr(features, "target", None) else 0

    return int(n_monthly), int(n_quarterly)


def resolve_variable_lists(config, project_root: Path) -> Tuple[List[str], List[str]]:
    """Return ordered monthly and quarterly variable lists for the active run."""

    if is_simulation_enabled(config):
        n_monthly, n_quarterly = _resolve_sim_counts(config)
        monthly_vars = [f"X{i + 1}" for i in range(n_monthly)]
        quarterly_vars = [f"Y{j + 1}" for j in range(n_quarterly)]
        return monthly_vars, quarterly_vars

    raw_md_path = project_root / config.paths.data_raw_fred_monthly
    md_cols = pd.read_csv(raw_md_path, nrows=0).columns.tolist()

    if getattr(config.features, "all_monthly", False):
        monthly_vars = [c for c in md_cols if c != "date"]
        quarterly_vars = [resolve_target_variable(config)]
    else:
        monthly_vars = list(config.features.monthly_vars)
        quarterly_vars = list(config.features.quarterly_vars)

    return monthly_vars, quarterly_vars


def build_suffix(monthly_vars: List[str], quarterly_vars: List[str]) -> str:
    return f"{len(monthly_vars)}M_{len(quarterly_vars)}Q"


def resolve_data_paths(config, project_root: Path) -> Tuple[Path, str, List[str], List[str]]:
    """Return the processed-data path, suffix, and variable lists."""

    monthly_vars, quarterly_vars = resolve_variable_lists(config, project_root)
    suffix = build_suffix(monthly_vars, quarterly_vars)
    template_attr = (
        "data_processed_template_simulation"
        if is_simulation_enabled(config)
        else "data_processed_template"
    )
    template = getattr(config.paths, template_attr)
    data_path = project_root / template.format(suffix=suffix)
    return data_path, suffix, monthly_vars, quarterly_vars


def get_output_path(config, project_root: Path, output_key: str, suffix: str) -> Path:
    """Return the full path for an output artifact (predictions, etc.)."""

    template = getattr(config.paths.outputs, output_key)
    return project_root / template.format(suffix=suffix)


def resolve_target_variable(config) -> str:
    """Return the name of the target variable for the active configuration."""

    features = getattr(config, "features", None)
    feature_target = getattr(features, "target", None) if features else None

    if is_simulation_enabled(config):
        sim_cfg = getattr(config, "simulation", None)
        target = _coalesce(
            getattr(sim_cfg, "target", None) if sim_cfg else None,
            feature_target,
            "Y1",
        )
        return target

    if feature_target is None:
        raise ValueError("Real-data runs require `features.target` to be set in the config.")

    return feature_target


# ---------------------------------------------------------------------------
# Equity-mode helpers
# ---------------------------------------------------------------------------

def is_equity_mode(config) -> bool:
    """Return ``True`` if the config specifies an equity experiment."""
    equity_cfg = getattr(config, "equity", None)
    if equity_cfg is None:
        return False
    return bool(getattr(equity_cfg, "active_ticker", None))


def resolve_equity_data_paths(
    config, project_root: Path
) -> Tuple[Path, str, str]:
    """Return ``(csv_path, suffix, target_variable)`` for the active equity ticker."""
    ticker = config.equity.active_ticker
    suffix = getattr(config.equity, "suffix", "7D_43M_14Q")
    template = config.paths.data_processed_equity_template
    csv_path = project_root / template.format(ticker=ticker, suffix=suffix)
    target = config.equity.target_template.replace("{TKR}", ticker)
    return csv_path, suffix, target


def resolve_all_equity_csv_paths(
    config, project_root: Path
) -> dict:
    """Return ``{ticker: csv_path}`` for every ticker in the universe."""
    tickers = resolve_equity_tickers(config)
    suffix = getattr(config.equity, "suffix", "7D_43M_14Q")
    template = config.paths.data_processed_equity_template
    return {t: project_root / template.format(ticker=t, suffix=suffix) for t in tickers}


def resolve_equity_tickers(config) -> List[str]:
    """Return the full ticker universe from universe_file or config.equity.tickers."""
    universe_file = getattr(config.equity, "universe_file", None)
    if universe_file:
        _project_root = Path(__file__).resolve().parents[2]
        universe_path = _project_root / universe_file
        if universe_path.exists():
            return pd.read_csv(universe_path)["ticker"].tolist()
    return list(config.equity.tickers)
