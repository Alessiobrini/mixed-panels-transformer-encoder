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
        quarterly_vars = [config.features.target]
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
