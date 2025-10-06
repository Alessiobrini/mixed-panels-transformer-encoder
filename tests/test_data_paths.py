import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.data_paths import (
    is_simulation_enabled,
    resolve_target_variable,
    use_quarterly_only_predictors,
)


def build_config(simulation=None, features=None):
    sim_ns = SimpleNamespace(**simulation) if simulation else None
    feat_defaults = {
        "monthly_vars": [],
        "quarterly_vars": [],
        "target": None,
        "all_monthly": False,
    }
    if features:
        feat_defaults.update(features)
    features_ns = SimpleNamespace(**feat_defaults)
    return SimpleNamespace(simulation=sim_ns, features=features_ns)


def test_is_simulation_enabled_false_without_block():
    config = build_config(simulation=None)
    assert is_simulation_enabled(config) is False


def test_use_quarterly_only_predictors_disabled_when_simulation_off():
    config = build_config(
        simulation={"simulate": False, "use_y_only_predictors": True},
        features={
            "monthly_vars": ["RPI"],
            "quarterly_vars": ["GDP"],
            "target": "GDP",
        },
    )
    assert use_quarterly_only_predictors(config) is False


def test_use_quarterly_only_predictors_respects_explicit_flag():
    config = build_config(
        simulation={
            "simulate": True,
            "use_y_only_predictors": True,
            "p_x": 5,
            "p_y": 2,
        },
    )
    assert use_quarterly_only_predictors(config) is True


def test_use_quarterly_only_predictors_falls_back_to_monthly_count():
    config = build_config(
        simulation={
            "simulate": True,
            "p_x": 0,
            "p_y": 1,
        },
        features={"target": "Y1"},
    )
    assert use_quarterly_only_predictors(config) is True


def test_use_quarterly_only_predictors_false_when_monthly_present():
    config = build_config(
        simulation={
            "simulate": True,
            "p_x": 2,
            "p_y": 1,
        },
        features={
            "monthly_vars": ["X1", "X2"],
            "quarterly_vars": ["Y1"],
            "target": "Y1",
        },
    )
    assert use_quarterly_only_predictors(config) is False


def test_resolve_target_variable_real_data():
    config = build_config(
        simulation={"simulate": False},
        features={"target": "GDP"},
    )
    assert resolve_target_variable(config) == "GDP"


def test_resolve_target_variable_simulation_explicit():
    config = build_config(
        simulation={"simulate": True, "target": "Y3"},
        features={"target": "GDP"},
    )
    assert resolve_target_variable(config) == "Y3"


def test_resolve_target_variable_simulation_fallback_to_features():
    config = build_config(
        simulation={"simulate": True},
        features={"target": "Y2"},
    )
    assert resolve_target_variable(config) == "Y2"


def test_resolve_target_variable_simulation_default():
    config = build_config(
        simulation={"simulate": True},
        features={"target": None},
    )
    assert resolve_target_variable(config) == "Y1"
