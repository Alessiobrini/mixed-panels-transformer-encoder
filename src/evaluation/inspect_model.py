"""Reload a trained Mixed-Frequency Transformer from an experiment folder."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.train import build_model, prepare_data  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.data_paths import resolve_data_paths  # noqa: E402


def _default_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required path does not exist: {path}")
    return path


def _load_overrides(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    overrides: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        overrides[key] = value
    return overrides


def _coerce_override(original: Any, value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "none":
            return None
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if any(ch in lowered for ch in [".", "e"]):
                return float(value)
            return int(value)
        except ValueError:
            pass
    if original is None:
        return value
    if isinstance(original, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)
    if isinstance(original, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return original
    if isinstance(original, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return original
    try:
        return type(original)(value)
    except Exception:  # pragma: no cover - defensive
        return value


def _apply_transformer_overrides(config: Config, overrides: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = getattr(config, "model", None)
    transformer_cfg = getattr(model_cfg, "transformer", None) if model_cfg else None
    if transformer_cfg is None:
        return {}
    applied: Dict[str, Any] = {}
    for key, value in overrides.items():
        if hasattr(transformer_cfg, key):
            original = getattr(transformer_cfg, key)
            coerced = _coerce_override(original, value)
            setattr(transformer_cfg, key, coerced)
            applied[key] = coerced
    return applied


def _as_int(value: Any, name: str) -> int:
    if value in (None, "None"):
        raise ValueError(f"Missing integer config value for '{name}'.")
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(float(value))


def _as_float(value: Any, name: str) -> float:
    if value in (None, "None"):
        raise ValueError(f"Missing float config value for '{name}'.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not convert '{name}' to float: {value}") from exc


def _as_optional_int(value: Any) -> Optional[int]:
    if value in (None, "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(float(value))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _resolve_experiment_dir(experiment_value: Any) -> Path:
    if experiment_value is None:
        raise ValueError("Config must define 'evaluation.experiment'.")
    path = Path(str(experiment_value))
    if not path.is_absolute():
        path = PROJECT_ROOT / "outputs" / "experiments" / path
    return _require(path.resolve())


def _load_checkpoint(directory: Path) -> Path:
    checkpoints = sorted(directory.glob("best_model_trial_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matching 'best_model_trial_*.pt' in {directory}."
        )
    if len(checkpoints) == 1:
        return checkpoints[0]
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def _prepare_training_artifacts(config: Config):
    csv_path, _, _, _ = resolve_data_paths(config, PROJECT_ROOT)
    data = prepare_data(csv_path, config)
    if config.training.optimize:
        full_dataset, train_loader, *_ = data
    else:
        full_dataset, train_loader, *_ = data
    return full_dataset, train_loader


def reload_from_config(
    config_path: Optional[str] = None, device: Optional[str] = None
) -> Tuple[torch.nn.Module, Config, Path, Dict[str, Any]]:
    """Reload a trained model specified by ``evaluation.experiment``."""

    cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "src" / "config" / "cfg.yaml"
    cfg_path = _require(cfg_path.resolve())

    root_config = Config(cfg_path)
    evaluation_cfg = getattr(root_config, "evaluation", None)
    experiment_dir = _resolve_experiment_dir(getattr(evaluation_cfg, "experiment", None))

    used_config_path = _require(experiment_dir / "used_config.yaml")
    params_path = experiment_dir / "full_final_params.yaml"
    checkpoint_path = _load_checkpoint(experiment_dir)

    run_config = Config(used_config_path)
    overrides = _load_overrides(params_path)
    applied_overrides = _apply_transformer_overrides(run_config, overrides)

    transformer_cfg = getattr(getattr(run_config, "model", None), "transformer", None)
    if transformer_cfg is None:
        raise ValueError("used_config.yaml must define model.transformer settings.")

    d_model = _as_int(getattr(transformer_cfg, "d_model", None), "model.transformer.d_model")
    nhead = _as_int(getattr(transformer_cfg, "nhead", None), "model.transformer.nhead")
    num_layers = _as_int(getattr(transformer_cfg, "num_layers", None), "model.transformer.num_layers")
    dropout = _as_float(getattr(transformer_cfg, "dropout", None), "model.transformer.dropout")
    d_freq = _as_optional_int(getattr(transformer_cfg, "d_freq", None))
    d_var = _as_optional_int(getattr(transformer_cfg, "d_var", None))
    dim_feedforward = _as_int(
        getattr(transformer_cfg, "dim_feedforward", 2048), "model.transformer.dim_feedforward"
    )
    activation = str(getattr(transformer_cfg, "activation", "relu"))
    use_nonlinearity = _as_bool(getattr(transformer_cfg, "use_nonlinearity", True))
    use_attention = _as_bool(getattr(transformer_cfg, "use_attention", True))

    full_dataset, train_loader = _prepare_training_artifacts(run_config)

    model = build_model(
        full_dataset,
        run_config,
        d_model,
        nhead,
        num_layers,
        dropout,
        train_loader=train_loader,
        d_freq=d_freq,
        d_var=d_var,
        dim_feedforward=dim_feedforward,
        activation=activation,
        use_nonlinearity=use_nonlinearity,
        use_attention=use_attention,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a state_dict mapping: {checkpoint_path}")
    model.load_state_dict(state_dict)

    target_device = _default_device(device)
    model.to(target_device)
    model.eval()

    meta = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(cfg_path),
        "used_config": str(used_config_path),
        "params_path": str(params_path) if params_path.exists() else None,
        "applied_overrides": applied_overrides,
        "device": str(target_device),
    }

    return model, run_config, checkpoint_path, meta


if __name__ == "__main__":
    model, _, checkpoint, meta = reload_from_config()
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {meta['device']}")
