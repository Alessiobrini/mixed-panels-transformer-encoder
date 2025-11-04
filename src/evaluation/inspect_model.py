"""Inspect trained Mixed-Frequency Transformer checkpoints.

This script reloads models trained via ``src/train.py`` and exposes utilities
for

* printing summary statistics of the stored weights, and
* capturing intermediate activations from selected submodules during a forward
  pass.

Only a single input – the experiment folder under ``outputs/experiments`` – is
required.  Each experiment directory must contain the following files emitted
by training:

* ``best_model_trial_*.pt`` – model weights to load,
* ``full_final_params.yaml`` – Optuna-optimised model hyper-parameters,
* ``used_config.yaml`` – the training configuration used to recreate the
  dataset and loaders.

Import ``inspect_experiment`` and call it directly, for example::

    from src.evaluation.inspect_model import inspect_experiment

    inspect_experiment(
        experiment="Test6",
        device="cuda",
        print_weights=True,
        capture=(
            "input_proj",
            "transformer_encoder.layers.0",
            "transformer_encoder.layers.-1",
            "prediction_head",
        ),
    )

Advanced options remain available via keyword arguments while paths are
validated eagerly to surface errors when misconfigured experiment folders are
provided.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "outputs" / "experiments"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_CAPTURE_MODULES = [
    "input_proj",
    "transformer_encoder.layers.0",
    "transformer_encoder.layers.-1",
    "prediction_head",
]

from src.train import build_model, prepare_data  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.data_paths import resolve_data_paths  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelParams:
    d_model: int
    nhead: int
    num_layers: int
    dropout: float
    d_freq: Optional[int]
    d_var: Optional[int]
    dim_feedforward: int
    activation: str
    use_nonlinearity: bool
    use_attention: bool

    @classmethod
    def from_config(
        cls,
        config: Config,
        overrides: Optional[Mapping[str, object]] = None,
    ) -> "ModelParams":
        transformer_cfg = getattr(config.model, "transformer", None)
        if transformer_cfg is None:
            raise ValueError("Config is missing 'model.transformer' settings.")

        params = dict(
            d_model=int(transformer_cfg.d_model),
            nhead=int(transformer_cfg.nhead),
            num_layers=int(transformer_cfg.num_layers),
            dropout=float(transformer_cfg.dropout),
            d_freq=(
                None
                if getattr(transformer_cfg, "d_freq", None) in (None, "None")
                else int(transformer_cfg.d_freq)
            ),
            d_var=(
                None
                if getattr(transformer_cfg, "d_var", None) in (None, "None")
                else int(transformer_cfg.d_var)
            ),
            dim_feedforward=int(getattr(transformer_cfg, "dim_feedforward", 2048)),
            activation=str(getattr(transformer_cfg, "activation", "relu")),
            use_nonlinearity=bool(getattr(transformer_cfg, "use_nonlinearity", True)),
            use_attention=bool(getattr(transformer_cfg, "use_attention", True)),
        )

        if overrides:
            for key, value in overrides.items():
                if key not in params:
                    continue
                params[key] = value

        return cls(**params)


@dataclass
class InspectionRequest:
    """Container describing what to inspect and how to do it."""

    experiment: str
    checkpoint: Path
    config_path: Path
    experiment_dir: Path
    params_path: Path
    device: Optional[str]
    print_weights: bool
    split: str
    batch_index: int
    capture: Sequence[str]
    run_forward: bool


def _load_params_override(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}

    with path.open("r") as fh:
        data = yaml.safe_load(fh) or {}

    # Ensure plain scalars (Optuna sometimes serialises single-element lists)
    cleaned: Dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        cleaned[key] = value
    return cleaned


def _resolve_experiment_dir(raw_value) -> Path:
    value = str(raw_value).strip() if raw_value is not None else ""
    if not value:
        raise ValueError("Experiment name or path must be provided.")

    path = Path(value)
    if not path.is_absolute():
        path = EXPERIMENTS_ROOT / path

    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Experiment directory '{resolved}' does not exist."
        )

    return resolved


def _resolve_required_file(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file '{filename}' inside experiment directory '{directory}'."
        )
    return path


def _locate_checkpoint(directory: Path) -> Path:
    candidates = sorted(directory.glob("best_model_trial_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint matching 'best_model_trial_*.pt' was found inside "
            f"'{directory}'."
        )

    if len(candidates) == 1:
        return candidates[0]

    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_inspection_request(
    experiment: str,
    *,
    device: Optional[str] = None,
    print_weights: bool = False,
    split: str = "test",
    batch_index: int = 0,
    capture: Optional[Sequence[str]] = None,
    run_forward: bool = True,
) -> InspectionRequest:
    """Construct an :class:`InspectionRequest` from simple keyword arguments."""

    experiment_dir = _resolve_experiment_dir(experiment)
    checkpoint = _locate_checkpoint(experiment_dir)
    config_path = _resolve_required_file(experiment_dir, "used_config.yaml")
    params_path = _resolve_required_file(experiment_dir, "full_final_params.yaml")

    capture_modules = list(capture) if capture is not None else list(DEFAULT_CAPTURE_MODULES)
    if not capture_modules:
        raise ValueError("At least one module must be supplied for activation capture.")

    return InspectionRequest(
        experiment=str(experiment),
        checkpoint=checkpoint,
        config_path=config_path,
        experiment_dir=experiment_dir,
        params_path=params_path,
        device=device,
        print_weights=bool(print_weights),
        split=str(split),
        batch_index=int(batch_index),
        capture=capture_modules,
        run_forward=bool(run_forward),
    )


def summarize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> List[str]:
    lines = []
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        stats = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "mean": tensor.float().mean().item(),
            "std": tensor.float().std(unbiased=False).item(),
            "min": tensor.float().min().item(),
            "max": tensor.float().max().item(),
        }
        lines.append(
            f"{name}: shape={stats['shape']} dtype={stats['dtype']} "
            f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f}"
        )
    return lines


def _register_hooks(model: torch.nn.Module, module_paths: Iterable[str]):
    activations: Dict[str, torch.Tensor] = {}
    hooks = []

    for path in module_paths:
        submodule = model.get_submodule(path)

        def _make_hook(name: str):
            def hook(_module, _inputs, output):
                if torch.is_tensor(output):
                    activations[name] = output.detach().cpu()
                elif isinstance(output, (list, tuple)):
                    tensors = [item.detach().cpu() for item in output if torch.is_tensor(item)]
                    if not tensors:
                        activations[name] = torch.tensor([])
                    elif len(tensors) == 1:
                        activations[name] = tensors[0]
                    else:
                        activations[name] = torch.stack(tensors)
                else:
                    activations[name] = torch.tensor([])

            return hook

        hooks.append(submodule.register_forward_hook(_make_hook(path)))

    return activations, hooks


def _cleanup_hooks(hooks: Iterable[torch.utils.hooks.RemovableHandle]):
    for hook in hooks:
        hook.remove()


def _prepare_dataset(config: Config):
    csv_path, suffix, _, _ = resolve_data_paths(config, PROJECT_ROOT)

    result = prepare_data(csv_path, config)
    if config.training.optimize:
        full_dataset, train_loader, val_loader, test_loader, test_indices = result
    else:
        full_dataset, train_loader, test_loader, test_indices = result
        val_loader = None

    return full_dataset, train_loader, val_loader, test_loader, test_indices, suffix


def _build_model(
    config: Config,
    params: ModelParams,
    full_dataset,
    train_loader,
) -> torch.nn.Module:
    model = build_model(
        full_dataset,
        config,
        params.d_model,
        params.nhead,
        params.num_layers,
        params.dropout,
        train_loader=train_loader,
        d_freq=params.d_freq,
        d_var=params.d_var,
        dim_feedforward=params.dim_feedforward,
        activation=params.activation,
        use_nonlinearity=params.use_nonlinearity,
        use_attention=params.use_attention,
    )
    return model


def _select_loader(split: str, train_loader, val_loader, test_loader):
    split = split.lower()
    if split == "train":
        return train_loader
    if split == "val":
        if val_loader is None:
            raise ValueError("Validation loader requested but Optuna/validation split not enabled.")
        return val_loader
    if split == "test":
        return test_loader
    raise ValueError(f"Unknown split '{split}'. Expected one of 'train', 'val', or 'test'.")


def _resolve_capture_modules(model: torch.nn.Module, requested: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for name in requested:
        if name.endswith(".-1"):
            prefix, _ = name.rsplit(".", 1)
            parent = model.get_submodule(prefix)
            child_indices = [str(i) for i, _ in enumerate(parent.children())]
            if not child_indices:
                raise ValueError(f"Module '{prefix}' has no children to index.")
            resolved.append(f"{prefix}.{child_indices[-1]}")
        else:
            resolved.append(name)
    return resolved


def _advance_iterator(iterator, steps: int):
    for _ in range(steps):
        next(iterator, None)
    return iterator


def _prepare_inspection_environment(config: Config):
    (
        full_dataset,
        train_loader,
        val_loader,
        test_loader,
        _,
        _,
    ) = _prepare_dataset(config)
    return full_dataset, train_loader, val_loader, test_loader


def _run_inspection(
    request: InspectionRequest,
    config: Config,
    full_dataset,
    train_loader,
    val_loader,
    test_loader,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    print(f"\n=== Inspection: {request.experiment} ===")
    if not request.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint '{request.checkpoint}' does not exist.")

    overrides = _load_params_override(request.params_path)
    params = ModelParams.from_config(config, overrides)

    state_dict = torch.load(request.checkpoint, map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a valid state_dict mapping.")

    if request.print_weights:
        print("\nParameter statistics:")
        for line in summarize_state_dict(state_dict):
            print(line)

    model = _build_model(config, params, full_dataset, train_loader)
    model.load_state_dict(state_dict)

    device = torch.device(request.device) if request.device else _default_device()
    model.to(device)
    model.eval()

    if not request.run_forward:
        print("Checkpoint loaded. Forward pass skipped per configuration.")
        return None, {}

    loader = _select_loader(request.split, train_loader, val_loader, test_loader)
    if loader is None:
        raise RuntimeError("Requested data loader could not be resolved.")

    batch_iter = _advance_iterator(iter(loader), request.batch_index)
    batch = next(batch_iter)

    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"value", "var_id", "freq_id"}
    }

    capture_modules = _resolve_capture_modules(model, request.capture)
    activations, hooks = _register_hooks(model, capture_modules)

    with torch.no_grad():
        preds = model(**inputs)

    _cleanup_hooks(hooks)

    print("\nCaptured activations:")
    for name, tensor in activations.items():
        print(f"- {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    scaler = getattr(full_dataset, "scaler", None)
    if scaler is not None:
        preds_unscaled = scaler.inverse_transform(preds.detach().cpu().reshape(-1, 1)).flatten()
        print("\nPredictions (original scale):")
        print(preds_unscaled)
    else:
        print("\nPredictions (scaled):")
        print(preds.detach().cpu().numpy())

    return preds, activations


def inspect_experiment(
    experiment: str,
    *,
    device: Optional[str] = None,
    print_weights: bool = False,
    split: str = "test",
    batch_index: int = 0,
    capture: Optional[Sequence[str]] = None,
    run_forward: bool = True,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Inspect a trained experiment located under ``outputs/experiments``.

    Parameters mirror :func:`build_inspection_request` and are intentionally kept
    simple – supply the experiment folder name or path and tweak optional flags
    as needed.  Any missing files raise immediately to avoid silently falling
    back to incorrect defaults.

    Returns
    -------
    Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]
        The predictions tensor (``None`` when ``run_forward`` is ``False``) and a
        mapping of captured activation names to tensors on the CPU.
    """

    request = build_inspection_request(
        experiment,
        device=device,
        print_weights=print_weights,
        split=split,
        batch_index=batch_index,
        capture=capture,
        run_forward=run_forward,
    )

    run_config = Config(request.config_path)
    (
        full_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = _prepare_inspection_environment(run_config)

    return _run_inspection(
        request,
        run_config,
        full_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Call inspect_experiment(experiment=...) from Python instead of using CLI arguments."
    )

