"""Inspect trained Mixed-Frequency Transformer checkpoints.

This script reloads models trained via ``src/train.py`` and exposes utilities
for

* printing summary statistics of the stored weights, and
* capturing intermediate activations from selected submodules during a forward
  pass.

Runtime configuration is supplied via the project YAML configuration file
(``src/config/cfg.yaml`` by default).  Under the ``evaluation.inspection``
section you may describe one or more inspection "runs", for example::

    evaluation:
      inspection:
        runs:
          - name: optuna-best
            checkpoint: outputs/experiments/Test6/best_model.pt
            params: outputs/experiments/Test6/full_final_params.yaml
            device: cuda
            print_weights: true
            split: test
            batch_index: 0
            capture:
              - input_proj
              - transformer_encoder.layers.0
              - transformer_encoder.layers.-1
              - prediction_head
            forward_pass: true

To trigger the inspector simply execute ``python -m src.evaluation.inspect_model``.
You can override the configuration path by exporting the ``INSPECTION_CONFIG``
environment variable before running the module.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import yaml

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "cfg.yaml"
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
    """Container for an inspection run described in the YAML configuration."""

    name: str
    checkpoint: Path
    params_path: Optional[Path]
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


def _resolve_override_path(checkpoint: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit

    # Common naming convention from train.py when Optuna is enabled.
    candidate = checkpoint.parent / "full_final_params.yaml"
    if candidate.exists():
        return candidate

    candidate = checkpoint.parent / "best_params.yaml"
    if candidate.exists():
        return candidate

    return None


def _resolve_path(value) -> Optional[Path]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Path):
        path = value.expanduser()
    else:
        value_str = str(value).strip()
        if not value_str:
            return None
        path = Path(value_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_inspection_requests(config: Config) -> List[InspectionRequest]:
    evaluation_cfg = getattr(config, "evaluation", None)
    inspection_cfg = getattr(evaluation_cfg, "inspection", None) if evaluation_cfg else None
    if inspection_cfg is None:
        return []

    raw_runs = getattr(inspection_cfg, "runs", None)
    if raw_runs is None:
        raw_runs = [inspection_cfg]

    requests: List[InspectionRequest] = []
    for idx, raw in enumerate(raw_runs):
        if raw is None:
            continue

        if isinstance(raw, dict):
            data = raw
        else:
            data = dict(getattr(raw, "__dict__", {}))

        name = str(data.get("name", f"run_{idx}"))
        checkpoint = _resolve_path(data.get("checkpoint"))
        if checkpoint is None:
            raise ValueError(f"Inspection run '{name}' is missing a checkpoint path.")

        params_path = _resolve_path(data.get("params"))
        device = data.get("device")
        print_weights = bool(data.get("print_weights", False))
        split = str(data.get("split", "test"))
        batch_index = int(data.get("batch_index", 0))
        capture = data.get("capture", DEFAULT_CAPTURE_MODULES)
        if isinstance(capture, str):
            capture = [capture]
        elif capture is None:
            capture = DEFAULT_CAPTURE_MODULES
        else:
            capture = list(capture)
        if not capture:
            capture = []
        run_forward = bool(data.get("forward_pass", data.get("forward", True)))

        requests.append(
            InspectionRequest(
                name=name,
                checkpoint=checkpoint,
                params_path=params_path,
                device=device,
                print_weights=print_weights,
                split=split,
                batch_index=batch_index,
                capture=list(capture),
                run_forward=run_forward,
            )
        )

    return requests


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
):
    print(f"\n=== Inspection: {request.name} ===")
    if not request.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint '{request.checkpoint}' does not exist.")

    override_path = _resolve_override_path(request.checkpoint, request.params_path)
    overrides = _load_params_override(override_path)
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
        return

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


def _determine_config_path() -> Path:
    env_path = os.environ.get("INSPECTION_CONFIG")
    if env_path:
        path = Path(env_path)
    else:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' does not exist.")
    return path


def main():
    config_path = _determine_config_path()
    config = Config(config_path)

    requests = _load_inspection_requests(config)
    if not requests:
        print(
            "No inspection runs configured. Please add entries under "
            "'evaluation.inspection.runs' in the YAML configuration."
        )
        return

    (
        full_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = _prepare_inspection_environment(config)

    for request in requests:
        _run_inspection(
            request,
            config,
            full_dataset,
            train_loader,
            val_loader,
            test_loader,
        )


if __name__ == "__main__":
    main()

