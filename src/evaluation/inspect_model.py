"""Reload a trained Mixed-Frequency Transformer from an experiment folder."""
from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.train import build_model, prepare_data  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.data_paths import (  # noqa: E402
    is_simulation_enabled,
    resolve_data_paths,
)


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


def _ensure_processed_data(config: Config) -> Path:
    csv_path, _, _, _ = resolve_data_paths(config, PROJECT_ROOT)
    if csv_path.exists():
        return csv_path

    script_name = "simulate_to_long.py" if is_simulation_enabled(config) else "convert_fred_to_long.py"
    script_path = PROJECT_ROOT / "src" / "data" / script_name
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=PROJECT_ROOT)

    if not csv_path.exists():  # pragma: no cover - defensive
        raise FileNotFoundError(f"Expected processed data at {csv_path} after running {script_name}.")
    return csv_path


def _prepare_training_artifacts(config: Config) -> Dict[str, Any]:
    csv_path = _ensure_processed_data(config)
    data = prepare_data(csv_path, config)

    if config.training.optimize:
        full_dataset, train_loader, val_loader, test_loader, test_idx = data
    else:
        full_dataset, train_loader, test_loader, test_idx = data
        val_loader = None

    return {
        "dataset": full_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "test_indices": test_idx,
        "csv_path": str(csv_path),
    }


def _clone_example_sequence(example: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in example.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone().detach()
        else:
            cloned[key] = value
    return cloned


def _select_example_sequence(data_artifacts: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    dataset = data_artifacts["dataset"]
    total_items = len(dataset)
    if total_items == 0:
        raise ValueError("Dataset is empty; cannot select an example sequence for inspection.")

    test_indices: Iterable[int] = data_artifacts.get("test_indices") or []
    try:
        example_index = int(next(iter(test_indices)))
    except StopIteration:
        example_index = 0

    example_index = max(0, min(example_index, total_items - 1))
    return example_index, dataset[example_index]


def _batch_from_example(example: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    batch = {}
    for key in ("value", "var_id", "freq_id"):
        tensor = example[key]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        batch[key] = tensor.to(device)
    if "target" in example:
        target = example["target"]
        if target.dim() == 0:
            target = target.unsqueeze(0)
        batch["target"] = target.to(device)
    return batch


def _shape_to_str(shape: torch.Size) -> str:
    return " × ".join(str(dim) for dim in shape)


def _trace_model_flow(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    with torch.no_grad():
        value = batch["value"]
        var_id = batch["var_id"]
        freq_id = batch["freq_id"]

        value_unsqueezed = value.unsqueeze(-1)
        steps.append({"op": "value.unsqueeze(-1)", "shape": tuple(value_unsqueezed.shape)})

        var_emb = model.var_embedding(var_id)
        steps.append({"op": "var_embedding(var_id)", "shape": tuple(var_emb.shape)})

        freq_emb = model.freq_embedding(freq_id)
        steps.append({"op": "freq_embedding(freq_id)", "shape": tuple(freq_emb.shape)})

        z = torch.cat([value_unsqueezed, var_emb, freq_emb], dim=-1)
        steps.append({"op": "concat([value, var_emb, freq_emb])", "shape": tuple(z.shape)})

        z_proj = model.input_proj(z)
        steps.append({"op": "input_proj", "shape": tuple(z_proj.shape)})

        z_proj = model.z_proj_norm(z_proj)
        steps.append({"op": "z_proj_norm", "shape": tuple(z_proj.shape)})

        if model.positional_encoding_enabled and model.positional_encoding is not None:
            pos_enc = model._get_positional_encoding(z_proj)
            pos_enc = model.pos_enc_norm(pos_enc)
            steps.append({"op": "positional_encoding", "shape": tuple(pos_enc.shape)})
            z_proj = z_proj + pos_enc
            steps.append({"op": "add(positional_encoding)", "shape": tuple(z_proj.shape)})

        encoded = model.transformer_encoder(z_proj)
        steps.append({"op": "transformer_encoder", "shape": tuple(encoded.shape)})

        pooled = encoded.mean(dim=1)
        steps.append({"op": "mean_pool", "shape": tuple(pooled.shape)})

        pred = model.prediction_head(pooled)
        steps.append({"op": "prediction_head", "shape": tuple(pred.shape)})

    return steps


def _register_encoder_hooks(
    encoder: torch.nn.Module, storage: List[Dict[str, Any]]
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return handles

    for idx, layer in enumerate(layers):
        def _make_hook(index: int):
            def hook(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                storage.append({
                    "layer": index,
                    "tensor": output.detach().cpu(),
                    "shape": tuple(output.shape),
                })

            return hook

        handle = layer.register_forward_hook(_make_hook(idx))
        handles.append(handle)
    return handles


@contextmanager
def _capture_attention_weights(encoder: torch.nn.Module, storage: List[torch.Tensor]):
    patched: List[Tuple[torch.nn.Module, Any]] = []
    layers = getattr(encoder, "layers", None)
    if layers is None:
        yield
        return

    try:
        for layer in layers:
            mha = getattr(layer, "self_attn", None)
            if mha is None:
                continue

            original_forward = mha.forward

            def make_wrapped(original):
                def wrapped(query, key, value, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    attn_output, attn_weights = original(query, key, value, **kwargs)
                    storage.append(attn_weights.detach().cpu())
                    return attn_output, attn_weights

                return wrapped

            mha.forward = make_wrapped(original_forward)
            patched.append((mha, original_forward))

        yield
    finally:
        for module, original in patched:
            module.forward = original


def _collect_attention_and_hidden_states(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    encoder = getattr(model, "transformer_encoder", None)
    if encoder is None:
        return [], []

    hidden_states: List[Dict[str, Any]] = []
    attention_matrices: List[torch.Tensor] = []

    handles = _register_encoder_hooks(encoder, hidden_states)
    try:
        with _capture_attention_weights(encoder, attention_matrices):
            with torch.no_grad():
                model(
                    value=batch["value"],
                    var_id=batch["var_id"],
                    freq_id=batch["freq_id"],
                )
    finally:
        for handle in handles:
            handle.remove()

    hidden_states.sort(key=lambda entry: entry["layer"])
    return attention_matrices, hidden_states


def _inspect_example_sequence(
    model: torch.nn.Module,
    example: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    batch = _batch_from_example(example, device)
    flow = _trace_model_flow(model, batch)
    attention_matrices, hidden_states = _collect_attention_and_hidden_states(model, batch)

    return {
        "forward_flow": flow,
        "attention_matrices": attention_matrices,
        "encoder_hidden_states": hidden_states,
    }


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

    data_artifacts = _prepare_training_artifacts(run_config)
    full_dataset = data_artifacts["dataset"]
    train_loader = data_artifacts["train_loader"]

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

    example_index, example_sequence = _select_example_sequence(data_artifacts)
    cloned_example = _clone_example_sequence(example_sequence)
    inspection = _inspect_example_sequence(model, cloned_example, target_device)

    meta = {
        "experiment_dir": str(experiment_dir),
        "config_path": str(cfg_path),
        "used_config": str(used_config_path),
        "params_path": str(params_path) if params_path.exists() else None,
        "applied_overrides": applied_overrides,
        "device": str(target_device),
        "data_artifacts": data_artifacts,
        "example_sequence_index": example_index,
        "example_sequence": cloned_example,
        "example_inspection": inspection,
    }

    return model, run_config, checkpoint_path, meta


if __name__ == "__main__":
    model, run_config, checkpoint, meta = reload_from_config()
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {meta['device']}")
    
    # Compute context vector length based on config
    context_days = run_config.data.context_days
    n_monthly = len(run_config.features.monthly_vars)
    n_quarterly = len(run_config.features.quarterly_vars)
    
    # Each year has 12 months and 4 quarters
    years_fraction = context_days / 360
    context_length = years_fraction * (n_monthly * 12 + n_quarterly * 4)
    
    print(
        f"Context vector length is {context_length:.0f}, "
        f"based on a context window of {context_days} days "
        f"({years_fraction:.2f} years), with {n_monthly} monthly "
        f"and {n_quarterly} quarterly variables."
    )


    inspection = meta.get("example_inspection", {})
    example_index = meta.get("example_sequence_index")
    if example_index is not None:
        print(f"Example sequence index: {example_index}")

    forward_flow = inspection.get("forward_flow", [])
    if forward_flow:
        print("\nForward pass operation order (single-sequence view):")
        for step in forward_flow:
            op = step["op"]
            shape = _shape_to_str(torch.Size(step["shape"]))
            print(f"  - {op}: {shape}")

    attention_matrices = inspection.get("attention_matrices", [])
    if attention_matrices:
        print("\nAttention matrices (post-softmax) captured per encoder layer:")
        for layer_idx, matrix in enumerate(attention_matrices):
            shape = _shape_to_str(matrix.shape)
            print(f"  - Layer {layer_idx}: {shape} (batch × heads × tgt_len × src_len)")
    else:
        print("\nAttention matrices: not available (attention disabled or no encoder layers).")

    hidden_states = inspection.get("encoder_hidden_states", [])
    if hidden_states:
        print("\nEncoder hidden states after each layer:")
        for entry in hidden_states:
            layer = entry["layer"]
            shape = _shape_to_str(torch.Size(entry["shape"]))
            print(f"  - Layer {layer}: {shape}")
    else:
        print("\nEncoder hidden states: no layers recorded.")
