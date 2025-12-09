"""Aggregate attention logits across test sequences for an experiment and its ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "src" / "config" / "cfg.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation.compare_embeddings import (
    _extract_base_experiment,
    _iter_related_experiments,
    compute_Ax,
    compute_B,
)
from src.evaluation.inspect_model import (
    _apply_transformer_overrides,
    _as_bool,
    _as_float,
    _as_int,
    _as_optional_int,
    _batch_from_example,
    _collect_attention_and_hidden_states,
    _default_device,
    _load_checkpoint,
    _load_overrides,
    _prepare_training_artifacts,
    _resolve_experiment_dir,
)

from src.train import build_model
from src.utils.config import Config





def _load_model_and_data(experiment_dir: Path, device: torch.device) -> tuple:
    used_config_path = experiment_dir / "used_config.yaml"
    params_path = experiment_dir / "full_final_params.yaml"
    checkpoint_path = _load_checkpoint(experiment_dir)

    run_config = Config(used_config_path)
    overrides = _load_overrides(params_path)
    _apply_transformer_overrides(run_config, overrides)

    transformer_cfg = getattr(getattr(run_config, "model", None), "transformer", None)
    if transformer_cfg is None:
        raise ValueError("used_config.yaml must define model.transformer settings.")

    d_model = _as_int(getattr(transformer_cfg, "d_model", None), "model.transformer.d_model")
    nhead = _as_int(getattr(transformer_cfg, "nhead", None), "model.transformer.nhead")
    num_layers = _as_int(
        getattr(transformer_cfg, "num_layers", None), "model.transformer.num_layers"
    )
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

    model.to(device)
    model.eval()

    return model, run_config, data_artifacts


def _build_time_blocks(context_rows, dataset) -> List[tuple]:
    tokens = []
    for row in context_rows.to_dict("records"):
        tokens.append(
            {
                "time": row.get(dataset.time_column),
                "variable": row.get(dataset.variable_column),
                "frequency": row.get(dataset.freq_column),
            }
        )

    time_blocks: List[tuple] = []
    current_time = None
    current_indices: List[int] = []
    current_variables: List[str | None] = []
    current_frequencies: List[str | None] = []

    for idx, token in enumerate(tokens):
        token_time = str(token.get("time"))

        if current_time is None:
            current_time = token_time

        if token_time != current_time:
            time_blocks.append(
                (current_time, current_indices, current_variables, current_frequencies)
            )
            current_time = token_time
            current_indices = []
            current_variables = []
            current_frequencies = []

        current_indices.append(idx)
        current_variables.append(token.get("variable"))
        current_frequencies.append(token.get("frequency"))

    if current_indices:
        time_blocks.append((current_time, current_indices, current_variables, current_frequencies))

    return time_blocks


def _analyze_sequence(
    model: torch.nn.Module,
    example: Dict[str, Any],
    device: torch.device,
    time_blocks: List[tuple],
    n_monthly: int,
    n_quarterly: int,
) -> tuple[list[Dict[str, Any]], Dict[str, torch.Tensor]]:
    batch = _batch_from_example(example, device)
    _, attention_logits, _ = _collect_attention_and_hidden_states(model, batch)
    if not attention_logits:
        raise ValueError("No attention logits captured; ensure attention is enabled in the model.")

    first_layer_logits = attention_logits[0]
    if first_layer_logits.dim() == 4:
        first_layer_logits = first_layer_logits.squeeze(0)

    per_head_results: list[Dict[str, Any]] = []
    for head_idx in range(first_layer_logits.shape[0]):
        head_logits = first_layer_logits[head_idx]
        _, Ax, contribution_counts = compute_Ax(head_logits, time_blocks, n_monthly, n_quarterly)
        _, B = compute_B(head_logits, time_blocks)

        per_head_results.append(
            {
                "head_index": head_idx,
                "Ax": Ax.cpu(),
                "B": B.cpu(),
                "contribution_counts": contribution_counts.cpu(),
            }
        )

    stacked_Ax = torch.stack([result["Ax"] for result in per_head_results])
    stacked_B = torch.stack([result["B"] for result in per_head_results])
    mean_by_head = {"Ax": stacked_Ax.mean(0), "B": stacked_B.mean(0)}
    return per_head_results, mean_by_head


def _to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _save_results(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")


def _resolve_experiments(config: Config, include_ablations: bool) -> Iterable[Path]:
    evaluation_cfg = getattr(config, "evaluation", None)
    agg_cfg = getattr(evaluation_cfg, "attention_aggregation", None) if evaluation_cfg else None
    experiments = getattr(agg_cfg, "experiments", None) if agg_cfg else None

    if experiments:
        for experiment in experiments:
            base_dir = _resolve_experiment_dir(experiment)
            if include_ablations:
                yield from _iter_related_experiments(base_dir.name)
            else:
                yield base_dir
        return

    base_experiment = _extract_base_experiment(config)
    if include_ablations:
        yield from _iter_related_experiments(base_experiment)
    else:
        yield _resolve_experiment_dir(base_experiment)


def _aggregation_settings(config: Config) -> tuple[bool, str, torch.device]:
    evaluation_cfg = getattr(config, "evaluation", None)
    agg_cfg = getattr(evaluation_cfg, "attention_aggregation", None) if evaluation_cfg else None

    include_ablations = bool(getattr(agg_cfg, "include_ablations", False))
    output_name = str(getattr(agg_cfg, "output_name", "attention_summary.json"))
    device_override = getattr(agg_cfg, "device", None)
    device = _default_device(device_override)

    return include_ablations, output_name, device


def _analyze_experiment(experiment_dir: Path, device: torch.device) -> Dict[str, Any]:
    model, run_config, data_artifacts = _load_model_and_data(experiment_dir, device)
    dataset = data_artifacts["dataset"]
    test_indices: List[int] = list(data_artifacts.get("test_indices") or range(len(dataset)))

    n_monthly = len(run_config.features.monthly_vars)
    n_quarterly = len(run_config.features.quarterly_vars)
    variable_order = [
        *map(str, run_config.features.monthly_vars),
        *map(str, run_config.features.quarterly_vars),
    ]

    num_heads = getattr(getattr(run_config.model, "transformer", None), "nhead", None)
    if num_heads is None:
        raise ValueError("Transformer head count unavailable in configuration.")

    head_accum_Ax: list[list[torch.Tensor]] = [[] for _ in range(num_heads)]
    head_accum_B: list[list[torch.Tensor]] = [[] for _ in range(num_heads)]
    sequence_means: list[Dict[str, torch.Tensor]] = []
    per_sequence_payload: list[Dict[str, Any]] = []
    b_time_labels: list[str] | None = None

    for seq_index in tqdm(test_indices, desc=f"{experiment_dir.name} sequences"):
        sequence_window = dataset.sequence_windows[seq_index]
        context_rows = dataset.df.loc[sequence_window["context_idx"]].reset_index(drop=True)
        time_blocks = _build_time_blocks(context_rows, dataset)

        if b_time_labels is None:
            b_time_labels = [str(block[0]) for block in time_blocks]

        example = dataset[seq_index]
        per_head_results, mean_by_head = _analyze_sequence(
            model,
            example,
            device,
            time_blocks,
            n_monthly,
            n_quarterly,
        )

        for head_result in per_head_results:
            head_idx = head_result["head_index"]
            head_accum_Ax[head_idx].append(head_result["Ax"])
            head_accum_B[head_idx].append(head_result["B"])

        sequence_means.append(mean_by_head)

        per_sequence_payload.append(
            {
                "sequence_index": int(seq_index),
                "context_length": len(sequence_window["context_idx"]),
                "time_blocks": [
                    {
                        "time": block[0],
                        "indices": list(block[1]),
                        "variables": list(block[2]),
                        "frequencies": list(block[3]),
                    }
                    for block in time_blocks
                ],
                "B_time_labels": b_time_labels if b_time_labels is not None else [],
                "per_head": [
                    {
                        "head_index": head_result["head_index"],
                        "Ax": head_result["Ax"],
                        "B": head_result["B"],
                        "contribution_counts": head_result["contribution_counts"],
                    }
                    for head_result in per_head_results
                ],
                "mean_by_head": mean_by_head,
            }
        )

    head_means = []
    for head_idx in range(num_heads):
        if head_accum_Ax[head_idx]:
            head_means.append(
                {
                    "head_index": head_idx,
                    "Ax": torch.stack(head_accum_Ax[head_idx]).mean(0),
                    "B": torch.stack(head_accum_B[head_idx]).mean(0),
                }
            )

    if not sequence_means:
        raise ValueError(
            f"No sequences were analyzed for experiment {experiment_dir.name}; "
            "ensure the test split is non-empty."
        )

    overall_by_sequence = {
        "Ax": torch.stack([entry["Ax"] for entry in sequence_means]).mean(0),
        "B": torch.stack([entry["B"] for entry in sequence_means]).mean(0),
    }

    flattened_Ax = [tensor for head_list in head_accum_Ax for tensor in head_list]
    flattened_B = [tensor for head_list in head_accum_B for tensor in head_list]
    overall_by_head_and_sequence = {
        "Ax": torch.stack(flattened_Ax).mean(0) if flattened_Ax else torch.tensor([]),
        "B": torch.stack(flattened_B).mean(0) if flattened_B else torch.tensor([]),
    }

    return {
        "experiment": experiment_dir.name,
        "experiment_dir": str(experiment_dir),
        "num_sequences": len(test_indices),
        "num_heads": num_heads,
        "n_monthly": n_monthly,
        "n_quarterly": n_quarterly,
        "variable_order": variable_order,
        "B_time_labels": b_time_labels if b_time_labels is not None else [],
        "per_sequence": per_sequence_payload,
        "mean_by_head_across_sequences": head_means,
        "mean_by_sequence": overall_by_sequence,
        "overall_mean": overall_by_head_and_sequence,
    }


def main() -> None:
    config = Config(DEFAULT_CONFIG)
    include_ablations, output_name, device = _aggregation_settings(config)

    for experiment_dir in _resolve_experiments(config, include_ablations):
        try:
            results = _analyze_experiment(experiment_dir, device)
        except ValueError as exc:
            if str(exc) == "No attention logits captured; ensure attention is enabled in the model.":
                print(f"Skipping {experiment_dir.name}: no attention logits captured.")
                continue
            raise
        output_dir = experiment_dir / "model_inspection" / "attention_analysis"
        _save_results(output_dir / output_name, results)
        print(
            f"Saved aggregated attention results for {experiment_dir.name} "
            f"to {output_dir / output_name}"
        )


if __name__ == "__main__":
    main()
