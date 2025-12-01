"""Utility to load inspection metadata for a base experiment and its ablations."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import torch
import matplotlib.pyplot as plt




PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "src" / "config" / "cfg.yaml"
INSPECTION_DIR_NAME = "model_inspection"
META_FILENAME = "meta.json"

sys.path.append(str(PROJECT_ROOT))
from src.utils.config import Config

def _require(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description}: {path}")
    return path


def _load_config(path: Path | str | None = None) -> Config:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    return Config(cfg_path)


def _extract_base_experiment(config: Config) -> str:
    evaluation_cfg = getattr(config, "evaluation", None)
    experiment = getattr(evaluation_cfg, "experiment", None) if evaluation_cfg else None
    if not experiment:
        raise ValueError("Config must define evaluation.experiment with a base experiment name.")
    return str(experiment)


def _iter_related_experiments(base_experiment: str) -> Iterable[Path]:
    experiments_dir = _require(
        PROJECT_ROOT / "outputs" / "experiments",
        "Missing experiments directory",
    )

    def _is_related(path: Path) -> bool:
        return path.is_dir() and (
            path.name == base_experiment or path.name.startswith(f"{base_experiment}_")
        )

    matches = [path for path in experiments_dir.iterdir() if _is_related(path)]
    if not matches:
        raise FileNotFoundError(
            f"No experiment folders found for base experiment '{base_experiment}' in {experiments_dir}."
        )

    yield from sorted(matches)


def load_meta_for_experiment(experiment_dir: Path) -> Dict:
    inspection_dir = _require(
        experiment_dir / INSPECTION_DIR_NAME,
        "Missing inspection output directory",
    )
    meta_path = _require(inspection_dir / META_FILENAME, "Missing inspection metadata file")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_all_inspection_meta(base_experiment: str) -> Dict[str, Dict]:
    meta: Dict[str, Dict] = {}
    for experiment_dir in _iter_related_experiments(base_experiment):
        meta[experiment_dir.name] = load_meta_for_experiment(experiment_dir)
    return meta


def plot_attention_heatmap(
    attention: Sequence[Sequence[float]],
    title: str | None = None,
    xlabel: str = "Source",
    ylabel: str = "Target",
):
    """Plot an attention heatmap using the viridis colormap.

    Parameters
    ----------
    attention
        2D attention matrix or logits with shape (target_length, source_length).
    title
        Optional title for the plot.
    xlabel
        Label for the x-axis (source tokens).
    ylabel
        Label for the y-axis (target tokens).
    """

    fig, ax = plt.subplots()
    heatmap = ax.imshow(attention, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    fig.colorbar(heatmap, ax=ax, label="Attention")
    return fig, ax


if __name__ == "__main__":
    config = _load_config()
    experiment = _extract_base_experiment(config)
    meta_by_experiment = load_all_inspection_meta(experiment)

    print(
        f"Loaded inspection metadata for {len(meta_by_experiment)} "
        f"experiment(s) using base '{experiment}'."
    )

    inspection = meta_by_experiment[experiment]["example_inspection"]
    forward_flow = inspection["forward_flow"]
    attention_matrices = torch.tensor(inspection["attention_matrices"]).squeeze()
    attention_logits = torch.tensor(inspection["attention_logits"]).squeeze()
    encoder_hidden_states = inspection["encoder_hidden_states"]

    base_att = attention_logits[0][0]
    run_config = Config(meta_by_experiment[experiment]["used_config"])

    context_days = run_config.data.context_days
    n_monthly = len(run_config.features.monthly_vars)
    n_quarterly = len(run_config.features.quarterly_vars)
    years_fraction = context_days / 360
    context_length = int(years_fraction * (n_monthly * 12 + n_quarterly * 4))
    print(f"Computed context length: {context_length}")
    if context_length != base_att.shape[0]:
        print(
            "Warning: context length does not match attention matrix size "
            f"({context_length} vs {base_att.shape[0]})."
        )

    token_meta_raw = meta_by_experiment[experiment]["example_context_token_metadata"]
    token_meta = {int(k): v for k, v in token_meta_raw.items()}
    ordered_indices = sorted(token_meta)
    if ordered_indices != list(range(base_att.shape[0])):
        print(
            "Warning: token metadata indices do not cover expected range "
            f"0-{base_att.shape[0] - 1}: {ordered_indices[:10]} ..."
        )
    ordered_tokens = [token_meta[i] for i in range(base_att.shape[0])]

    variable_blocks: List[tuple[str, int, int]] = []
    current_variable = None
    block_start = 0

    for idx, token in enumerate(ordered_tokens):
        variable = token.get("variable")
        if current_variable is None:
            current_variable = variable
            block_start = idx
            continue

        if variable != current_variable:
            variable_blocks.append((current_variable, block_start, idx - 1))
            current_variable = variable
            block_start = idx

    if ordered_tokens:
        variable_blocks.append((current_variable, block_start, len(ordered_tokens) - 1))

    full_partition = bool(variable_blocks) and (
        variable_blocks[0][1] == 0
        and variable_blocks[-1][2] == len(ordered_tokens) - 1
        and all(variable_blocks[i][2] + 1 == variable_blocks[i + 1][1] for i in range(len(variable_blocks) - 1))
    )

    print(f"Total variable blocks: {len(variable_blocks)}")
    print("First 10 blocks:", variable_blocks[:10])
    print("Blocks fully partition [0, 943]:", full_partition)

    print(f"Base attention_logits length: {len(attention_logits)}")

    ablation_keys = sorted(
        key
        for key in meta_by_experiment
        if key != experiment and key.startswith(f"{experiment}_")
    )

    if ablation_keys:
        print("Available ablations:")
        for idx, ablation_name in enumerate(ablation_keys, start=1):
            suffix = ablation_name.split("_", maxsplit=1)[-1]
            print(f" {idx}: {suffix} ({ablation_name})")

        ablation_selection = input(
            "Select ablation by number, name, or B-identifier (Enter to skip): "
        ).strip()

        if ablation_selection:
            ablation_experiment = None

            if ablation_selection.isdigit():
                idx = int(ablation_selection) - 1
                if 0 <= idx < len(ablation_keys):
                    ablation_experiment = ablation_keys[idx]
            elif ablation_selection.upper().startswith("B") and ablation_selection[1:].isdigit():
                ablation_experiment = next(k for k in meta_by_experiment if ablation_selection in k)

            if ablation_experiment is None:
                raise ValueError(
                    f"Could not resolve selection '{ablation_selection}' to a known ablation."
                )

            inspection_ablation = meta_by_experiment[ablation_experiment]["example_inspection"]
            forward_flow_ablation = inspection_ablation["forward_flow"]
            attention_matrices_ablation = torch.tensor(inspection_ablation["attention_matrices"]).squeeze()
            attention_logits_ablation = torch.tensor(inspection_ablation["attention_logits"]).squeeze()
            encoder_hidden_states_ablation = inspection_ablation["encoder_hidden_states"]
            # torch.tensor(encoder_hidden_states[0]['tensor']).squeeze()

            print("Ablation selected:", ablation_experiment)
            print(f"Ablation attention_logits length: {len(attention_logits_ablation)}")
    
    
    

