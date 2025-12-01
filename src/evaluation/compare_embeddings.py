"""Utility to load inspection metadata for a base experiment and its ablations."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import torch
import matplotlib.pyplot as plt


def get_submatrix_block(
    matrix: torch.Tensor | Sequence[Sequence[float]],
    row_indices: Sequence[int],
    col_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the submatrix defined by ``row_indices`` and ``col_indices``.

    If ``col_indices`` is omitted, the same indices are used for both axes,
    preserving the previous diagonal-block behavior while allowing rectangular
    blocks when separate index lists are provided.
    """

    tensor = matrix if isinstance(matrix, torch.Tensor) else torch.tensor(matrix)
    row_index_tensor = torch.tensor(row_indices, device=tensor.device)
    col_index_tensor = (
        torch.tensor(col_indices, device=tensor.device)
        if col_indices is not None
        else row_index_tensor
    )
    return tensor.index_select(0, row_index_tensor).index_select(1, col_index_tensor)


def compute_padded_block_attention_average(
    attention_matrix: torch.Tensor,
    time_blocks: Sequence[tuple[str, Sequence[int], Sequence[str | None]]],
    n_monthly: int,
    n_quarterly: int,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Extract, normalize, pad, and average block-diagonal attention slices.

    Parameters
    ----------
    attention_matrix
        Full attention (or attention logits) matrix of shape ``(T, T)``.
    time_blocks
        Iterable of ``(time_label, indices, variable_names)`` entries where
        ``indices`` define the token positions for each time block.
    n_monthly
        Number of monthly variables for the context, used to derive pad size.
    n_quarterly
        Number of quarterly variables for the context, used to derive pad size.

    Returns
    -------
    tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]
        A tuple containing the list of padded block matrices (after softmax),
        the averaged padded matrix, and a count matrix indicating how many
        blocks contributed to each position in the averaged matrix.
    """

    max_block_length = n_monthly + n_quarterly
    device = attention_matrix.device

    padded_blocks: list[torch.Tensor] = []
    sum_matrix = torch.zeros((max_block_length, max_block_length), device=device)
    count_matrix = torch.zeros_like(sum_matrix)

    for _, indices, _ in time_blocks:
        block = get_submatrix_block(attention_matrix, indices)
        block_softmax = torch.softmax(block, dim=1)

        padded_block = torch.zeros_like(sum_matrix)
        block_height, block_width = block_softmax.shape
        padded_block[:block_height, :block_width] = block_softmax

        contribution_mask = torch.zeros_like(sum_matrix)
        contribution_mask[:block_height, :block_width] = 1

        sum_matrix += padded_block
        count_matrix += contribution_mask
        padded_blocks.append(padded_block)

    averaged_matrix = torch.where(
        count_matrix > 0, sum_matrix / count_matrix, torch.zeros_like(sum_matrix)
    )

    return padded_blocks, averaged_matrix, count_matrix


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

    time_blocks: List[tuple[str, List[int], List[str | None]]] = []
    current_time = None
    current_indices: List[int] = []
    current_variables: List[str | None] = []

    for idx, token in enumerate(ordered_tokens):
        token_time = str(token.get("time"))

        if current_time is None:
            current_time = token_time

        if token_time != current_time:
            time_blocks.append((current_time, current_indices, current_variables))
            current_time = token_time
            current_indices = []
            current_variables = []

        current_indices.append(idx)
        current_variables.append(token.get("variable"))

    if current_indices:
        time_blocks.append((current_time, current_indices, current_variables))

    block_lengths = [len(indices) for _, indices, _ in time_blocks]
    combined_length = n_monthly + n_quarterly
    combined_blocks = sum(1 for length in block_lengths if length == combined_length)
    monthly_only_blocks = sum(1 for length in block_lengths if length == n_monthly)

    print(f"Constructed {len(time_blocks)} time blocks from ordered tokens.")
    print(f"Unique block lengths: {sorted(set(block_lengths))}")
    print(
        "Blocks with monthly + quarterly length: "
        f"{combined_blocks}/{len(time_blocks)} ({combined_blocks / len(time_blocks):.2%})"
    )
    print(
        "Blocks with monthly length only: "
        f"{monthly_only_blocks}/{len(time_blocks)} ({monthly_only_blocks / len(time_blocks):.2%})"
    )
    print("First 5 time blocks (timestamp, indices, variables):", time_blocks[:5])

    padded_blocks, averaged_block, contribution_counts = compute_padded_block_attention_average(
        base_att,
        time_blocks,
        n_monthly,
        n_quarterly,
    )

    print(f"Computed {len(padded_blocks)} padded block attention matrices.")
    print(
        "Padded block shape (monthly + quarterly): "
        f"{n_monthly + n_quarterly} x {n_monthly + n_quarterly}"
    )
    print("Averaged padded block attention matrix (rows sum to 1 where present):")
    print(averaged_block)
    print("Contribution count matrix (per-entry divisor used in averaging):")
    print(contribution_counts)

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
    
    
    

