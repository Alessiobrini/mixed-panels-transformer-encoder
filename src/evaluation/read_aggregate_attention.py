import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 10          # base font
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "src" / "config" / "cfg.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import Config  # noqa: E402
from src.evaluation.inspect_model import _resolve_experiment_dir  # noqa: E402

import re

def latex_escape(label: str) -> str:
    if label is None:
        return ""
    # Escape special LaTeX characters
    return (
        label.replace("\\", r"\\")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
    )


# Path to your saved file
config = Config(DEFAULT_CONFIG)
evaluation_cfg = getattr(config, "evaluation", None)
experiment = getattr(evaluation_cfg, "experiment", None) if evaluation_cfg else None
experiment_dir = _resolve_experiment_dir(experiment)
p = (
    experiment_dir
    / "model_inspection"
    / "attention_analysis"
    / "attention_summary.json"
)

with p.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Print top-level structure
print("Top-level keys:")
for k in data.keys():
    print("  -", k)

print("\nSummary fields:")
print("experiment:", data["experiment"])
print("num_sequences:", data["num_sequences"])
print("num_heads:", data["num_heads"])
print("n_monthly:", data["n_monthly"])
print("n_quarterly:", data["n_quarterly"])

# Show structure of one per-sequence entry
print("\nStructure of data['per_sequence'][0]:")
example = data["per_sequence"][0]
for k, v in example.items():
    if isinstance(v, list):
        print(f"  {k}: list[{len(v)}]")
    elif isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())}")
    else:
        print(f"  {k}: {type(v).__name__}")

# Show shape of mean matrices
print("\nShape info:")
print(
    "mean_by_sequence Ax size:",
    len(data["mean_by_sequence"]["Ax"]),
    "x",
    len(data["mean_by_sequence"]["Ax"][0]),
)
print(
    "overall_mean Ax size:",
    len(data["overall_mean"]["Ax"]),
    "x",
    len(data["overall_mean"]["Ax"][0]),
)

ax_labels = data.get("variable_order")
if ax_labels:
    ax_labels = [latex_escape(str(v)) for v in ax_labels]
b_labels = data.get("B_time_labels")
b_labels = [str(v).split(" ")[0] for v in b_labels] if b_labels else None


def plot_heatmap(mat, title, outfile, xlabels=None, ylabels=None):
    array = mat
    if isinstance(mat, torch.Tensor):
        array = mat.detach().cpu().numpy()
    else:
        array = np.array(mat)

    fig, ax = plt.subplots()
    heatmap = ax.imshow(array, cmap="coolwarm", aspect="auto")
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=90)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    
    

    fig.colorbar(heatmap, ax=ax)
    ax.set_title(title)
    # plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


plots_dir = (
    experiment_dir
    / "model_inspection"
    / "attention_analysis"
    / "plots"
)
plots_dir.mkdir(parents=True, exist_ok=True)

# (A) Mean-by-sequence matrices
plot_heatmap(
    data["mean_by_sequence"]["Ax"],
    "Mean by Sequence - Ax",
    plots_dir / "mean_by_sequence_Ax.pdf",
    ax_labels,
    ax_labels,
)
plot_heatmap(
    data["mean_by_sequence"]["B"],
    "Mean by Sequence - B",
    plots_dir / "mean_by_sequence_B.pdf",
    b_labels,
    b_labels,
)

# (B) Overall mean matrices
plot_heatmap(
    data["overall_mean"]["Ax"],
    "Overall Mean - Ax",
    plots_dir / "overall_mean_Ax.pdf",
    ax_labels,
    ax_labels,
)
plot_heatmap(
    data["overall_mean"]["B"],
    "Overall Mean - B",
    plots_dir / "overall_mean_B.pdf",
    b_labels,
    b_labels,
)

# (C) Per-head matrices
for head in data.get("mean_by_head_across_sequences", []):
    head_index = head.get("head_index")
    if head_index is None:
        continue
    plot_heatmap(
        head["Ax"],
        f"Head {head_index} - Ax",
        plots_dir / f"head_{head_index}_Ax.pdf",
        ax_labels,
        ax_labels,
    )
    plot_heatmap(
        head["B"],
        f"Head {head_index} - B",
        plots_dir / f"head_{head_index}_B.pdf",
        b_labels,
        b_labels,
    )
