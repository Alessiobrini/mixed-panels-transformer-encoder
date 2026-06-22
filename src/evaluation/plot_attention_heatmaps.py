"""Reproducible source for the paper's attention heatmaps (Fig:Az_*, Fig:B_*).

The full per-sequence attention dumps (``attention_summary.json``, ~17 MB each) are not in
the repo. This script instead plots from the *compact* exhibit data committed under
``replication/data/attention/`` -- one small JSON per (target, variant) holding exactly the
fields the heatmaps use: the averaged cross-sectional matrix ``overall_mean.Ax``, the temporal
matrix ``overall_mean.B``, and the axis labels (``variable_order`` / ``B_time_labels``). Those
compact matrices are byte-identical to what the committed figures were plotted from, so this
regenerates the heatmaps faithfully. The plotting matches read_aggregate_attention.py.

Usage:
    python src/evaluation/plot_attention_heatmaps.py \
        --data-dir replication/data/attention --outdir paper/figs
Writes overall_mean_Ax_<VARIANT>_REVISED.pdf and overall_mean_B_<VARIANT>_REVISED.pdf into
paper/figs/<FIGFOLDER>/ (GDPC1 -> GDPC1, OUTNFB -> OUTFNB, matching the paths in main.tex).
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 6
mpl.rcParams["ytick.labelsize"] = 6

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# compact-file stem -> (figs subfolder used in main.tex, variant suffix in the pdf name)
SOURCES = {
    "GDPC1_MPTE": ("GDPC1", "MPTE"),
    "GDPC1_AB1": ("GDPC1", "AB1"),
    "OUTNFB_MPTE": ("OUTFNB", "MPTE"),
    "OUTNFB_AB1": ("OUTFNB", "AB1"),
}


def latex_escape(label):
    if label is None:
        return ""
    return (
        str(label)
        .replace("\\", r"\\")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def plot_heatmap(mat, outfile, xlabels=None, ylabels=None, mask_upper_triangle=False,
                 add_lag_labels=False, enlarge_yaxis=False, square_cells=False):
    array = np.array(mat)
    plot_array = array.copy()
    if mask_upper_triangle:
        mask = np.triu(np.ones_like(plot_array, dtype=bool), k=1)
        plot_array = np.ma.array(plot_array, mask=mask)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(alpha=0)
    else:
        cmap = "coolwarm"

    if square_cells:
        figsize = (6, 6)
    else:
        figsize = (6, max(4, len(ylabels) * 0.2)) if enlarge_yaxis and ylabels else None
    fig, ax = plt.subplots(figsize=figsize)
    aspect = "equal" if square_cells else "auto"
    heatmap = ax.imshow(plot_array, cmap=cmap, aspect=aspect)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=90)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6, pad=2)
    if add_lag_labels:
        ax.set_xlabel("lags")
        ax.set_ylabel("lags")
    if square_cells:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(heatmap, cax=cax)
    else:
        fig.colorbar(heatmap, ax=ax)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO / "replication" / "data" / "attention"))
    ap.add_argument("--outdir", default=str(REPO / "paper" / "figs"))
    args = ap.parse_args()
    data_dir, outdir = Path(args.data_dir), Path(args.outdir)

    for stem, (figfolder, variant) in SOURCES.items():
        d = json.load((data_dir / f"{stem}.json").open())
        ax_labels = [latex_escape(v) for v in d.get("variable_order") or []] or None
        b_labels = [str(v).split(" ")[0] for v in d.get("B_time_labels") or []] or None
        plot_heatmap(d["overall_mean"]["Ax"],
                     outdir / figfolder / f"overall_mean_Ax_{variant}_REVISED.pdf",
                     ax_labels, ax_labels, enlarge_yaxis=True, square_cells=True)
        plot_heatmap(d["overall_mean"]["B"],
                     outdir / figfolder / f"overall_mean_B_{variant}_REVISED.pdf",
                     b_labels, b_labels, mask_upper_triangle=True, add_lag_labels=True)
        print(f"wrote {figfolder}/overall_mean_{{Ax,B}}_{variant}_REVISED.pdf")


if __name__ == "__main__":
    main()
