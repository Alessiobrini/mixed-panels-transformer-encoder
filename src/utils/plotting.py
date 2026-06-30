"""Figure-sizing helpers (golden-ratio, textwidth-scaled).

Copied verbatim (with this header) from the canonical helper at
.../teaching/2024_2025/FINTECH540_Fall/material/utils/figsize.py so that paper
figures are sized to the document's real \\textwidth rather than the matplotlib
default. Reused by the theorysim report plots.
"""

import matplotlib.pyplot as plt


def set_size(width, fraction=1, subplots=(1, 1)) -> tuple:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: tuple, optional
            (nrows, ncols) used to set the aspect ratio of a grid

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5 ** 0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def create_figure(width=800):
    return plt.subplots(figsize=set_size(width=width))


def create_figures(nrows, ncols, width=800, tupsize=None):
    if width:
        return plt.subplots(nrows=nrows, ncols=ncols, figsize=set_size(width=width))
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=tupsize)
