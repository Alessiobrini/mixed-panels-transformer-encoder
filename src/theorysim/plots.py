# theorysim (JBES theory-validation add-on) -- report figure generators.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Figures for the theory-validation report. Each function reads a results CSV and writes
a PDF sized to the document text width (golden ratio) via src.utils.plotting.set_size."""

from __future__ import annotations

import csv
import collections
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.plotting import set_size

TEXTWIDTH_PT = 469.0  # standalone article \textwidth
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _slope(x, y):
    lx, ly = np.log(np.asarray(x)), np.log(np.asarray(y))
    A = np.vstack([lx, np.ones_like(lx)]).T
    m, b = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(m), float(b)


def plot_e1_rate(csv_by_arm, out):
    """log(mean e_C) vs log(alpha_bar), one series per arm, with fitted slopes."""
    fig, ax = plt.subplots(figsize=set_size(TEXTWIDTH_PT, fraction=0.7))
    for arm, path in csv_by_arm.items():
        if not Path(path).exists():
            continue
        rows = _read(path)
        cells = collections.defaultdict(list)
        ab = {}
        for r in rows:
            key = (r["T"], r["N"])
            cells[key].append(float(r["e_C"]))
            ab[key] = float(r["alpha_bar"])
        xs = [ab[k] for k in cells]
        ys = [np.mean(v) for v in cells.values()]
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        m, b = _slope(xs, ys)
        ax.loglog(xs, ys, "o", label=f"{arm} (slope {m:.2f})")
        ax.loglog(xs, np.exp(b) * xs ** m, "-", lw=1, alpha=0.7)
    ax.set_xlabel(r"theoretical rate $\bar\alpha$")
    ax.set_ylabel(r"mean common-component error $e_C$")
    ax.set_title("E1: consistency (slope $\\approx$ 1 validates Theorem 1)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


ARM_LABEL = {
    "oracle": r"oracle ($A_z{=}I$)",
    "parameter_free": r"parameter-free ($Q{=}K{=}Z$)",
    "independent_lag": "learned, frozen",
    "independent": "learned, frozen",
}


def _e1_rows(csv_paths, arms):
    rows = []
    for p in csv_paths:
        if Path(p).exists():
            rows += _read(p)
    out = {}
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        if not sub:
            continue
        sub.sort(key=lambda r: float(r["N"]))
        out[a] = sub
    return out


def plot_e1_relative(csv_paths, out, arms=("oracle", "parameter_free", "independent_lag")):
    """2x2: relative error vs N (log-log, slope ~ -1) for (a) the common component, (b) factors,
    (c) loadings, and (d) the operator op-norm (grows for the data-driven arms, yet the rates in
    (a)-(c) are unaffected -- the A.7 note)."""
    data = _e1_rows(csv_paths, arms)
    ref = next(iter(data.values()))
    w, _ = set_size(TEXTWIDTH_PT, fraction=1.0, subplots=(2, 2))
    fig, axes = plt.subplots(2, 2, figsize=(w, w * 0.82))
    (axc, axf), (axl, axo) = axes

    conv = [("e_C_rel", axc, r"(a) common component $C$"),
            ("e_F_rel", axf, r"(b) factors $F$"),
            ("e_L_rel", axl, r"(c) loadings $\Lambda$")]
    for key, ax, title in conv:
        for a, sub in data.items():
            N = np.array([float(r["N"]) for r in sub])
            y = np.array([float(r[key]) for r in sub])
            ax.loglog(N, y, "o-", lw=1, label=ARM_LABEL.get(a, a))
        Nref = np.array([float(r["N"]) for r in ref])
        y0 = max(float(r[key]) for r in ref)
        ax.loglog(Nref, y0 * Nref[0] / Nref, "k--", lw=0.8, alpha=0.6, label="slope $-1$")
        ax.set_xlabel(r"$N=T$")
        ax.set_ylabel("relative error")
        ax.set_title(title, fontsize=9)
    axc.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7, loc="upper right")

    for a, sub in data.items():
        N = np.array([float(r["N"]) for r in sub])
        op = np.array([float(r["op_norm"]) for r in sub])
        axo.semilogx(N, op, "o-", lw=1, label=ARM_LABEL.get(a, a))
    axo.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.6)
    axo.set_xlabel(r"$N$")
    axo.set_ylabel(r"$\|A_z\|_{\mathrm{op}}$")
    axo.set_title(r"(d) operator op-norm grows", fontsize=9)
    axo.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_e1_error_vs_alpha(csv_paths, out, arms=("oracle", "parameter_free", "independent_lag")):
    """Addition (July 2): log relative error against the theoretical rate log(alpha_bar) for
    (a) the factors and (b) the loadings, one series per operator arm, with a slope-1 reference.
    The oracle traces the slope-1 line; the data-driven arms scatter because their realized
    alpha_bar carries the growing cross-sectional operator norm, so the rate the theory prices in
    is no longer a clean scalar. We show it as requested; it is read together with the error-vs-N
    panels, which stay on slope -1 for every arm."""
    data = _e1_rows(csv_paths, arms)
    w, _ = set_size(TEXTWIDTH_PT, fraction=1.0, subplots=(1, 2))
    fig, (axf, axl) = plt.subplots(1, 2, figsize=(w, w * 0.44))
    ora = data.get("oracle")
    for key, ax, title in [("e_F_rel", axf, r"(a) factors $F$"),
                           ("e_L_rel", axl, r"(b) loadings $\Lambda$")]:
        for a, sub in data.items():
            ab = np.array([float(r["alpha_bar"]) for r in sub])
            y = np.array([float(r[key]) for r in sub])
            ax.loglog(ab, y, "o", ms=4, label=ARM_LABEL.get(a, a))
        if ora is not None:
            ab0 = np.array([float(r["alpha_bar"]) for r in ora])
            y0 = np.array([float(r[key]) for r in ora])
            c = float(np.median(y0 / ab0))
            xr = np.array([ab0.min(), max(float(r["alpha_bar"]) for s in data.values() for r in s)])
            ax.loglog(xr, c * xr, "k--", lw=0.8, alpha=0.6, label="slope $1$")
        ax.set_xlabel(r"theoretical rate $\bar\alpha$")
        ax.set_ylabel("relative error")
        ax.set_title(title, fontsize=9)
    axf.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_e2_qq(qq_csv, out):
    """QQ plot of the studentized statistic vs N(0,1)."""
    from scipy import stats
    z = np.array([float(r["z"]) for r in _read(qq_csv)])
    fig, ax = plt.subplots(figsize=set_size(TEXTWIDTH_PT, fraction=0.55))
    stats.probplot(z, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(2)
    ax.set_title("QQ of studentized $C_{y,i,t}$ (mixed regime)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_e2_learned_qq(qq_dir, out, arm="learned_wb"):
    """Addition (July): QQ of the studentized learned-arm statistic (mixed regime) under three
    variance channels: iid plug-in (collapses), feasible general plug-in (item 3c), and the
    Monte Carlo sd (item 3a). If the iid tails fan out but general/MC sit on the line, the CLT
    survives concentration and only the plug-in width was wrong."""
    from scipy import stats
    from pathlib import Path as _P
    channels = [("iid", "iid plug-in"), ("general", "general plug-in"),
                ("mc", "Monte Carlo sd")]
    w, _ = set_size(TEXTWIDTH_PT, fraction=1.0, subplots=(1, 3))
    fig, axes = plt.subplots(1, 3, figsize=(w, w * 0.36))
    for ax, (key, title) in zip(axes, channels):
        p = _P(qq_dir) / f"qq_{arm}_{key}.csv"
        if not p.exists():
            ax.set_visible(False)
            continue
        z = np.array([float(r["z"]) for r in _read(p)])
        stats.probplot(z, dist="norm", plot=ax)
        ax.get_lines()[0].set_markersize(2)
        ax.get_lines()[1].set_linewidth(1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("theoretical quantiles", fontsize=8)
        ax.set_ylabel("ordered $z$" if key == "iid" else "", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_e2_bias_sweep(csv_path, out):
    """Addition (July 6): the learned-arm coverage gap does not close as N grows. Left: coverage
    vs N under the Monte Carlo sd and the feasible general plug-in, both plateauing below the
    nominal 0.95. Right: the standardized bias (does not vanish) and the operator participation
    ratio PR/N (the effective dimension keeps shrinking, so the concentration persists)."""
    rows = _read(csv_path)
    N = np.array([float(r["N"]) for r in rows])
    mc = np.array([float(r["mc_cov"]) for r in rows])
    gen = np.array([float(r["gen_cov"]) for r in rows])
    bias = np.array([float(r["bias_sd"]) for r in rows])
    pr = np.array([float(r["pr_over_n"]) for r in rows])

    w, _ = set_size(TEXTWIDTH_PT, fraction=1.0, subplots=(1, 2))
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(w, w * 0.42))

    axc.semilogx(N, mc, "o-", lw=1, label="Monte Carlo sd")
    axc.semilogx(N, gen, "s-", lw=1, label="general plug-in")
    axc.axhline(0.95, color="k", ls="--", lw=0.8, alpha=0.6, label="nominal")
    axc.set_xlabel(r"$N=T$")
    axc.set_ylabel("coverage of 95% CI")
    axc.set_ylim(0.5, 1.0)
    axc.set_title("(a)", fontsize=9)
    axc.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7, loc="lower left")

    axb.semilogx(N, bias, "o-", color="C3", lw=1, label=r"standardized bias $|b|/\mathrm{sd}$")
    axb.set_xlabel(r"$N=T$")
    axb.set_ylabel(r"$|b|/\mathrm{sd}$", color="C3")
    axb.tick_params(axis="y", labelcolor="C3")
    axb.set_ylim(0, max(0.8, bias.max() * 1.2))
    axr = axb.twinx()
    axr.semilogx(N, pr, "^--", color="C0", lw=1, label=r"participation ratio $\mathrm{PR}/N$")
    axr.set_ylabel(r"$\mathrm{PR}/N$", color="C0")
    axr.tick_params(axis="y", labelcolor="C0")
    axr.set_ylim(0, max(pr) * 1.3)
    axb.set_title("(b)", fontsize=9)
    for ax in (axc, axb):
        ax.set_xticks(N)
        ax.set_xticklabels([str(int(n)) for n in N])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_e2_delta_sweep(csv_path, out, kappa0=3.0):
    """Addition (July 6): coverage of the blend arm against effective rank as the shrinkage delta
    grows. The x-axis is PR/N, the effective dimension (equivalently the middle term of alpha_bar,
    1/PR); a vertical line marks the A.7 threshold c_A/kappa0^2. As delta lifts PR/N through the
    threshold, the general-plug-in coverage climbs to nominal: inference switches on exactly as
    the operator re-enters the theory's domain."""
    rows = _read(csv_path)
    regimes = []
    for r in rows:
        if r["regime"] not in regimes:
            regimes.append(r["regime"])
    fig, ax = plt.subplots(figsize=set_size(TEXTWIDTH_PT, fraction=0.7))
    for rg in regimes:
        sub = sorted((r for r in rows if r["regime"] == rg), key=lambda r: float(r["pr_over_n"]))
        pr = [float(r["pr_over_n"]) for r in sub]
        cov = [float(r["gen_cov"]) for r in sub]
        ax.plot(pr, cov, "o-", lw=1, ms=3, label=rg.replace("_", "-"))
    ax.axhline(0.95, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(1.0 / kappa0 ** 2, color="grey", ls=":", lw=1,
               label=r"A.7 threshold $c_A/\kappa_0^2$")
    ax.set_xlabel(r"effective rank PR/$N$ (blend shrinkage $\delta:0\!\to\!5$)")
    ax.set_ylabel("coverage of 95% CI")
    ax.set_ylim(0.75, 1.0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_e3_efficiency(e3_csv, out):
    rows = _read(e3_csv)
    nx = [float(r["N_x"]) for r in rows]
    ratio = [float(r["ratio_yonly_over_joint"]) for r in rows]
    fig, ax = plt.subplots(figsize=set_size(TEXTWIDTH_PT, fraction=0.6))
    ax.plot(nx, ratio, "o-")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"auxiliary size $N_x$")
    ax.set_ylabel("MSE ratio (Y-only / joint)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_e4_cca(e4_csv, out):
    rows = _read(e4_csv)
    vals = {r["metric"]: float(r["value"]) for r in rows}
    nl = [vals[k] for k in sorted(vals) if k.startswith("cca_nl_") and "_ys_" not in k and "_rest_" not in k]
    lin = [vals[k] for k in sorted(vals) if k.startswith("cca_lin_") and "_ys_" not in k and "_rest_" not in k]
    idx = np.arange(len(nl))
    w, _ = set_size(TEXTWIDTH_PT, fraction=1.0)
    fig, ax = plt.subplots(figsize=(w, w * 0.42))   # landscape: shorter on the page
    ax.bar(idx - 0.2, nl, width=0.4, label="nonlinear latent")
    ax.bar(idx + 0.2, lin, width=0.4, label="linear PCA")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"cc{i+1}" for i in idx])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"canonical correlation vs $F^{(B)}$")
    ax.set_title("E4: factor-space recovery on linear truth")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_operator_heatmaps(Az_npy, B_npy, out):
    A_z = np.load(Az_npy)
    B = np.load(B_npy)
    w, _ = set_size(TEXTWIDTH_PT, fraction=1.0, subplots=(1, 2))
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.30))   # short/wide: fits under its section
    for ax, M, title in zip(axes, (A_z, B), (r"$A_z$ (cross-sectional)", r"$B$ (temporal)")):
        im = ax.imshow(M, cmap="coolwarm", aspect="auto")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
