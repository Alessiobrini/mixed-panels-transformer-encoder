# theorysim (JBES theory-validation add-on) -- report LaTeX table generators.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""LaTeX tables for the theory-validation report, reusing the paper's scriptsize +
\\best / \\second conventions (\\best, \\second, \\cellcolor defined in report.tex)."""

from __future__ import annotations

import csv
from pathlib import Path


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _wrap(body, colspec):
    return ("{\\scriptsize\\setlength{\\tabcolsep}{4pt}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{body}\\bottomrule\n"
            "\\end{tabular}}\n")


def make_e2_coverage_table(coverage_csv, out):
    """E2 coverage table; bold the 95% coverage cell closest to nominal (0.95)."""
    rows = _read(coverage_csv)
    cov95 = [float(r["coverage_95"]) for r in rows]
    best = min(range(len(rows)), key=lambda j: abs(cov95[j] - 0.95))
    lines = ["Regime & $N$ & $T$ & Cov.\\ 90\\% & Cov.\\ 95\\% & $z$ sd \\\\\n\\midrule\n"]
    for j, r in enumerate(rows):
        c95 = f"{float(r['coverage_95']):.3f}"
        c95 = f"\\best{{{c95}}}" if j == best else c95
        name = r["regime"].replace("_", "-")
        lines.append(f"{name} & {r['N']} & {r['T']} & {float(r['coverage_90']):.3f} "
                     f"& {c95} & {float(r['z_sd']):.3f} \\\\\n")
    Path(out).write_text(_wrap("".join(lines), "l c c c c c"))


def make_e1_slope_table(csv_by_arm, out):
    """E1 fitted log-log slope per arm (target 1.0)."""
    import numpy as np
    import collections
    lines = ["Operator arm & fitted slope & target \\\\\n\\midrule\n"]
    for arm, path in csv_by_arm.items():
        if not Path(path).exists():
            continue
        rows = _read(path)
        cells = collections.defaultdict(list)
        ab = {}
        for r in rows:
            cells[(r["T"], r["N"])].append(float(r["e_C"]))
            ab[(r["T"], r["N"])] = float(r["alpha_bar"])
        xs = np.array([ab[k] for k in cells])
        ys = np.array([np.mean(v) for v in cells.values()])
        lx, ly = np.log(xs), np.log(ys)
        A = np.vstack([lx, np.ones_like(lx)]).T
        m = float(np.linalg.lstsq(A, ly, rcond=None)[0][0])
        lines.append(f"{arm} & {m:.3f} & 1.000 \\\\\n")
    Path(out).write_text(_wrap("".join(lines), "l c c"))


ARM_LABEL = {
    "oracle": "oracle ($A_z{=}I$)",
    "parameter_free": "parameter-free ($Q{=}K{=}Z$)",
    "independent_lag": "learned, frozen",
    "independent": "learned, frozen",
}


def make_e1_relative_table(csv_paths, out,
                           arms=("oracle", "parameter_free", "independent_lag")):
    """E1 on the relative metric: per arm, the log-log slope of the relative error vs N (target
    -1) for the common component, factors, and loadings, plus the op-norm at the largest N."""
    import numpy as np
    rows = []
    for p in csv_paths:
        if Path(p).exists():
            rows += _read(p)

    def _slope(N, y):
        A = np.vstack([np.log(N), np.ones_like(N)]).T
        return float(np.linalg.lstsq(A, np.log(y), rcond=None)[0][0])

    lines = ["Operator arm & slope $C$ & slope $F$ & slope $\\Lambda$ "
             "& $\\|A_z\\|_{\\mathrm{op}}$ at $N_{\\max}$ \\\\\n\\midrule\n"]
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        if not sub:
            continue
        sub.sort(key=lambda r: float(r["N"]))
        N = np.array([float(r["N"]) for r in sub])
        mC = _slope(N, np.array([float(r["e_C_rel"]) for r in sub]))
        mF = _slope(N, np.array([float(r["e_F_rel"]) for r in sub]))
        mL = _slope(N, np.array([float(r["e_L_rel"]) for r in sub]))
        lines.append(f"{ARM_LABEL.get(a, a)} & {mC:.2f} & {mF:.2f} & {mL:.2f} "
                     f"& {float(sub[-1]['op_norm']):.1f} \\\\\n")
    Path(out).write_text(_wrap("".join(lines), "l c c c c"))


def make_e4_block_table(e4_csv, out):
    """Addition (July 2): block-wise canonical correlations. Y-strong block F^(B)_S (loads the
    target) vs the rest block F^(B)_R (X-only factors), for the nonlinear latent and the linear
    PCA factors. Reads cca_{nl,lin}_{ys,rest}_j from the E4 results CSV."""
    rows = _read(e4_csv)
    v = {r["metric"]: float(r["value"]) for r in rows}

    def col(prefix):
        ks = sorted(k for k in v if k.startswith(prefix))
        return [v[k] for k in ks]

    nl_ys, nl_rest = col("cca_nl_ys_"), col("cca_nl_rest_")
    lin_ys, lin_rest = col("cca_lin_ys_"), col("cca_lin_rest_")

    def fmt(xs):
        return " & ".join(f"{x:.3f}" for x in xs)

    lines = [("Model & \\multicolumn{2}{c}{Y-strong block $F^{(B)}_S$} "
              "& \\multicolumn{2}{c}{rest block $F^{(B)}_R$} \\\\\n"),
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n",
             "& cc$_1$ & cc$_2$ & cc$_1$ & cc$_2$ \\\\\n\\midrule\n",
             f"nonlinear latent & {fmt(nl_ys)} & {fmt(nl_rest)} \\\\\n",
             f"linear PCA & {fmt(lin_ys)} & {fmt(lin_rest)} \\\\\n"]
    Path(out).write_text(_wrap("".join(lines), "l c c c c"))


def make_e2_gate_table(csv_path, out):
    """Addition (July): E2 coverage across operator arms and variance channels. Per regime and
    arm: op-norm and the 95% coverage under the iid plug-in, the feasible general plug-in, and
    the Monte Carlo sd. Reads the long-format coverage_gate.csv."""
    rows = _read(csv_path)
    ARM = [("oracle", "oracle ($A_z{=}I$)"),
           ("learned_raw", "learned, raw"),
           ("learned_wb", "learned, block-restricted"),
           ("clipped_wb", "learned, clipped")]
    regimes = []
    for r in rows:
        if r["regime"] not in regimes:
            regimes.append(r["regime"])

    def cell(regime, arm, method):
        for r in rows:
            if r["regime"] == regime and r["arm"] == arm and r["method"] == method:
                return float(r["coverage_95"]), float(r["op_norm"])
        return None, None

    lines = ["Regime & Operator arm & $\\|A_z\\|_{\\mathrm{op}}$ & iid & general & MC \\\\\n\\midrule\n"]
    for rg in regimes:
        name = rg.replace("_", "-")
        for a, lab in ARM:
            c_iid, opn = cell(rg, a, "iid")
            if c_iid is None:
                continue
            c_gen, _ = cell(rg, a, "general")
            c_mc, _ = cell(rg, a, "mc")

            def f(x):
                return "--" if x is None else f"{x:.3f}"
            gen = f(c_gen)
            # highlight the general-plug-in cell where it sits inside the theory's domain
            if a in ("oracle", "clipped_wb") and c_gen is not None:
                gen = f"\\best{{{gen}}}"
            mc = f(c_mc)
            lines.append(f"{name} & {lab} & {opn:.1f} & {f(c_iid)} & {gen} & {mc} \\\\\n")
        lines.append("\\midrule\n")
    body = "".join(lines[:-1])  # drop trailing midrule
    Path(out).write_text(_wrap(body, "l l c c c c"))


def make_e2_both_arms_table(csv_path, out):
    """Addition (July 2): E2 coverage under oracle vs learned operators, side by side. Bold the
    oracle 95% cells (calibrated); the learned arm is left plain (its coverage collapses)."""
    rows = _read(csv_path)
    by = {(r["regime"], r["arm"]): r for r in rows}
    regimes = []
    for r in rows:
        if r["regime"] not in regimes:
            regimes.append(r["regime"])
    lines = ["Regime & $N$ & $T$ & \\multicolumn{3}{c}{oracle ($A_z{=}I$)} "
             "& \\multicolumn{3}{c}{learned, frozen} \\\\\n",
             "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n",
             "& & & Cov.\\ 90 & Cov.\\ 95 & $z$ sd & Cov.\\ 90 & Cov.\\ 95 "
             "& $z$ sd \\\\\n\\midrule\n"]
    for rg in regimes:
        o = by[(rg, "oracle")]
        l = by[(rg, "learned")]
        name = rg.replace("_", "-")
        c95 = f"\\best{{{float(o['coverage_95']):.3f}}}"
        lines.append(
            f"{name} & {o['N']} & {o['T']} & {float(o['coverage_90']):.3f} & {c95} "
            f"& {float(o['z_sd']):.2f} & {float(l['coverage_90']):.3f} "
            f"& {float(l['coverage_95']):.3f} & {float(l['z_sd']):.2f} \\\\\n")
    Path(out).write_text(_wrap("".join(lines), "l c c c c c c c c"))


def make_e3_ratio_table(e3_csv, out):
    rows = _read(e3_csv)
    lines = ["$N_x$ & MSE joint & MSE Y-only & ratio \\\\\n\\midrule\n"]
    for r in rows:
        lines.append(f"{r['N_x']} & {float(r['mse_joint']):.4f} & {float(r['mse_yonly']):.4f} "
                     f"& {float(r['ratio_yonly_over_joint']):.3f} \\\\\n")
    Path(out).write_text(_wrap("".join(lines), "c c c c"))
