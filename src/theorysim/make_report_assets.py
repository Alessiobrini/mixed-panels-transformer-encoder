# theorysim (JBES theory-validation add-on) -- build all report figures + tables.
# Isolated; never edits the empirical pipeline. See src/theorysim/README.md.
"""Generate every figure and table for the theory-validation report from the experiment
CSVs in outputs/theorysim/, plus a learned-operator heatmap. Run after the exp_* runners."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
np.seterr(divide="ignore", over="ignore", invalid="ignore")

from src.theorysim import dgp, operators, plots, tables


def build_heatmap_operators(out_dir, N=150, T=200, epochs=800, lr=1e-2, seed=0):
    """Train independent-sample AB1, extract+scale, freeze A_z, B for the heatmap figure."""
    N_y = max(2, round(N / 3))
    N_x = N - N_y
    dims = dict(k_ys=2, k_R=2, sigma2=2.0, T=T, N_x=N_x, N_y=N_y)
    train = dgp.draw(seed=seed, dims=dims)
    model = operators.train_operators(train, k=4, d_model=16, epochs=epochs, lr=lr, seed=seed)
    oos = [dgp.draw(seed=seed + 1000 + j, dims=dims) for j in range(5)]
    S_z, S_tau = operators.extract_and_average(model, oos)
    A_z, B = operators.scale_operators(S_z, S_tau, N, T)
    return operators.freeze_to_disk(A_z, B, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs/theorysim")
    ap.add_argument("--report", default="reports/theory_validation")
    args = ap.parse_args()

    o = Path(args.outputs)
    figs = Path(args.report) / "figs"
    tabs = Path(args.report) / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    # E1 on the relative (scale-invariant) metric: three paper-grounded arms (oracle,
    # parameter-free, learned-frozen-lagged) from the rerun CSVs.
    e1_csvs = [str(o / "e1" / "relative_rerun.csv"), str(o / "e1" / "relative_rerun_lag.csv")]
    arms = ("oracle", "parameter_free", "independent_lag")

    # Figures.
    if any(Path(p).exists() for p in e1_csvs):
        plots.plot_e1_relative(e1_csvs, figs / "e1_rate.pdf", arms=arms)
        tables.make_e1_relative_table(e1_csvs, tabs / "e1_slope.tex", arms=arms)
    if (o / "e2" / "qq_mixed.csv").exists():
        plots.plot_e2_qq(o / "e2" / "qq_mixed.csv", figs / "e2_qq.pdf")
    if (o / "e2" / "coverage.csv").exists():
        tables.make_e2_coverage_table(o / "e2" / "coverage.csv", tabs / "e2_coverage.tex")
    if (o / "e3" / "results.csv").exists():
        plots.plot_e3_efficiency(o / "e3" / "results.csv", figs / "e3_efficiency.pdf")
        tables.make_e3_ratio_table(o / "e3" / "results.csv", tabs / "e3_ratio.tex")
    if (o / "e4" / "results.csv").exists():
        plots.plot_e4_cca(o / "e4" / "results.csv", figs / "e4_cca.pdf")

    # --- Addition (July 2): three appended exhibits, kept separate from the above. ---
    if any(Path(p).exists() for p in e1_csvs):
        plots.plot_e1_error_vs_alpha(e1_csvs, figs / "e1_error_vs_alpha.pdf", arms=arms)
    e4_csv = o / "e4" / "results.csv"
    if e4_csv.exists():
        tables.make_e4_block_table(e4_csv, tabs / "e4_block.tex")
    e2_both = o / "e2" / "coverage_both_arms.csv"
    if e2_both.exists():
        tables.make_e2_both_arms_table(e2_both, tabs / "e2_both_arms.tex")

    # --- Addition (July 6): coverage across operator arms and variance channels. ---
    e2_gate = o / "e2" / "coverage_gate.csv"
    if e2_gate.exists():
        tables.make_e2_gate_table(e2_gate, tabs / "e2_gate.tex")
    if (o / "e2" / "qq_learned_wb_iid.csv").exists():
        plots.plot_e2_learned_qq(o / "e2", figs / "e2_learned_qq.pdf")
    e2_bias = o / "e2" / "bias_sweep.csv"
    if e2_bias.exists():
        plots.plot_e2_bias_sweep(e2_bias, figs / "e2_bias_sweep.pdf")

    # Heatmaps of the learned, frozen operators.
    hm = build_heatmap_operators(o / "operators" / "heatmap")
    plots.plot_operator_heatmaps(hm / "Az.npy", hm / "B.npy", figs / "operators.pdf")

    print(f"assets written to {figs} and {tabs}")


if __name__ == "__main__":
    main()
