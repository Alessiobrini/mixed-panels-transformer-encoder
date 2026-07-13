"""Aggregate the init-seed robustness sweep (INTERNAL check, not for the paper).

For each (target, variant) it computes full-sample RMSE/MAE/DA from each seed's
transformer_preds (outputs/experiments/{base}_ss{seed}/) and compares the 10-seed
mean +/- SD against the published lead2 single-seed value (outputs/experiments/{base}/).
Writes a markdown report flagging whether the published number sits inside the seed spread.

Usage:
    python src/evaluation/aggregate_seed_sweep.py --datetag 2026-06-12_lead2 \
        --out revision_exhibits/seed_sweep_report.md
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXPDIR = REPO / "outputs" / "experiments"
TARGETS = ["GDPC1", "GPDIC1", "PCECC96", "DPIC96", "OUTNFB", "UNRATE",
           "PCECTPI", "PCEPILFE", "CPIAUCSL", "CPILFESL", "FPIx", "EXPGSC1", "IMPGSC1"]
VARIANTS = [("MPTE", "full"), ("AB1", "B1_no_nonlinearity"), ("AB2", "B2_no_attention"),
            ("AB3", "B6_y_only"), ("AB4", "B3_no_attention_no_nonlinearity"),
            ("AB5", "B5_no_positional_encoding")]
SEEDS = list(range(1001, 1011))


def metrics(folder: Path):
    f = next(folder.glob("transformer_preds_*.csv"), None)
    if f is None:
        return None
    df = pd.read_csv(f)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "true", df.columns[2]: "pred"})
    yt, yp = df["true"].to_numpy(), df["pred"].to_numpy()
    if len(yt) < 2:
        return None
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    da = float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp))))
    return {"RMSE": rmse, "MAE": mae, "DA": da}


def base_name(target, scen, datetag):
    return f"{target}_{datetag}" if scen == "full" else f"{target}_{datetag}_{scen}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datetag", default="2026-06-12_lead2")
    ap.add_argument("--out", default=str(REPO / "revision_exhibits" / "seed_sweep_report.md"))
    args = ap.parse_args()

    rows = []
    for target in TARGETS:
        for lab, scen in VARIANTS:
            base = base_name(target, scen, args.datetag)
            ref = metrics(EXPDIR / base)
            seed_m = [metrics(EXPDIR / f"{base}_ss{s}") for s in SEEDS]
            seed_m = [m for m in seed_m if m is not None]
            if not seed_m:
                continue
            row = {"target": target, "variant": lab, "n_seeds": len(seed_m)}
            for met in ["RMSE", "MAE", "DA"]:
                vals = np.array([m[met] for m in seed_m])
                row[f"{met}_ref"] = ref[met] if ref else np.nan
                row[f"{met}_mean"] = vals.mean()
                row[f"{met}_sd"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
                row[f"{met}_min"] = vals.min()
                row[f"{met}_max"] = vals.max()
                if ref:
                    lo, hi = vals.mean() - vals.std(ddof=1), vals.mean() + vals.std(ddof=1)
                    row[f"{met}_in1sd"] = bool(lo <= ref[met] <= hi)
                    row[f"{met}_inrange"] = bool(vals.min() <= ref[met] <= vals.max())
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_out = Path(args.out).with_suffix(".csv")
    df.to_csv(csv_out, index=False)

    # Markdown report focused on RMSE (the headline metric)
    L = ["# Init-seed robustness sweep vs. published lead2 (RMSE) -- INTERNAL\n",
         f"10 init seeds, frozen tuned HPs, faithful val-based path. Cells: published ref vs seed mean (SD) "
         f"[min, max]. `in1SD`=ref within mean+/-1SD; `inRange`=ref within [min,max].\n",
         "| Target | Variant | ref RMSE | mean (SD) | [min, max] | in1SD | inRange |",
         "|--------|---------|---------|-----------|------------|-------|---------|"]
    n_in1, n_inr, n_tot, n_noref = 0, 0, 0, 0
    for _, r in df.iterrows():
        has_ref = pd.notna(r.get("RMSE_ref"))
        in1 = bool(r.get("RMSE_in1sd")) if pd.notna(r.get("RMSE_in1sd")) else False
        inr = bool(r.get("RMSE_inrange")) if pd.notna(r.get("RMSE_inrange")) else False
        if has_ref:
            n_tot += 1
            n_in1 += int(in1)
            n_inr += int(inr)
        else:
            n_noref += 1
        ref_str = f"{r['RMSE_ref']:.4f}" if has_ref else "--"
        L.append(f"| {r['target']} | {r['variant']} | {ref_str} | "
                 f"{r['RMSE_mean']:.4f} ({r['RMSE_sd']:.4f}) | "
                 f"[{r['RMSE_min']:.4f}, {r['RMSE_max']:.4f}] | "
                 f"{'yes' if in1 else ('NO' if has_ref else '--')} | "
                 f"{'yes' if inr else ('NO' if has_ref else '--')} |")
    L.insert(2, f"\n**Summary (RMSE):** published value within 1 SD of the seed mean in "
                f"{n_in1}/{n_tot} cells; within [min,max] in {n_inr}/{n_tot} cells."
                + (f" ({n_noref} cells had no local reference preds.)" if n_noref else "") + "\n")
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}\nwrote {csv_out}")
    print(f"RMSE: ref within 1SD in {n_in1}/{n_tot}; within range in {n_inr}/{n_tot}")


if __name__ == "__main__":
    main()
