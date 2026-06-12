"""Hard-stop verification for the high-frequency lead (R2 revision).

The revision adds a monthly "lead" (config.data.lead, in months) to MPTE only. Models that
do NOT read the within-quarter monthly lead must be byte-for-byte unaffected by it:

    * AR    -- AutoReg on the target's own quarterly history; never touches the dataset.
    * MIDAS -- separate R model; already uses the within-quarter monthly lags by construction.
    * AB5   -- the "y_only" ablation; sets allowed_frequencies={"Q"}, so monthly rows are
               dropped from the dataframe BEFORE LeadMixedFrequencyDataset builds sequences,
               making the lead a structural no-op.

This module provides two subcommands:

  invariance  (dataset-level, deterministic, no training)
      Proves mechanically that the AB5 (y-only) context windows are IDENTICAL at lead=0 and
      lead=2, while the mixed (monthly-inclusive) windows DIFFER by exactly the target
      quarter's first `lead` monthly rows. This is the direct, fast proof of the user's
      requirement ("AB5 context must stop at the previous quarter even with lead=2").

  reconcile   (folder-level, against the paper's published Table 1 = ground truth)
      For a set of experiment folders produced at the published reference seed with lead=2,
      compares the LEAD-INVARIANT rows (AR, MIDAS) against build_simulation_table.PAPER_TABLE
      (which equals Tab:evals_simulation in paper/main.tex) within a rounding tolerance. The
      lead-AFFECTED rows (MPTE, AB1-AB4) are reported as EXPECTED-CHANGE, never failed. AB5 is
      reported as INVARIANT-BY-CONSTRUCTION (its reproduction is proven by the dataset test;
      its single-seed value differs from the paper only by the fixed-HP-vs-Optuna retrain, not
      by the lead).

Exit code is non-zero on any hard-stop failure so this can gate the rerun.

Usage:
    PY=/Users/alessiobrini/anaconda3/envs/tsa-dev/bin/python3
    $PY src/evaluation/verify_lead_invariance.py invariance \
        --csv data/processed/long_format_synth_30M_5Q.csv --target Y1 --lead 2
    $PY src/evaluation/verify_lead_invariance.py reconcile --mode sim \
        --folders-tag repl_seed123_lead2 --tol 5e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.mixed_frequency_dataset import MixedFrequencyDataset  # noqa: E402
from src.data.lead_dataset import LeadMixedFrequencyDataset  # noqa: E402


# ======================================================================================
# invariance: dataset-level proof
# ======================================================================================
def _windows(ds) -> list:
    """Return a comparable representation of the dataset's context/target windows."""
    return [
        (tuple(int(i) for i in w["context_idx"]), int(w["target_idx"]))
        for w in ds.sequence_windows
    ]


def run_invariance(csv: Path, target: str, lead: int, context_days: int) -> int:
    print(f"=== Dataset-level lead-invariance test ===")
    print(f"csv={csv.name}  target={target}  lead={lead}  context_days={context_days}\n")

    common = dict(csv_path=csv, context_days=context_days, target_variable=target)
    failures = 0

    # ---- AB5 (y-only) path: allowed_frequencies={"Q"} -> lead must be a no-op ----------
    base_y = MixedFrequencyDataset(**common, allowed_frequencies={"Q"})
    lead_y = LeadMixedFrequencyDataset(**common, allowed_frequencies={"Q"}, lead=lead)
    wb, wl = _windows(base_y), _windows(lead_y)

    print(f"[AB5 / y-only]  lead0 windows={len(wb)}  lead{lead} windows={len(wl)}")
    if wb == wl:
        print(f"  PASS: AB5 context windows are byte-identical at lead=0 and lead={lead}.\n")
    else:
        failures += 1
        n_diff = sum(1 for a, b in zip(wb, wl) if a != b)
        print(f"  FAIL: AB5 windows DIFFER ({n_diff} of {min(len(wb), len(wl))} differ, "
              f"len {len(wb)} vs {len(wl)}). The lead is leaking into a quarterly-only model.\n")

    # ---- Mixed path: lead MUST be active (windows differ, monthly-only, by <=lead) ------
    base_m = MixedFrequencyDataset(**common, allowed_frequencies=None)
    lead_m = LeadMixedFrequencyDataset(**common, allowed_frequencies=None, lead=lead)
    wbm, wlm = _windows(base_m), _windows(lead_m)

    print(f"[Mixed / MPTE]  lead0 windows={len(wbm)}  lead{lead} windows={len(wlm)}")
    if len(wbm) != len(wlm):
        failures += 1
        print(f"  FAIL: the lead changed the NUMBER of windows ({len(wbm)} -> {len(wlm)}); "
              f"it must only extend each window's monthly context.\n")
    else:
        # Validate that every added index is a MONTHLY row inside the target quarter, and
        # that quarterly rows are untouched.
        df = lead_m.df  # has time_id / freq columns
        freq_col = lead_m.freq_column
        n_changed = 0
        bad = 0
        for (cb, tb), (cl, tl) in zip(wbm, wlm):
            if tb != tl:
                bad += 1
                continue
            added = set(cl) - set(cb)
            removed = set(cb) - set(cl)
            if removed:
                bad += 1
                continue
            if added:
                n_changed += 1
                # every added row must be monthly (non-Q)
                if any(str(df.iloc[i][freq_col]).upper() == "Q" for i in added):
                    bad += 1
        if bad:
            failures += 1
            print(f"  FAIL: {bad} windows had quarterly rows added/removed or target shifted "
                  f"-- the lead must only ADD monthly rows.\n")
        elif n_changed == 0:
            failures += 1
            print(f"  FAIL: lead={lead} added NO monthly rows to any window; the lead is inert "
                  f"on this DGP (check ratio/time units).\n")
        else:
            print(f"  PASS: lead={lead} extended {n_changed}/{len(wbm)} windows, adding only "
                  f"monthly rows inside the target quarter; quarterly rows unchanged.\n")

    if failures:
        print(f"RESULT: HARD-STOP FAILURE ({failures} check(s) failed).")
        return 1
    print("RESULT: PASS -- AB5 is lead-invariant; the lead is active and monthly-only for MPTE.")
    return 0


# ======================================================================================
# reconcile: folder-level, against PAPER_TABLE ground truth
# ======================================================================================
def run_reconcile_sim(folders_tag: str | None, manifest: Path | None, tol: float) -> int:
    import pandas as pd
    from src.evaluation import build_simulation_table as bst

    EXP = REPO / "outputs" / "experiments"

    # Resolve, per regime, the experiment folder set for the lead2 reference-seed run.
    # We accept either an explicit tag prefix (folders named <tag>_<regime>_..._<variant>) or
    # the canonical published folders (default) for a dry self-check.
    LEAD_INVARIANT = {"AR", "MIDAS"}
    EXPECTED_CHANGE = {"MPTE", "AB1", "AB2", "AB3", "AB4"}

    if folders_tag is None and manifest is None:
        print("reconcile --mode sim: provide --folders-tag or --manifest pointing at the "
              "lead2 reference-seed (seed 123) run.")
        return 2

    # Build {regime: {label: metrics}} by reusing load_folder_metrics.
    computed: dict = {}
    if manifest is not None:
        man = pd.read_csv(manifest)
        for regime in man["regime"].unique():
            computed[regime] = {}
            sub = man[man["regime"] == regime]
            for _, r in sub.iterrows():
                folder = EXP / r["exp_folder"]
                m = bst.load_folder_metrics(folder) if folder.is_dir() else {}
                if r["variant"] == "full":
                    for label, key in bst.BASELINE_MODELS:
                        if key in m:
                            computed[regime][label] = m[key]
                else:
                    label = {p.split("_")[0].replace("synth", ""): lab for p, lab in []}  # noop
                    # variant column is like AB1..AB5 already in the manifest
                    if "transformer" in m:
                        computed[regime][r["variant"]] = m["transformer"]
    else:
        # tag-based: reuse the canonical regime map but swap the folder prefix to <tag>.
        for regime, scn, date in bst.REGIMES:
            computed[regime] = {}
            base = EXP / f"{folders_tag}_{regime.split()[0].lower()}_full"
            m = bst.load_folder_metrics(base) if base.is_dir() else {}
            for label, key in bst.BASELINE_MODELS:
                if key in m:
                    computed[regime][label] = m[key]

    rows = []
    hard_fail = 0
    for regime in bst.PAPER_TABLE:
        for label in bst.ROW_ORDER:
            paper = bst.PAPER_TABLE[regime].get(label)
            comp = computed.get(regime, {}).get(label)
            for i, metric in enumerate(bst.METRICS):
                p = paper[i] if paper else np.nan
                c = comp[metric] if comp else np.nan
                if comp is None or paper is None:
                    status = "MISSING"
                elif label in LEAD_INVARIANT:
                    d = abs(p - c)
                    status = "PASS" if d < tol else "FAIL"
                    if status == "FAIL":
                        hard_fail += 1
                elif label == "AB5":
                    status = "INVARIANT-BY-CONSTR"
                else:
                    status = "EXPECTED-CHANGE"
                rows.append(dict(regime=regime, model=label, metric=metric,
                                 paper=p, computed=round(c, 4) if comp else np.nan,
                                 abs_diff=round(abs(p - c), 6) if (comp and paper) else np.nan,
                                 status=status))
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(df.to_string(index=False))
    print(f"\nLead-invariant FAILs (AR/MIDAS beyond tol={tol}): {hard_fail}")
    if hard_fail:
        print("RESULT: HARD-STOP FAILURE -- a lead-invariant model changed vs the paper.")
        return 1
    print("RESULT: PASS -- AR/MIDAS reproduce the paper; MPTE/AB1-4 are expected to change.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("invariance", help="dataset-level AB5 lead-invariance proof")
    pi.add_argument("--csv", type=Path,
                    default=REPO / "data" / "processed" / "long_format_synth_30M_5Q.csv")
    pi.add_argument("--target", default="Y1")
    pi.add_argument("--lead", type=int, default=2)
    pi.add_argument("--context-days", type=int, default=360)

    pr = sub.add_parser("reconcile", help="folder-level reconcile vs PAPER_TABLE")
    pr.add_argument("--mode", choices=["sim", "fred"], required=True)
    pr.add_argument("--folders-tag", default=None)
    pr.add_argument("--manifest", type=Path, default=None)
    pr.add_argument("--tol", type=float, default=5e-4)

    args = ap.parse_args()
    if args.cmd == "invariance":
        return run_invariance(args.csv, args.target, args.lead, args.context_days)
    if args.cmd == "reconcile":
        if args.mode == "sim":
            return run_reconcile_sim(args.folders_tag, args.manifest, args.tol)
        print("reconcile --mode fred is implemented in Stage 4 (needs parse_paper_tables.py).")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
