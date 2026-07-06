#!/usr/bin/env bash
# theorysim (JBES theory-validation add-on) -- build the report end to end.
# Runs the experiments, builds figures/tables, the reproducibility appendix, then latexmk.
# QUICK=1 (default) uses modest rep counts for a fast first compile; QUICK=0 uses full counts.
set -euo pipefail

PY=/Users/alessiobrini/anaconda3/envs/tsa-dev/bin/python3
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
REPORT=reports/theory_validation
QUICK="${QUICK:-1}"

if [ "$QUICK" = "1" ]; then
  E1REPS=40; E1GRID="80 160 320"; E2REPS=2000; E3REPS=150; E4OOS=30; E4N=200; E4EP=800; EP=400
  E2BIASREPS=150
else
  E1REPS=2000; E1GRID="80 160 320 640 1000"; E2REPS=2000; E3REPS=2000; E4OOS=2000; E4N=300; E4EP=1200; EP=800
  E2BIASREPS=400
fi

echo "== E1 relative rerun: oracle + parameter_free =="
$PY -m src.theorysim.rerun_e1_relative --reps "$E1REPS" --epochs "$EP" --grid $E1GRID \
    --arms oracle parameter_free --out outputs/theorysim/e1/relative_rerun.csv
echo "== E1 relative rerun: independent_lag (paper-faithful) =="
$PY -m src.theorysim.rerun_e1_relative --reps "$E1REPS" --epochs "$EP" --grid $E1GRID \
    --arms independent_lag --out outputs/theorysim/e1/relative_rerun_lag.csv
echo "== E2 coverage =="
$PY -m src.theorysim.exp_e2_coverage --reps "$E2REPS"
echo "== E2 gate: operator arms x variance channels (July 6 addition) =="
$PY -m src.theorysim.rerun_e2_both_arms --reps "$E2REPS" --epochs "$EP" --kappa 3.0
echo "== E2 bias-vs-N sweep (July 6 addition) =="
$PY -m src.theorysim.rerun_e2_bias_sweep --reps "$E2BIASREPS" --epochs "$EP"
echo "== E3 efficiency =="
$PY -m src.theorysim.exp_e3_efficiency --reps "$E3REPS"
echo "== E4 bridge =="
$PY -m src.theorysim.exp_e4_bridge --N "$E4N" --T "$E4N" --n-oos "$E4OOS" --epochs "$E4EP"
echo "== figures + tables + heatmaps =="
$PY -m src.theorysim.make_report_assets

echo "== latexmk =="
cd "$REPORT"
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex >/dev/null
echo "built: $REPORT/report.pdf"
