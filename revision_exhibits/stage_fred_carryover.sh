#!/bin/bash
# Stage the FRED lead2 experiment folders LOCALLY for exhibit generation.
#
# Decision (2026-06-13): AR/MIDAS/OLS reproduce the published numbers EXACTLY (deterministic),
# confirming the rerun is faithful. XGB and NN are NOT bit-reproducible (XGBoost multi-thread
# tree building + NN training nondeterminism, even with a fixed seed), and they are lead-INVARIANT
# (quarterly-only). So per the author's decision we CARRY OVER the published XGB/NN values into
# the revised tables, and use the lead2 rerun for everything else (MPTE/AR/MIDAS/OLS).
#
# This script: (1) pulls the lead2 folder CSVs from DCC into outputs/experiments/, and (2) replaces
# each folder's XGB/NN preds with the published 2025-09-26 ones (the fresh rerun is moved to
# rerun_backup/ for provenance). Idempotent.
set -e
cd "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"
PY="${PY:-/Users/alessiobrini/anaconda3/envs/tsa-dev/bin/python3}"
LEAD_DATE="${1:-2026-06-12_lead2}"
PUB_DATE="${2:-2025-09-26}"
REMOTE="dcc:/hpc/group/darec/ab978/mixed-panels-transformer-encoder/outputs/experiments/"

echo "== pull lead2 folder CSVs from DCC =="
rsync -az --include="*_${LEAD_DATE}/" --include="*_${LEAD_DATE}/*.csv" --exclude='*' "$REMOTE" outputs/experiments/

echo "== carry over published XGB/NN (fresh reruns -> rerun_backup/) =="
LEAD_DATE="$LEAD_DATE" PUB_DATE="$PUB_DATE" "$PY" - <<'PY'
import glob, shutil, os
TGT=["GDPC1","GPDIC1","PCECC96","DPIC96","OUTNFB","UNRATE","PCECTPI","PCEPILFE","CPIAUCSL","CPILFESL","FPIx","EXPGSC1","IMPGSC1"]
ld=os.environ["LEAD_DATE"]; pb=os.environ["PUB_DATE"]; base="outputs/experiments"; done=0
for t in TGT:
    lead=f"{base}/{t}_{ld}"; pub=f"{base}/{t}_{pb}"
    if not os.path.isdir(lead): print("  missing lead2 folder", t); continue
    bk=os.path.join(lead,"rerun_backup"); os.makedirs(bk, exist_ok=True)
    for m in ("xgb","nn"):
        for f in glob.glob(f"{lead}/{m}_preds_*.csv"): shutil.move(f, os.path.join(bk, os.path.basename(f)))
        ps=glob.glob(f"{pub}/{m}_preds_*.csv")
        if ps: shutil.copy(ps[0], lead)
        else: print("  NO published", m, "for", t)
    done+=1
print(f"staged {done}/13 lead2 folders")
PY
echo "Done. Generate tables with: $PY src/evaluation/build_empirical_tables.py --experiment-date ${LEAD_DATE} --outdir outputs/tables/lead2"
