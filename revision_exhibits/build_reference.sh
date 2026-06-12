#!/bin/bash
# Build the REVISED reference comparison PDF (lead=2 exhibits, paper order).
# Usage: bash build_reference.sh [FRED_DATETAG] [SIM_TABLE_TEX]
#   FRED_DATETAG   experiment-date suffix of the FRED lead2 folders (default 2026-06-12_lead2)
#   SIM_TABLE_TEX  path to the sim Table 1 .tex (default outputs/tables/repl100_lead2_replications.tex)
set -e
cd "$(dirname "$0")"
ROOT=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)
PY="${PY:-/Users/alessiobrini/anaconda3/envs/tsa-dev/bin/python3}"
FRED_DATE="${1:-2026-06-12_lead2}"
SIM_TEX="${2:-$ROOT/outputs/tables/repl100_lead2_replications.tex}"
mkdir -p tables figs

echo "== empirical tables (+DM/MCS) -> tables/ =="
$PY "$ROOT/src/evaluation/build_empirical_tables.py" --experiment-date "$FRED_DATE" --outdir tables

echo "== forecast plots -> figs/ =="
$PY "$ROOT/src/evaluation/plot_forecasts_paper.py" --experiment-date "$FRED_DATE" \
    --outdir figs --reference-outdir figs || echo "(forecast plots skipped)"

echo "== sim Table 1 =="
if [ -f "$SIM_TEX" ]; then cp "$SIM_TEX" tables/sim_table1.tex; else echo "($SIM_TEX not found yet)"; fi

echo "== reproducibility appendix =="
$PY - <<'PY'
import csv
rows=list(csv.DictReader(open("exhibit_manifest.csv")))
def esc(s):
    s=s.replace("\\","").replace("_",r"\_").replace("{",r"\{").replace("}",r"\}").replace("&",r"\&")
    return s
def brk(s):  # let long paths break at slashes/spaces inside \texttt
    return esc(s).replace("/", "/\\allowbreak ").replace(" ", "\\ \\allowbreak ")
L=[r"\begin{longtable}{p{3.0cm} p{3.3cm} p{8.2cm}}", r"\toprule",
   r"Exhibit & Script & Command \\", r"\midrule", r"\endhead"]
for r in rows:
    L.append(f"{esc(r['exhibit'])} & \\texttt{{\\footnotesize {brk(r['script'])}}} & "
             f"\\texttt{{\\scriptsize {brk(r['command'])}}} \\\\")
L+=[r"\bottomrule", r"\end{longtable}"]
open("repro_appendix.tex","w").write("\n".join(L)+"\n")
print("wrote repro_appendix.tex")
PY

echo "== compile reference.pdf =="
latexmk -pdf -interaction=nonstopmode -halt-on-error reference.tex >/tmp/ref_latex.log 2>&1 \
  || { echo "latex errors:"; grep -iE "^!|error" /tmp/ref_latex.log | head -20; }
echo "Done -> $(pwd)/reference.pdf"
