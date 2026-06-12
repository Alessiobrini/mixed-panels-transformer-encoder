#!/bin/bash
# Backfill MIDAS + evaluate for the FRED lead2 FULL-model folders. The first F1 run broke at
# the MIDAS step (relative config path vs midas.R's setwd), so those folders only have
# transformer_preds. Training is intact; this re-runs only midas.R + evaluate_forecasts (cheap)
# using the per-target runtime config (absolute paths), filling in ar/midas preds + DM/rmse.
# Submit AFTER F1: sbatch --dependency=afterany:<F1> --array=0-12 src/fred_midas_backfill.sh
#
#SBATCH --job-name=fred-midasfix
#SBATCH --partition=common
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-12
#SBATCH --output=outputs/replications/slurm/midasfix_%A_%a.out
#SBATCH --error=outputs/replications/slurm/midasfix_%A_%a.err

set -e
set -o pipefail
source ~/.bashrc
conda activate tsa-dev

PROJECT_ROOT="/hpc/group/darec/ab978/mixed-panels-transformer-encoder"
cd "$PROJECT_ROOT"
RSCRIPT="/hpc/group/darec/ab978/miniconda3/envs/tsa-dev/bin/Rscript"

targets=("GDPC1" "GPDIC1" "PCECC96" "DPIC96" "OUTNFB" "UNRATE" "PCECTPI" "PCEPILFE" "CPIAUCSL" "CPILFESL" "FPIx" "EXPGSC1" "IMPGSC1")
TARGET="${targets[$SLURM_ARRAY_TASK_ID]}"
RT="$PROJECT_ROOT/src/config/_lead_runtime/cfg_${TARGET}.yaml"

echo "=== MIDAS backfill: ${TARGET} (cfg ${RT}) ==="
[ -f "$RT" ] || { echo "ERROR: runtime config missing for ${TARGET}"; exit 1; }

# MIDAS (absolute config so midas.R's setwd does not break path resolution)
"$RSCRIPT" src/models/midas.R "$RT"
# evaluate: copies ar/midas into the experiment folder + computes DM / rmse_summary
python3 src/evaluation/evaluate_forecasts.py --config "$RT"

echo "=== Done backfill ${TARGET} ==="
