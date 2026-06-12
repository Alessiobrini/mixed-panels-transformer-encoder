#!/bin/bash
#
# FRED single-frequency baselines (OLS / XGB / NN) for the lead2 rerun. These are
# quarterly-only and lead-INVARIANT, so they must reproduce the published numbers exactly.
# Writes ols/xgb/nn_preds into each {target}_{DATETAG} folder (created by the full-model run).
# Run AFTER the full-model array:
#   sbatch --dependency=afterok:<F1_JOBID> src/single_freq_lead.sh
#
#SBATCH --job-name=fred-sf-lead2
#SBATCH --partition=common
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=outputs/replications/slurm/fredsf_%A.out
#SBATCH --error=outputs/replications/slurm/fredsf_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=alessio.brini@duke.edu

set -e
set -o pipefail

source ~/.bashrc
conda activate tsa-dev

PROJECT_ROOT="/hpc/group/darec/ab978/mixed-panels-transformer-encoder"
cd "$PROJECT_ROOT"
DATETAG="${DATETAG:-2026-06-12_lead2}"

echo "=== FRED single-freq baselines (OLS/XGB/NN) for ${DATETAG} ==="
python3 src/models/single_freq_models.py --config src/config/cfg.yaml --experiment-date "${DATETAG}"
echo "=== Done single-freq ${DATETAG} ==="
