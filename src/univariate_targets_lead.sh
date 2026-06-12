#!/bin/bash
#
# FRED empirical rerun WITH the high-frequency lead (R2 revision), as a SLURM array.
# One array task per macro target (0..12), run concurrently -> wall-clock ~= one target.
# Each task writes its OWN per-target config (so parallel tasks never collide on cfg.yaml)
# and runs the full pipeline (convert_fred -> train[Optuna] -> ar -> midas -> evaluate)
# via run_pipeline.py --config. The base config src/config/cfg.yaml is the published
# empirical configuration with data.lead: 2 (simulate: false, optimize: true, n_trials: 500).
#
# Submit:
#   sbatch --array=0-12 src/univariate_targets_lead.sh
# Single-target smoke first (recommended):
#   sbatch --array=0-0 src/univariate_targets_lead.sh
#
#SBATCH --job-name=fred-lead2
#SBATCH --partition=gpu-common
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --array=0-12
#SBATCH --output=outputs/replications/slurm/fredlead_%A_%a.out
#SBATCH --error=outputs/replications/slurm/fredlead_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=alessio.brini@duke.edu

set -e
set -o pipefail

source ~/.bashrc
conda activate tsa-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PROJECT_ROOT="/hpc/group/darec/ab978/mixed-panels-transformer-encoder"
cd "$PROJECT_ROOT"
mkdir -p outputs/replications/slurm src/config/_lead_runtime

RSCRIPT="/hpc/group/darec/ab978/miniconda3/envs/tsa-dev/bin/Rscript"
DATETAG="${DATETAG:-2026-06-12_lead2}"

targets=("GDPC1" "GPDIC1" "PCECC96" "DPIC96" "OUTNFB" "UNRATE" "PCECTPI" "PCEPILFE" "CPIAUCSL" "CPILFESL" "FPIx" "EXPGSC1" "IMPGSC1")
TARGET="${targets[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="${TARGET}_${DATETAG}"
RUNTIME_CFG="src/config/_lead_runtime/cfg_${TARGET}.yaml"

echo "=== FRED lead2 rerun: target=${TARGET} exp=${EXP_NAME} on $(hostname) ==="

# Per-target config: clone the base cfg.yaml, set target + experiment_name. The base already
# carries data.lead: 2, simulate: false, optimize: true, and the published 500-trial space.
TARGET="$TARGET" EXP_NAME="$EXP_NAME" RUNTIME_CFG="$RUNTIME_CFG" PROJECT_ROOT="$PROJECT_ROOT" python3 - <<'EOF'
import os, yaml
from pathlib import Path
tgt = os.environ["TARGET"]
root = os.environ["PROJECT_ROOT"]
cfg = yaml.safe_load(Path("src/config/cfg.yaml").read_text())
cfg["features"]["target"] = tgt
cfg["training"]["experiment_name"] = os.environ["EXP_NAME"]
cfg.setdefault("evaluation", {})["experiment"] = os.environ["EXP_NAME"]
# Namespace the shared intermediate files per target so concurrent array tasks never collide
# on the processed CSV or the flat ar/midas/transformer pred files (data/split identical across
# targets -> file NAMES change only, not numbers). ABSOLUTE paths so midas.R works despite its
# internal setwd (it resolves data/output relative to its own cwd).
cfg["paths"]["data_processed_template"] = f"{root}/data/processed/long_format_fred_{tgt}_{{suffix}}.csv"
outs = cfg["paths"]["outputs"]
for k in ["transformer_preds", "ar_preds", "midas_preds"]:
    outs[k] = f"{root}/outputs/{k}_{tgt}_{{suffix}}.csv"
Path(os.environ["RUNTIME_CFG"]).write_text(yaml.dump(cfg, sort_keys=False))
print("wrote", os.environ["RUNTIME_CFG"])
EOF

python3 src/run_pipeline.py --config "$RUNTIME_CFG" --rscript "$RSCRIPT"

echo "=== Done ${EXP_NAME} ==="
