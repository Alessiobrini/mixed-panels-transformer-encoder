#!/bin/bash
#
#SBATCH --job-name=tsa-experiments
#SBATCH --partition=gpu-common
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=alessio.brini@duke.edu

set -e
set -o pipefail

source ~/.bashrc
conda activate tsa-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Set project root and move there
PROJECT_ROOT="/hpc/group/darec/ab978/mixed-panels-transformer-encoder"
cd "$PROJECT_ROOT"

# Today's date in YYYY-MM-DD format
TODAY=$(date +%F)

targets=("GDPC1" "GPDIC1" "PCECC96" "DPIC96" "OUTNFB" "UNRATE" "PCECTPI" "PCEPILFE" "CPIAUCSL" "CPILFESL" "FPIx" "EXPGSC1" "IMPGSC1")
CONFIG_PATH="$PROJECT_ROOT/src/config/cfg.yaml"
PYTHON_RUNNER="$PROJECT_ROOT/src/run_pipeline.py"

for target in "${targets[@]}"
do
  EXP_NAME="${target}_${TODAY}"

  echo "==============================="
  echo "Running experiment for: $EXP_NAME"
  echo "==============================="

  python3 - <<EOF
import yaml
from pathlib import Path

config_path = Path("$CONFIG_PATH")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["features"]["target"] = "$target"
cfg["training"]["experiment_name"] = "$EXP_NAME"

with open(config_path, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)
EOF

  python3 "$PYTHON_RUNNER"
done
