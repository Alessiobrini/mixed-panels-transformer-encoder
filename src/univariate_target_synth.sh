#!/bin/bash
#
#SBATCH --job-name=tsa-experiments-synth
#SBATCH --partition=gpu-common
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --output=slurm_synth.out
#SBATCH --error=slurm_synth.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=alessio.brini@duke.edu

set -e
set -o pipefail

source ~/.bashrc
conda activate tsa-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Set project root and move there
PROJECT_ROOT="/hpc/group/darec/ab978/tsa-dev"
cd "$PROJECT_ROOT"

# Today's date in YYYY-MM-DD format
TODAY=$(date +%F)

CONFIG_PATH="$PROJECT_ROOT/src/config/cfg.yaml"
PYTHON_RUNNER="$PROJECT_ROOT/src/run_pipeline.py"
CONFIG_BACKUP="${CONFIG_PATH}.bak"

cp "$CONFIG_PATH" "$CONFIG_BACKUP"

scenarios=(
  "no_nonlinearity"
  "no_attention"
  "no_attention_no_nonlinearity"
  "no_positional_encoding"
  "y_only_no_positional"
)

for scenario in "${scenarios[@]}"
do
  cp "$CONFIG_BACKUP" "$CONFIG_PATH"
  EXP_NAME="synth_${scenario}_${TODAY}"

  echo "==============================="
  echo "Running synthetic experiment: $EXP_NAME"
  echo "==============================="

  python3 - <<EOF_PY
import yaml
from pathlib import Path

config_path = Path("$CONFIG_PATH")
scenario = "$scenario"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["training"]["experiment_name"] = "$EXP_NAME"
cfg.setdefault("simulation", {})
cfg["simulation"]["simulate"] = True

transformer_cfg = cfg.setdefault("model", {}).setdefault("transformer", {})

if scenario == "no_nonlinearity":
    transformer_cfg["use_nonlinearity"] = False
elif scenario == "no_attention":
    transformer_cfg["use_attention"] = False
elif scenario == "no_attention_no_nonlinearity":
    transformer_cfg["use_attention"] = False
    transformer_cfg["use_nonlinearity"] = False
elif scenario == "no_positional_encoding":
    transformer_cfg["use_positional_encoding"] = False
elif scenario == "y_only_no_positional":
    cfg["simulation"]["use_y_only_predictors"] = True
    transformer_cfg["use_positional_encoding"] = False
else:
    raise ValueError(f"Unknown scenario: {scenario}")

with open(config_path, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)
EOF_PY

  python3 "$PYTHON_RUNNER"
done

mv "$CONFIG_BACKUP" "$CONFIG_PATH"
