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
  "B1_no_nonlinearity_nss"
  "B2_no_attention_nss"
  "B3_no_attention_no_nonlinearity_nss"
  "B5_no_positional_encoding_nss"
  "B6_y_only_nss"
  
  "B1_no_nonlinearity_lss"
  "B2_no_attention_lss"
  "B3_no_attention_no_nonlinearity_lss"
  "B5_no_positional_encoding_lss"
  "B6_y_only_lss"
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

if nss in scenario:
    if "no_nonlinearity" in scenario:
        transformer_cfg["use_nonlinearity"] = False
    elif "no_attention" in scenario:
        transformer_cfg["use_attention"] = False
    elif "no_attention_no_nonlinearity" in scenario:
        transformer_cfg["use_attention"] = False
        transformer_cfg["use_nonlinearity"] = False
    elif "no_positional_encoding" in scenario:
        transformer_cfg["use_positional_encoding"] = False
    elif "y_only_no_positional" in scenario:
        cfg["simulation"]["use_y_only_predictors"] = True
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
elif lss in scenario:
    if "no_nonlinearity" in scenario:
        transformer_cfg["use_nonlinearity"] = False
    elif "no_attention" in scenario:
        transformer_cfg["use_attention"] = False
    elif "no_attention_no_nonlinearity" in scenario:
        transformer_cfg["use_attention"] = False
        transformer_cfg["use_nonlinearity"] = False
    elif "no_positional_encoding" in scenario:
        transformer_cfg["use_positional_encoding"] = False
    elif "y_only_no_positional" in scenario:
        cfg["simulation"]["use_y_only_predictors"] = True
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
else:
    raise ValueError(f"Unknown simulated dynamics: {scenario}")

with open(config_path, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)
EOF_PY

  python3 "$PYTHON_RUNNER"
done

mv "$CONFIG_BACKUP" "$CONFIG_PATH"
