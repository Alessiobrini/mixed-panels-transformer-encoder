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

usage() {
  cat <<'USAGE'
Usage: univariate_targets_ablation.sh [options]

Options:
  --targets LIST      Comma-separated list of targets to run. Defaults to the
                      full macro target list used in prior experiments.
  --scenarios LIST    Comma-separated list of ablation scenarios to run.
                      Defaults to all supported ablation scenarios.
  -h, --help          Show this help message and exit.
USAGE
}

TARGETS=("GDPC1" "GPDIC1" "PCECC96" "DPIC96" "OUTNFB" "UNRATE" "PCECTPI" "PCEPILFE" "CPIAUCSL" "CPILFESL" "FPIx" "EXPGSC1" "IMPGSC1")
USER_SCENARIOS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --targets)
      if [[ -z "$2" ]]; then
        echo "Error: --targets requires a value." >&2
        usage
        exit 1
      fi
      IFS=',' read -r -a TARGETS <<< "$2"
      shift 2
      ;;
    --scenarios)
      if [[ -z "$2" ]]; then
        echo "Error: --scenarios requires a value." >&2
        usage
        exit 1
      fi
      IFS=',' read -r -a USER_SCENARIOS <<< "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

source ~/.bashrc
conda activate tsa-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PROJECT_ROOT="/hpc/group/darec/ab978/tsa-dev"
cd "$PROJECT_ROOT"

CONFIG_PATH="$PROJECT_ROOT/src/config/cfg.yaml"
PYTHON_RUNNER="$PROJECT_ROOT/src/run_pipeline.py"
CONFIG_BACKUP="${CONFIG_PATH}.bak"

if [[ -f "$CONFIG_BACKUP" ]]; then
  echo "Error: backup config already exists at $CONFIG_BACKUP. Remove it before running." >&2
  exit 1
fi

cp "$CONFIG_PATH" "$CONFIG_BACKUP"
trap 'if [[ -f "$CONFIG_BACKUP" ]]; then mv "$CONFIG_BACKUP" "$CONFIG_PATH"; fi' EXIT

DEFAULT_SCENARIOS=(
  "B1_no_nonlinearity"
  "B2_no_attention"
  "B3_no_attention_no_nonlinearity"
  "B5_no_positional_encoding"
  "B6_y_only"
)

if [[ ${#USER_SCENARIOS[@]} -gt 0 ]]; then
  SELECTED_SCENARIOS=("${USER_SCENARIOS[@]}")
else
  SELECTED_SCENARIOS=("${DEFAULT_SCENARIOS[@]}")
fi

for scenario in "${SELECTED_SCENARIOS[@]}"; do
  case "$scenario" in
    B1_no_nonlinearity|B2_no_attention|B3_no_attention_no_nonlinearity|B5_no_positional_encoding|B6_y_only)
      ;;
    *)
      echo "Unknown scenario requested: $scenario" >&2
      exit 1
      ;;
  esac
done

for target in "${TARGETS[@]}"; do
  for scenario in "${SELECTED_SCENARIOS[@]}"; do
    cp "$CONFIG_BACKUP" "$CONFIG_PATH"

    TARGET="$target" \
    SCENARIO="$scenario" \
    PROJECT_ROOT="$PROJECT_ROOT" \
    CONFIG_PATH="$CONFIG_PATH" \
    python3 - <<'EOF_PY'
import os
from pathlib import Path
from typing import Dict

import yaml


def split_target_date(mapping: Dict[str, str], target: str) -> str:
    try:
        date = mapping[target]
    except KeyError as exc:
        raise ValueError(
            "No base experiment date configured for target "
            f"{target!r}. Populate retraining.ablation.base_dates in cfg.yaml."
        ) from exc
    if not isinstance(date, str) or not date.strip():
        raise ValueError(
            "Configured base date for target "
            f"{target!r} must be a non-empty string."
        )
    return date.strip()


project_root = Path(os.environ["PROJECT_ROOT"])
config_path = Path(os.environ["CONFIG_PATH"])
target = os.environ["TARGET"]
scenario = os.environ["SCENARIO"]

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

features_cfg = cfg.setdefault("features", {})
training_cfg = cfg.setdefault("training", {})
simulation_cfg = cfg.setdefault("simulation", {})
model_cfg = cfg.setdefault("model", {})
model_cfg["transformer"] = {}
transformer_cfg = model_cfg["transformer"]
hyperopt_cfg = cfg.setdefault("hyperopt", {})
retraining_cfg = cfg.setdefault("retraining", {})
ablation_cfg = retraining_cfg.setdefault("ablation", {})
base_dates = ablation_cfg.get("base_dates", {}) if isinstance(ablation_cfg, dict) else {}
if not isinstance(base_dates, dict):
    raise ValueError("retraining.ablation.base_dates must be a mapping from target to date string.")

base_date = split_target_date(base_dates, target)
base_experiment_name = f"{target}_{base_date}"
experiment_name = f"{base_experiment_name}_{scenario}"

features_cfg["target"] = target
training_cfg["experiment_name"] = experiment_name

experiments_dir = project_root / "outputs" / "experiments"
used_config = experiments_dir / base_experiment_name / "used_config.yaml"
if not used_config.exists():
    raise FileNotFoundError(
        f"Expected used_config.yaml in {used_config.parent}, but it was not found."
    )

full_final_params = experiments_dir / base_experiment_name / "full_final_params.yaml"
if not full_final_params.exists():
    raise FileNotFoundError(
        f"Expected full_final_params.yaml in {full_final_params.parent}, but it was not found."
    )

with used_config.open("r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f) or {}

with full_final_params.open("r", encoding="utf-8") as f:
    full_params_cfg = yaml.safe_load(f) or {}

base_training = base_cfg.get("training", {})
for key in ["batch_size", "epochs", "lr", "patience", "seed"]:
    if key in base_training:
        training_cfg[key] = base_training[key]

base_features = base_cfg.get("features", {})
for key in ["monthly_vars", "quarterly_vars", "all_monthly"]:
    if key in base_features:
        features_cfg[key] = base_features[key]

base_simulation = base_cfg.get("simulation", {})
for key in ["simulate", "use_y_only_predictors"]:
    if key in base_simulation:
        simulation_cfg[key] = base_simulation[key]

simulation_cfg["simulate"] = False

base_model_params = full_params_cfg.get("model", {})
base_transformer_params = base_model_params.get("transformer", {})

for key in [
    "d_model",
    "nhead",
    "num_layers",
    "d_freq",
    "d_var",
    "dim_feedforward",
    "activation",
    "dropout",
]:
    if key in base_transformer_params:
        transformer_cfg[key] = base_transformer_params[key]

for key in [
    "use_nonlinearity",
    "use_attention",
    "use_positional_encoding",
]:
    if key in base_transformer_params:
        transformer_cfg[key] = base_transformer_params[key]
    else:
        transformer_cfg[key] = True

training_cfg["optimize"] = True

meta_keys = {"study_name", "n_trials"}
allowed = {"lr", "dropout"}
existing_spaces = {key: hyperopt_cfg.get(key) for key in allowed if key in hyperopt_cfg}
for key in list(hyperopt_cfg.keys()):
    if key not in meta_keys and key not in allowed:
        hyperopt_cfg.pop(key, None)
for key in allowed:
    if key in existing_spaces:
        hyperopt_cfg[key] = existing_spaces[key]
    else:
        raise ValueError(
            f"Hyperparameter '{key}' must be defined in cfg.yaml hyperopt section before running ablations."
        )

hyperopt_cfg["study_name"] = f"{experiment_name}_study"

if scenario == "B1_no_nonlinearity":
    transformer_cfg["use_nonlinearity"] = False
elif scenario == "B2_no_attention":
    transformer_cfg["use_attention"] = False
elif scenario == "B3_no_attention_no_nonlinearity":
    transformer_cfg["use_attention"] = False
    transformer_cfg["use_nonlinearity"] = False
elif scenario == "B5_no_positional_encoding":
    transformer_cfg["use_positional_encoding"] = False
elif scenario == "B6_y_only":
    features_cfg["all_monthly"] = False
    features_cfg["monthly_vars"] = []
else:
    raise ValueError(f"Unknown ablation scenario: {scenario}")

with config_path.open("w", encoding="utf-8") as f:
    yaml.dump(cfg, f, sort_keys=False)

print(f"Configured ablation run using base experiment '{base_experiment_name}'.")
print(f"New experiment name: {experiment_name}")
EOF_PY

    echo "==============================="
    echo "Running ablation ($scenario) for target: $target"
    echo "==============================="

    python3 "$PYTHON_RUNNER"
  done

done
