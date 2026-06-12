#!/bin/bash
#
# FRED ablations (AB1..AB5) WITH the high-frequency lead, as a SLURM array.
# 13 targets x 5 scenarios = 65 tasks (array 0..64), one per GPU, run concurrently.
# Each task reads the matching full-model lead2 run ({target}_{DATETAG}) for its tuned
# architecture (full_final_params.yaml) and base settings (used_config.yaml), fixes the
# architecture, re-tunes ONLY lr+dropout with n_trials=20 (matching the published
# ablations), applies the ablation flag, and runs the pipeline via run_pipeline.py --config.
# Per-task configs + absolute namespaced paths => no collision across concurrent tasks.
#
# Requires the full-model array (univariate_targets_lead.sh) to have finished first:
#   sbatch --dependency=afterok:<F1_JOBID> --array=0-64 src/univariate_targets_ablation_lead.sh
#
#SBATCH --job-name=fred-abl-lead2
#SBATCH --partition=gpu-common
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --array=0-64
#SBATCH --output=outputs/replications/slurm/fredabl_%A_%a.out
#SBATCH --error=outputs/replications/slurm/fredabl_%A_%a.err
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
ABL_TRIALS="${ABL_TRIALS:-20}"

targets=("GDPC1" "GPDIC1" "PCECC96" "DPIC96" "OUTNFB" "UNRATE" "PCECTPI" "PCEPILFE" "CPIAUCSL" "CPILFESL" "FPIx" "EXPGSC1" "IMPGSC1")
scenarios=("B1_no_nonlinearity" "B2_no_attention" "B3_no_attention_no_nonlinearity" "B5_no_positional_encoding" "B6_y_only")

T_IDX=$(( SLURM_ARRAY_TASK_ID / ${#scenarios[@]} ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % ${#scenarios[@]} ))
TARGET="${targets[$T_IDX]}"
SCENARIO="${scenarios[$S_IDX]}"
BASE_EXP="${TARGET}_${DATETAG}"
EXP_NAME="${BASE_EXP}_${SCENARIO}"
RUNTIME_CFG="src/config/_lead_runtime/cfg_${EXP_NAME}.yaml"

echo "=== FRED ablation: target=${TARGET} scenario=${SCENARIO} base=${BASE_EXP} on $(hostname) ==="

TARGET="$TARGET" SCENARIO="$SCENARIO" BASE_EXP="$BASE_EXP" EXP_NAME="$EXP_NAME" \
RUNTIME_CFG="$RUNTIME_CFG" PROJECT_ROOT="$PROJECT_ROOT" ABL_TRIALS="$ABL_TRIALS" python3 - <<'EOF'
import os, yaml
from pathlib import Path

root = os.environ["PROJECT_ROOT"]
tgt = os.environ["TARGET"]
scenario = os.environ["SCENARIO"]
base_exp = os.environ["BASE_EXP"]
exp_name = os.environ["EXP_NAME"]
n_trials = int(os.environ["ABL_TRIALS"])

cfg = yaml.safe_load(Path("src/config/cfg.yaml").read_text())
exp_dir = Path(root) / "outputs" / "experiments" / base_exp
base_cfg = yaml.safe_load((exp_dir / "used_config.yaml").read_text())
full_params = yaml.safe_load((exp_dir / "full_final_params.yaml").read_text())

cfg.setdefault("features", {})["target"] = tgt
cfg.setdefault("training", {})["experiment_name"] = exp_name
cfg.setdefault("evaluation", {})["experiment"] = exp_name

# Inherit base run settings (data/training/features), as the published ablation script does.
for k in ["batch_size", "epochs", "lr", "patience", "seed"]:
    if k in base_cfg.get("training", {}):
        cfg["training"][k] = base_cfg["training"][k]
cfg.setdefault("data", {})
for k in ["context_days", "train_ratio", "val_ratio"]:
    if k in base_cfg.get("data", {}):
        cfg["data"][k] = base_cfg["data"][k]
cfg["data"]["lead"] = int(base_cfg.get("data", {}).get("lead", 2))  # PRESERVE the lead
for k in ["monthly_vars", "quarterly_vars", "all_monthly"]:
    if k in base_cfg.get("features", {}):
        cfg["features"][k] = base_cfg["features"][k]
cfg.setdefault("simulation", {})["simulate"] = False

# Fix architecture to the tuned full-model params; only lr+dropout are re-tuned.
cfg.setdefault("model", {})["transformer"] = dict(full_params)
for k in ["use_nonlinearity", "use_attention", "use_positional_encoding"]:
    cfg["model"]["transformer"].setdefault(k, True)
cfg["model"]["transformer"].setdefault("calendar_pe", False)

cfg["training"]["optimize"] = True
ho = cfg.setdefault("hyperopt", {})
keep = {kk: ho.get(kk) for kk in ("lr", "dropout") if kk in ho}
for kk in list(ho):
    if kk not in ("study_name", "n_trials"):
        ho.pop(kk, None)
ho.update(keep)
ho["n_trials"] = n_trials
ho["study_name"] = f"{exp_name}_study"

# Apply the ablation flag.
tr = cfg["model"]["transformer"]
if scenario == "B1_no_nonlinearity":
    tr["use_nonlinearity"] = False
elif scenario == "B2_no_attention":
    tr["use_attention"] = False
elif scenario == "B3_no_attention_no_nonlinearity":
    tr["use_attention"] = False; tr["use_nonlinearity"] = False
elif scenario == "B5_no_positional_encoding":
    tr["use_positional_encoding"] = False
elif scenario == "B6_y_only":
    cfg["features"]["use_y_only_predictors"] = True

# Absolute, per-experiment namespaced intermediate paths (avoid cross-task collision; midas.R setwd).
cfg["paths"]["data_processed_template"] = f"{root}/data/processed/long_format_fred_{exp_name}_{{suffix}}.csv"
for k in ["transformer_preds", "ar_preds", "midas_preds"]:
    cfg["paths"]["outputs"][k] = f"{root}/outputs/{k}_{exp_name}_{{suffix}}.csv"

Path(os.environ["RUNTIME_CFG"]).write_text(yaml.dump(cfg, sort_keys=False))
print("wrote", os.environ["RUNTIME_CFG"], "| n_trials", n_trials, "| flags", {k: tr.get(k) for k in ["use_nonlinearity","use_attention","use_positional_encoding"]}, "| y_only", cfg["features"].get("use_y_only_predictors"))
EOF

python3 src/run_pipeline.py --config "$RUNTIME_CFG" --rscript "$RSCRIPT"

echo "=== Done ${EXP_NAME} ==="
