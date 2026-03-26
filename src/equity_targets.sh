#!/bin/bash
#
#SBATCH --job-name=equity-experiments
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --output=slurm_equity.out
#SBATCH --error=slurm_equity.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=alessio.brini@duke.edu

# Usage:
#   sbatch src/equity_targets.sh                        # default: gpu-common
#   sbatch -p scavenger-gpu src/equity_targets.sh       # override partition

set -e
set -o pipefail

source ~/.bashrc
conda activate tsa-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PROJECT_ROOT="/hpc/group/darec/ab978/mixed-panels-transformer-encoder"
cd "$PROJECT_ROOT"

# Run all 40 tickers sequentially
python3 src/run_equity_pipeline.py --config src/config/cfg_equity.yaml
