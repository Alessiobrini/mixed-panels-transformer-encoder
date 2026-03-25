#!/bin/bash
#
#SBATCH --job-name=equity-experiments
#SBATCH --partition=gpu-common
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --output=slurm_equity.out
#SBATCH --error=slurm_equity.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=alessio.brini@duke.edu

set -e
set -o pipefail

source ~/.bashrc
conda activate tsa-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PROJECT_ROOT="/hpc/group/darec/ab978/tsa-dev"
cd "$PROJECT_ROOT"

# Run all 40 tickers sequentially
python3 src/run_equity_pipeline.py --config src/config/cfg_equity.yaml
