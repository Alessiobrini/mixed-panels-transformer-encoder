#!/bin/bash
#
# Monte Carlo replications for the simulation study (referee R2-12), as a SLURM array.
# One array task per simulation seed; each task runs ALL regimes x variants for that
# seed (study=path: init seed fixed at 125, simulation seed varied). k=5 best-of-val.
#
#   array index i  ->  simulation seed = 2000 + i   (i = 0..99  =>  N = 100 seeds)
#
# Per-seed intermediate files are seed-suffixed by the driver, so concurrent tasks do
# not collide on the shared filesystem. Tasks write into outputs/experiments/<exp_name>
# (unique per seed+variant). The driver is resumable: re-submitting skips finished units.
#
# Submit:
#   mkdir -p outputs/replications/slurm
#   AID=$(sbatch --parsable src/slurm_replications.sh)
#   sbatch --dependency=afterany:$AID src/slurm_aggregate_replications.sh   # aggregate after
#
#SBATCH --job-name=mpte-repl
#SBATCH --partition=gpu-common
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --array=0-99%20
#SBATCH --output=outputs/replications/slurm/repl_%A_%a.out
#SBATCH --error=outputs/replications/slurm/repl_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=alessio.brini@duke.edu

set -o pipefail

source ~/.bashrc
conda activate tsa-dev
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # required for torch deterministic algos on CUDA

PROJECT_ROOT="/hpc/group/darec/ab978/mixed-panels-transformer-encoder"
cd "$PROJECT_ROOT"
mkdir -p outputs/replications/slurm data/processed outputs/experiments

RSCRIPT="/hpc/group/darec/ab978/miniconda3/envs/tsa-dev/bin/Rscript"
SEED=$((2000 + SLURM_ARRAY_TASK_ID))

echo "=== Replication task: sim_seed=${SEED} (array ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

# One seed, all regimes x variants. workers=1 (one GPU per task). --no-manifest avoids
# manifest write races across tasks; the unified manifest is built later by the aggregate
# job via --collect-only. --bases-dir uses the committed tuned-HP bundle (outputs/ is gitignored).
python3 src/run_simulation_replications.py \
  --seeds "${SEED}" \
  --regimes all --variants all \
  --study path --init-seed 125 \
  --k-inits 5 \
  --workers 1 \
  --tag repl100 \
  --bases-dir src/config/sim_replication_bases \
  --rscript "${RSCRIPT}" \
  --no-manifest

echo "=== Done sim_seed=${SEED} ==="
