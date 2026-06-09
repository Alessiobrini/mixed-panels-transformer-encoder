#!/bin/bash
#
# Collect + aggregate the Monte Carlo replications (referee R2-12) after the array job.
# Builds the unified manifest by scanning completed experiment folders (robust to any
# tasks that failed/were preempted -- aggregates over whatever finished), then writes the
# mean +/- SD table. Submit with a dependency on the array job:
#
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> src/slurm_aggregate_replications.sh
#
# (afterany, not afterok: aggregate over completed seeds even if a few tasks failed; just
#  re-submit the array to fill gaps, then re-run this.)
#
#SBATCH --job-name=mpte-repl-agg
#SBATCH --partition=common
#SBATCH --mem=8G
#SBATCH --output=outputs/replications/slurm/agg_%j.out
#SBATCH --error=outputs/replications/slurm/agg_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=alessio.brini@duke.edu

set -e
set -o pipefail

source ~/.bashrc
conda activate tsa-dev

PROJECT_ROOT="/hpc/group/darec/ab978/mixed-panels-transformer-encoder"
cd "$PROJECT_ROOT"

# 1) Build the unified manifest by scanning the full N=100 plan for completed units.
python3 src/run_simulation_replications.py \
  --regimes all --n 100 --start-seed 2000 \
  --study path --init-seed 125 --k-inits 5 \
  --tag repl100 \
  --bases-dir src/config/sim_replication_bases \
  --collect-only

# 2) Aggregate to mean +/- SD (CSV + LaTeX) under outputs/tables/.
python3 src/evaluation/aggregate_replications.py \
  --manifest outputs/replications/repl100_manifest.csv

echo "=== Aggregation done. See outputs/tables/repl100_replications.{tex,csv} ==="
