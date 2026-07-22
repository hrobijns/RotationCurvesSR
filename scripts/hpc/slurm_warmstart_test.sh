#!/bin/bash
#SBATCH -J wstest
#SBATCH -A ACCOUNT_CODE
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=outputs/hpc_test/warmstart_%j.log
#
# Throwaway test: confirms PySR checkpoint/warm_start actually resumes (not
# silently restarts) for a TemplateExpressionSpec model, before trusting a
# real chain unattended. Small pop, short time, separate run_id/output_dir —
# cannot collide with the production v2param_prod / gprop_v3_prod runs.
#
# Submit as a 2-link chain and diff the logs:
#   bash scripts/hpc/submit_chain.sh scripts/hpc/slurm_warmstart_test.sh 2
#
# PASS: link 2's log prints "[resume] loading checkpoint from
# outputs/hpc_test/warmstart_test/checkpoint.pkl" and its hall-of-fame
# continues improving on/past link 1's, not regressing at matching
# complexities. checkpoint_every=5: PySR only checkpoints before/after a
# fit() call, so this loops fit() in small chunks — a real, resumable
# checkpoint gets written every 5 iterations instead of only at the (never
# reached, given niterations=99999) end.
# FAIL: no "[resume]" line, or hall-of-fame clearly restarts from scratch —
# fall back to independent 11:45 replicate runs instead of chaining.

set -euo pipefail
# SLURM copies submitted scripts into a spool dir before exec, so $0-based
# path tricks land in the wrong place; SLURM_SUBMIT_DIR is set to wherever
# sbatch was actually invoked from and is the reliable way back to the repo.
cd "$SLURM_SUBMIT_DIR"
module purge
module load miniconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sparc-sr

mkdir -p outputs/hpc_test

python -c "
from datawrangling import produce_SPARC_df
from models.velocity2param import fit_vr_2param

df = produce_SPARC_df('data/SPARC', selected=True)
fit_vr_2param(
    df,
    output_directory='outputs/hpc_test',
    iterations=99999,
    n_galaxies=10,
    n_start_d=5,
    n_start_gamma=5,
    populations=8,
    population_size=20,
    ncycles_per_iteration=100,
    weight_optimize=0.0,
    optimizer_iterations=0,
    unary_operators=['atan', 'log1p'],
    maxsize=20,
    procs=8,
    run_id='warmstart_test',
    warm_start=True,
    checkpoint_every=5,
)
"
