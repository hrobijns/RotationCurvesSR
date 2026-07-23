#!/bin/bash
#SBATCH -J gprop
#SBATCH -A ACCOUNT_CODE
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=11:45:00
#SBATCH --output=outputs/production/hpc_gprop_%j.log
#
# Single-shot run: velocity_gprop (v3) on a full icelake node (76 cores),
# for up to SL3's ~12h wall-clock cap (11:45 to leave margin for
# job-start/module overhead). Not chained/resumable — when the time limit
# hits, this run is done; results are whatever's in hall_of_fame.csv at
# that point.
#
# Safe to submit alongside slurm_velocity2param.sh — separate physical
# nodes, no shared Julia runtime state, so the "one PySR process at a time"
# rule (same-machine GC corruption) doesn't apply across nodes. Together
# that's 152 cores, well within SL3's 448-core cluster-wide cap.
#
# First time only: fill in ACCOUNT_CODE above (see `mybalance` for your
# project codes) and run scripts/hpc/setup_env.sh once (conda env + a
# precompiled Julia depot).
#
# Submit with: sbatch scripts/hpc/slurm_velocity_gprop.sh

set -euo pipefail
# SLURM copies submitted scripts into a spool dir before exec, so $0-based
# path tricks land in the wrong place; SLURM_SUBMIT_DIR is set to wherever
# sbatch was actually invoked from and is the reliable way back to the repo.
cd "$SLURM_SUBMIT_DIR"
module purge
module load miniconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sparc-sr

mkdir -p outputs/production/gprop_v3_hpc

python -c "
from datawrangling import produce_SPARC_df
from models.velocity_gprop import fit_vr_gprop

df = produce_SPARC_df('data/SPARC', selected=True)
fit_vr_gprop(
    df,
    output_directory='outputs/production/gprop_v3_hpc',
    iterations=99999,
    n_galaxies=None,
    populations=160,
    population_size=30,
    ncycles_per_iteration=100,
    unary_operators=['atan', 'log1p'],
    procs=76,
    run_id='gprop_v3_prod',
)
"
