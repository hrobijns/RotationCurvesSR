#!/bin/bash
#SBATCH -J v2param
#SBATCH -A ACCOUNT_CODE
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=11:45:00
#SBATCH --output=outputs/production/hpc_v2param_%j.log
#
# Single-shot run: velocity2param on a full icelake node (76 cores), for up
# to SL3's ~12h wall-clock cap (11:45 to leave margin for job-start/module
# overhead). Not chained/resumable — when the time limit hits, this run is
# done; results are whatever's in hall_of_fame.csv at that point.
#
# First time only: fill in ACCOUNT_CODE above (see `mybalance` for your
# project codes) and run scripts/hpc/setup_env.sh once (conda env + a
# precompiled Julia depot).
#
# Submit with: sbatch scripts/hpc/slurm_velocity2param.sh

set -euo pipefail
# SLURM copies submitted scripts into a spool dir before exec, so $0-based
# path tricks land in the wrong place; SLURM_SUBMIT_DIR is set to wherever
# sbatch was actually invoked from and is the reliable way back to the repo.
cd "$SLURM_SUBMIT_DIR"
module purge
module load miniconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sparc-sr

mkdir -p outputs/production/thesis_hpc

python -c "
from datawrangling import produce_SPARC_df
from models.velocity2param import fit_vr_2param

df = produce_SPARC_df('data/SPARC', selected=True)
fit_vr_2param(
    df,
    output_directory='outputs/production/thesis_hpc',
    iterations=99999,
    n_galaxies=None,
    n_start_d=5,
    n_start_gamma=5,
    gamma_range=(0.0, 1.0),
    populations=160,
    population_size=50,
    ncycles_per_iteration=200,
    weight_optimize=0.0,
    optimizer_iterations=0,
    optimizer_niter=100,
    unary_operators=['atan', 'log1p'],
    origin_x=1e-6,
    maxsize=25,
    procs=76,
    run_id='v2param_prod',
)
"
