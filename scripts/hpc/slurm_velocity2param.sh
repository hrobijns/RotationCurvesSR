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
# Mega run: velocity2param on a full icelake node (76 cores). SL3 caps jobs at
# 12h / 448 cores cluster-wide — this alone (76 cores) is well within budget,
# but the WALLCLOCK CAP means one sbatch won't finish a long search. Chain
# 12h blocks via warm_start + PySR checkpoint.pkl (run_id fixed below) using:
#   bash scripts/hpc/submit_chain.sh scripts/hpc/slurm_velocity2param.sh 4
# (single submit: sbatch scripts/hpc/slurm_velocity2param.sh)
# Fill in ACCOUNT_CODE above first (see `mybalance` for your project codes).
# Run scripts/hpc/setup_env.sh once before this (needs the conda env + a
# precompiled Julia depot already in place).
# NOTE: PySR warns that resuming a TemplateExpressionSpec run from checkpoint
# is "not fully supported" — this project relies on it, so sanity-check a
# resume on the toy runner (scripts/run_v2p_toy.py) before trusting a long
# chain unattended.

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
    warm_start=True,
    checkpoint_every=1,
)
"
