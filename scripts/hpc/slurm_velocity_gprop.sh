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
# Mega run: velocity_gprop (v3) on a full icelake node (76 cores). SL3 caps
# jobs at 12h / 448 cores cluster-wide — running this alongside
# slurm_velocity2param.sh is 152 cores total, well within budget, but the
# WALLCLOCK CAP means one sbatch won't finish a long search. Chain 12h blocks
# via warm_start + PySR checkpoint.pkl (run_id fixed below) using:
#   bash scripts/hpc/submit_chain.sh scripts/hpc/slurm_velocity_gprop.sh 4
# (single submit: sbatch scripts/hpc/slurm_velocity_gprop.sh)
# Fill in ACCOUNT_CODE above first (see `mybalance` for your project codes).
# Run scripts/hpc/setup_env.sh once before this.
# Safe to submit alongside slurm_velocity2param.sh — separate physical nodes,
# no shared Julia runtime state, so the "one PySR process at a time" rule
# (same-machine GC corruption) doesn't apply across nodes.
# NOTE: PySR warns that resuming a TemplateExpressionSpec run from checkpoint
# is "not fully supported" — this project relies on it, so sanity-check a
# resume on the toy runner (scripts/run_gprop_toy.py) before trusting a long
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
    warm_start=True,
)
"
