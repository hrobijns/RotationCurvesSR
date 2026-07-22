#!/bin/bash
# One-time environment setup on a CSD3 LOGIN node (needs internet — compute
# nodes may not have it). Run this once before submitting any sbatch job.
#
#   ssh <user>@login-cpu.hpc.cam.ac.uk
#   git clone --depth 1 https://github.com/hrobijns/rotationcurvesSR.git ~/rds/hpc-work/rotationcurvesSR
#   cd ~/rds/hpc-work/rotationcurvesSR
#   bash scripts/hpc/setup_env.sh
# (private repo — clone will prompt for a GitHub username + PAT, or use an
# SSH remote / `gh auth login` if set up on the login node.)
#
set -euo pipefail
cd "$(dirname "$0")/../.."

module purge
module load miniconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n sparc-sr python=3.12
conda activate sparc-sr

pip install -e .

# Force Julia (juliaup) download + SymbolicRegression.jl precompilation NOW,
# while the login node still has internet — compute nodes may be offline.
# Reuses the existing tiny toy runners; kill after the HOF file appears
# (precompilation is done once it starts printing iterations).
timeout 600 python scripts/run_v2p_toy.py || true
timeout 600 python scripts/run_gprop_toy.py || true

echo "Setup done. Julia depot: $(python -c 'import os; print(os.environ.get("JULIA_DEPOT_PATH", "~/.julia"))')"
echo "Next: edit -A ACCOUNT_CODE in scripts/hpc/slurm_*.sh (see 'mybalance'), then sbatch them."
