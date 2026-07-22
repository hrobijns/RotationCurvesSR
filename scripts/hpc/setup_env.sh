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

# --override-channels + conda-forge: the HPC miniconda module's default
# ('defaults') channel frequently can't solve a consistent python=3.12 build
# set on these nodes (UnsatisfiableError on libzlib/openssl/expat versions).
conda create -y -n sparc-sr -c conda-forge --override-channels python=3.12
conda activate sparc-sr

# -u LD_LIBRARY_PATH: the miniconda module's conda-forge openssl shadows the
# system libcrypto that git's Kerberos support (libk5crypto) was built
# against, breaking `git clone` over https (symbol lookup error:
# EVP_KDF_ctrl) for pip's git-based pysr dependency below.
env -u LD_LIBRARY_PATH pip install -e .

# models/losses/*.jl (velocity2param, velocity_gprop) call Optim.jl for the
# inner d/gamma optimisation (Fminbox+NelderMead) — not a PySR dependency,
# so juliapkg never installs it. Add it to the same juliapkg-managed Julia
# project PySR uses. NOTE: no `env -u LD_LIBRARY_PATH` here — Julia's own
# libjulia-internal.so needs the conda env's newer libstdc++ (system
# /lib64/libstdc++.so.6 is too old: missing GLIBCXX_3.4.26), and Pkg.add
# uses Julia's bundled libgit2 rather than the system `git` binary, so it
# doesn't hit the openssl/krb5 clash that `pip install -e .` above did.
python -c "from juliacall import Main as jl; jl.seval('import Pkg; Pkg.add(\"Optim\")')"

# Force Julia (juliaup) download + SymbolicRegression.jl precompilation NOW,
# while the login node still has internet — compute nodes may be offline.
# Reuses the existing tiny toy runners; kill after the HOF file appears
# (precompilation is done once it starts printing iterations). Keep
# LD_LIBRARY_PATH set here too, same libstdc++ reasoning as above.
timeout 600 python scripts/run_v2p_toy.py || true
timeout 600 python scripts/run_gprop_toy.py || true

echo "Setup done. Julia depot: $(python -c 'import os; print(os.environ.get("JULIA_DEPOT_PATH", "~/.julia"))')"
echo "Next: edit -A ACCOUNT_CODE in scripts/hpc/slurm_*.sh (see 'mybalance'), then sbatch them."
