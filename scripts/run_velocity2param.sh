#!/bin/bash
# Production run: 2-parameter velocity model f(x, γ) on the selected SPARC sample.
cd "$(dirname "$0")/.."
mkdir -p outputs/production/thesis
nohup .venv/bin/python -c "
import sys, os
sys.path.insert(0, '.')
from datawrangling import produce_SPARC_df
from models.velocity2param import fit_vr_2param
df = produce_SPARC_df('data/SPARC', selected=True)
fit_vr_2param(
    df,
    output_directory='outputs/production/thesis',
    iterations=99999,
    n_galaxies=None,
    n_start_d=5,
    n_start_gamma=5,
    gamma_range=(0.0, 1.0),
    populations=72,
    population_size=50,
    ncycles_per_iteration=200,
    weight_optimize=0.0,
    optimizer_iterations=0,
    optimizer_niter=100,
    unary_operators=['atan', 'log1p'],
    origin_x=1e-6,
    maxsize=25,
    procs=24,
)
" > outputs/production/velocity2param.log 2>&1 &
echo "velocity2param PID: $!"
