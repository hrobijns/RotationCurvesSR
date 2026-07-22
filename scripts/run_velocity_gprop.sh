#!/bin/bash
# Production run (v3): joint SR of f(x, γ) and γ = g(x_M, x_f, x_S, x_R) on the
# selected SPARC sample — lognormal γ prior (σ_int = 0.05 dex), constants OFF
# (atom rationals), inner-opt settings accuracy-gated by diagnose_inner_opt.
# Laptop-sized (8 threads): populations=16 saturates cores (16 evals/s-class
# throughput on the 115-gal toy; planted (f, γ-law) recovered in 2 h there).
# No timeout — monitor the HOF and kill when the front stalls (<1-2%/12 h).
cd "$(dirname "$0")/.."
mkdir -p outputs/production/gprop_v3
nohup .venv/bin/python -c "
import sys, os
sys.path.insert(0, '.')
from datawrangling import produce_SPARC_df
from models.velocity_gprop import fit_vr_gprop
df = produce_SPARC_df('data/SPARC', selected=True)
fit_vr_gprop(
    df,
    output_directory='outputs/production/gprop_v3',
    iterations=99999,
    n_galaxies=None,
    populations=16,
    population_size=30,
    ncycles_per_iteration=100,
    unary_operators=['atan', 'log1p'],
)
" > outputs/production/velocity_gprop_v3.log 2>&1 &
echo "velocity_gprop v3 PID: $!"
