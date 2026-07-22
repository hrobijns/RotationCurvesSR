"""Toy validation for models/velocity_gprop.py.

Data: data/toydatasets/gprop_midnoise.csv (make_toy_gprop.py) — real SPARC
baryons + injected DM with f(x,γ) = x^(-γ)·(x - atan(x))/x and planted
relation γ_true = 0.5 - 0.2·z_Mstar. Success: g linear in #1 (zMstar) with
negative slope on the Pareto front, f in the atan family.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from models.velocity_gprop import fit_vr_gprop

df = pd.read_csv("data/toydatasets/gprop_midnoise.csv")

fit_vr_gprop(
    df,
    output_directory="outputs/toydatasets/gprop_toy",
    iterations=99999,
    n_galaxies=20,
    n_start_d=5,
    n_start_gamma=3,
    gamma_range=(0.0, 1.0),
    populations=10,
    population_size=30,
    ncycles_per_iteration=100,
    weight_optimize=0.0,
    optimizer_iterations=0,
    unary_operators=["atan", "log1p"],
    maxsize=28,
)
