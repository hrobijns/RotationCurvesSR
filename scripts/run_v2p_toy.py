import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.velocity2param import fit_vr_2param
from datawrangling import produce_SPARC_df

df_real = produce_SPARC_df("data/SPARC", selected=True)

fit_vr_2param(
    df_real,
    output_directory="outputs/toydatasets/v2p_toy",
    n_galaxies=5,
    n_start_d=5,
    n_start_gamma=5,
    populations=30,
    population_size=50,
    ncycles_per_iteration=200,
    n_irls=4,
    optimizer_niter=50,
    unary_operators=["atan", "log1p"],
    maxsize=20,
    error_weighting=True,
    upsilon_weight=0.1
)
