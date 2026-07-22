from pysr import PySRRegressor, TemplateExpressionSpec
import pandas as pd
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.julia_losses import load_loss


def fit_vr_inner_opt(df: pd.DataFrame,
                     output_directory: str = "outputs",
                     error_weighting: bool = True,
                     iterations: int = 99999,
                     n_galaxies: int | None = 5,
                     n_d_grid: int = 50,
                     n_d_refine: int = 20,
                     populations: int = 15,
                     population_size: int = 40,
                     ncycles_per_iteration: int = 100,
                     weight_optimize: float = 0.1,
                     optimizer_iterations: int = 8,
                     upsilon_weight: float = 1.0,
                     origin_weight: float = 1.0,
                     weight_nonneg: float = 0.01,
                     weight_mono: float = 0.01,
                     nu_t: float = 3.0,
                     n_irls: int = 5,
                     min_points: int = 5,
                     unary_operators: list = None,
                     maxsize: int = 22,
                     timeout_in_seconds: float | None = None,
                     procs: int = 0):
    """
    Symbolic regression for galaxy rotation curves using per-expression inner
    optimisation: for each candidate symbolic form f, the per-galaxy parameters
    (upsilon_disk a, upsilon_bulge b, DM amplitude c, scale radius d) are found
    analytically / via 1-D grid search.

    Model structure (per data point i in galaxy g):
        V²_obs = sign(Vgas)·V²_gas + a[g]·V²_disk + b[g]·V²_bul + c[g]·f(r/d[g])

    For fixed d[g], the model is LINEAR in a[g], b[g], c[g].  We solve via
    Iteratively Reweighted Least Squares (IRLS) for a Student-t(ν=nu_t) likelihood,
    using measurement-error weights w_i = 1/(2·Vobs_i·errV_i)².  The IRLS starts
    from a Gaussian WLS solution and refines it over n_irls iterations.  d[g] is
    chosen by a log-spaced grid search.  Galaxies are independent.

    Log-normal priors on upsilon_disk and upsilon_bulge (matching mcmc_fit.py:
    lnN(ln 0.5, 0.1 dex) and lnN(ln 0.7, 0.1 dex)) are added as penalty terms
    weighted by upsilon_weight (default 1.0 = always active).
    """
    if n_galaxies is not None:
        available = df["galaxy"].unique()
        rng = np.random.default_rng(seed=42)
        selected = rng.choice(available, size=min(n_galaxies, len(available)), replace=False)
        df = df[df["galaxy"].isin(selected)].copy()

    # drop galaxies with too few data points for the 3-parameter solve to be stable
    counts = df.groupby("galaxy")["galaxy"].transform("count")
    df = df[counts >= min_points].copy()

    galaxy_codes, galaxies = pd.factorize(df["galaxy"])
    df["galaxy_code"] = galaxy_codes + 1  # Julia is 1-indexed
    
    df = df.sort_values("galaxy_code").reset_index(drop=True)
    # sort by galaxy_code so Julia can use O(n_total) index scan instead of
    # O(n_gal × n_total) BitVector masks.

    y = df["Vobs_km/s"] ** 2
    X = pd.DataFrame({
        "r":       df["Rad_kpc"],
        "galaxy":  df["galaxy_code"],
        "Vgas2":   np.sign(df["Vgas_km/s"]) * df["Vgas_km/s"] ** 2,
        "Vdisk2":  df["Vdisk_km/s"] ** 2,
        "Vbulge2": df["Vbul_km/s"] ** 2,
    })

    # Custom loss (models/losses/velocity1param.jl): for each candidate f,
    # find optimal (a, b, c, d) per galaxy — d by log-grid search + local
    # refinement, (a, b, c) by WLS → Student-t IRLS.
    inner_opt_loss = load_loss(
        "velocity1param",
        nu_t=nu_t,
        upsilon_weight=upsilon_weight,
        n_irls=n_irls,
        n_d_grid=n_d_grid,
        n_d_refine=n_d_refine,
        origin_weight=origin_weight,
        weight_nonneg=weight_nonneg,
        weight_mono=weight_mono,
    )

    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["r", "galaxy", "Vgas2", "Vdisk2", "Vbulge2"],
        # No per-galaxy parameters: they are found analytically in the custom loss
        # and never used by the Julia template machinery.  Removing them eliminates
        # the SVector{N_galaxies} Julia type whose compilation scales super-linearly.
        combine="f(r)",   # display: raw DM profile shape f(r/d) ← d recovered post-hoc
    )

    active_unary = unary_operators if unary_operators is not None else ["atan", "log1p"]
    nested = {
        op: {other: 0 for other in active_unary}
        for op in active_unary
    }

    model = PySRRegressor(
        expression_spec=template,
        output_directory=output_directory,
        niterations=iterations,
        binary_operators=["*", "/", "-", "+"],
        unary_operators=active_unary,
        nested_constraints=nested,
        maxsize=maxsize,
        populations=populations,
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        weight_optimize=weight_optimize,   # If > 0, BFGS tunes constants *within* f
                                           # (e.g. the "1" in 1−atan(x)/x).
                                           # Template params a,b,c,d have zero gradient
                                           # in this loss so BFGS leaves them untouched.
        optimizer_iterations=optimizer_iterations,
        batching=False,
        turbo=True,
        complexity_of_constants=3,
        timeout_in_seconds=timeout_in_seconds,
        procs=procs,
        loss_function_expression=inner_opt_loss,
    )

    if error_weighting:
        errV_safe = np.maximum(df["errV_km/s"], 0.5)
        weights = 1.0 / (2 * df["Vobs_km/s"] * errV_safe) ** 2
    else:
        weights = np.ones(len(df))
    # Always pass weights so dataset.weights is never `nothing` in Julia.
    model.fit(X, y, weights=weights)


def recover_parameters(df: pd.DataFrame,
                       f_callable,
                       n_d_grid: int = 200,
                       upsilon_weight: float = 0.0) -> pd.DataFrame:
    """
    Post-hoc parameter recovery for the best symbolic form found by fit_vr_inner_opt.

    Given the discovered DM profile shape as a Python callable f(x) (where x = r/d),
    re-runs the same inner optimisation (log-spaced d-grid + weighted OLS) in Python
    to find the optimal (upsilon_disk, upsilon_bulge, c_DM, d_kpc) per galaxy.

    Returns a DataFrame with columns galaxy, upsilon_disk, upsilon_bulge, c_DM, d_kpc, wls
    — directly comparable to outputs/fits/pISO.csv from mcmc_fit.py.

    Example
    -------
    >>> f = lambda x: (x - np.arctan(x)) / x   # best complexity-6 expression
    >>> params = recover_parameters(df, f)
    """
    results = []
    for g, gdf in df.groupby("galaxy"):
        r     = gdf["Rad_kpc"].values.astype(float)
        y     = gdf["Vobs_km/s"].values ** 2
        vgas2 = np.sign(gdf["Vgas_km/s"].values) * gdf["Vgas_km/s"].values ** 2
        vd2   = gdf["Vdisk_km/s"].values ** 2
        vb2   = gdf["Vbul_km/s"].values ** 2
        errV  = np.maximum(gdf["errV_km/s"].values, 0.5)
        w     = 1.0 / (2 * gdf["Vobs_km/s"].values * errV) ** 2
        resid = y - vgas2

        r_max_g = r.max()
        d_min_g = r_max_g / 20
        d_max_g = max(r_max_g * 10, 50.0)
        best = (np.inf, None)
        for d in np.exp(np.linspace(np.log(d_min_g), np.log(d_max_g), n_d_grid)):
            fv = f_callable(r / d)
            if not np.all(np.isfinite(fv)):
                continue
            A = np.column_stack([vd2, vb2, fv])
            if np.max(np.abs(A)) < 1e-10:
                continue
            try:
                A_w = (w ** 0.5)[:, None] * A
                r_w = (w ** 0.5) * resid
                abc, *_ = np.linalg.lstsq(A_w, r_w, rcond=None)
                abc = np.maximum(abc, 0.0)
                wls = float(np.sum(w * (A @ abc - resid) ** 2) / len(r))
                if wls < best[0]:
                    best = (wls, dict(galaxy=g, upsilon_disk=abc[0],
                                      upsilon_bulge=abc[1], c_DM=abc[2],
                                      d_kpc=d, wls=wls))
            except Exception:
                pass

        if best[1] is not None:
            results.append(best[1])

    return pd.DataFrame(results)


if __name__ == "__main__":
    from datawrangling import produce_SPARC_df
    df = produce_SPARC_df("data/SPARC", selected=True)
    fit_vr_inner_opt(
        df,
        output_directory="outputs/SPARC/production/velocity1param",
        iterations=99999,
        n_galaxies=None,
        n_d_grid=50,
        populations=20,
        population_size=40,
        ncycles_per_iteration=100,
        weight_optimize=0.1,
        optimizer_iterations=8,
        unary_operators=["atan", "log1p"],
        maxsize=22,
    )
