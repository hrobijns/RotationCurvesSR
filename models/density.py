from pysr import PySRRegressor, TemplateExpressionSpec
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.julia_losses import load_loss


def fit_density_inner_opt(df: pd.DataFrame,
                          output_directory: str = "outputs",
                          error_weighting: bool = True,
                          iterations: int = 99999,
                          n_galaxies: int | None = 5,
                          n_d_grid: int = 50,
                          populations: int = 15,
                          population_size: int = 40,
                          ncycles_per_iteration: int = 500,
                          weight_optimize: float = 0.1,
                          optimizer_iterations: int = 8,
                          upsilon_weight: float = 1.0,
                          weight_nonneg: float = 0.1,
                          weight_slope: float = 0.1,
                          nu_t: float = 3.0,
                          n_irls_coarse: int = 1,
                          n_irls_fine: int = 5,
                          min_points: int = 5,
                          unary_operators: list[str] | None = None,
                          guesses: list | None = None,
                          fraction_replaced_guesses: float = 0.001,
                          procs: int = 0):
    """
    Symbolic regression for galaxy rotation curves: learns the DM *density*
    profile shape h(x) where x = r/d, rather than the velocity-squared shape.

    Model structure (per data point i in galaxy g):
        V²_obs = sign(Vgas)·V²_gas + a[g]·V²_disk + b[g]·V²_bul + c[g]·DM_col[i]

    where the DM column is built by spherical mass integration:
        DM_col[i] = (1/r_i) · ∫₀^{r_i} h(r'/d[g]) · r'² dr'

    This converts the symbolic density shape h(x) to a velocity-squared
    contribution via the spherical Jeans / circular-velocity relation
        V²_DM(r) = G·M(r)/r = (4πG·ρ₀·d³ / r) · ∫₀^{r/d} h(x) x² dx.
    The constant c[g] absorbs 4πG·ρ₀·d³.

    For fixed d[g], the model is LINEAR in a[g], b[g], c[g].  We solve via
    Iteratively Reweighted Least Squares (IRLS) for a Student-t(ν=nu_t)
    likelihood, using measurement-error weights w_i = 1/(2·Vobs_i·errV_i)².
    d[g] is chosen by a log-spaced grid search followed by local refinement.

    Physical motivation
    -------------------
    Standard density profiles are simpler expressions than their velocity
    counterparts:
        pISO:  h(x) = 1/(1+x²)            complexity ~5
               vs.  f(x) = 1−atan(x)/x    complexity ~6
        NFW:   h(x) = 1/(x·(1+x)²)        complexity ~7
               vs.  f(x) = [ln(1+x)−x/(1+x)]/x  complexity ~10
    The zero-at-origin constraint V²_DM(0)=0 is automatically satisfied by
    the integration, so no origin penalty is needed.

    NOTE: data is sorted by (galaxy_code, r) in Python before fitting.
    This is required so the Julia trapezoid loop can iterate i=1:n_g in
    ascending radius order within each galaxy.
    """
    if n_galaxies is not None:
        available = df["galaxy"].unique()
        rng = np.random.default_rng(seed=42)
        selected = rng.choice(available, size=min(n_galaxies, len(available)), replace=False)
        df = df[df["galaxy"].isin(selected)].copy()

    counts = df.groupby("galaxy")["galaxy"].transform("count")
    df = df[counts >= min_points].copy()

    galaxy_codes, galaxies = pd.factorize(df["galaxy"])
    df["galaxy_code"] = galaxy_codes + 1  # Julia is 1-indexed

    # Sort by (galaxy_code, r) — essential for the cumulative trapezoid in Julia.
    # Within each galaxy, data points must be in ascending radius order.
    df = df.sort_values(["galaxy_code", "Rad_kpc"]).reset_index(drop=True)

    y = df["Vobs_km/s"] ** 2
    X = pd.DataFrame({
        "r":       df["Rad_kpc"],
        "galaxy":  df["galaxy_code"],
        "Vgas2":   np.sign(df["Vgas_km/s"]) * df["Vgas_km/s"] ** 2,
        "Vdisk2":  df["Vdisk_km/s"] ** 2,
        "Vbulge2": df["Vbul_km/s"] ** 2,
    })

    # Custom loss (models/losses/density1param.jl): candidate density shape h
    # is integrated to a DM column via cumulative trapezoid, then (a, b, c, d)
    # found per galaxy as in the velocity model.  No origin penalty needed:
    # V²_DM(0)=0 automatically.  Data must be sorted by (galaxy_code, r).
    inner_opt_loss = load_loss(
        "density1param",
        nu_t=nu_t,
        upsilon_weight=upsilon_weight,
        n_irls_fine=n_irls_fine,
        n_irls_coarse=n_irls_coarse,
        n_d_grid=n_d_grid,
        weight_nonneg=weight_nonneg,
        weight_slope=weight_slope,
    )

    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["r", "galaxy", "Vgas2", "Vdisk2", "Vbulge2"],
        combine="f(r)",   # display: raw density shape h(r/d) ← d recovered post-hoc
    )

    # Algebraic operators only: no exp (dominated stage5), no log (diverges at x→0).
    # ^ binary allows power-law forms (e.g. 1/x², x^0.5); exponent constrained to
    # a single leaf so only x^const or x^var (not x^(a+b)).
    _unary = unary_operators if unary_operators is not None else []
    _nested = {}

    model = PySRRegressor(
        expression_spec=template,
        output_directory=output_directory,
        niterations=iterations,
        binary_operators=["*", "/", "-", "+"],
        unary_operators=_unary,
        nested_constraints=_nested,
        maxsize=22,
        populations=populations,
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        weight_optimize=weight_optimize,
        optimizer_iterations=optimizer_iterations,
        batching=False,
        turbo=True,
        complexity_of_constants=3,
        loss_function_expression=inner_opt_loss,
        procs=procs,
        guesses=guesses,
        fraction_replaced_guesses=fraction_replaced_guesses,
    )

    if error_weighting:
        errV_safe = np.maximum(df["errV_km/s"], 0.5)
        if "Inc_deg" in df.columns and "e_Inc_deg" in df.columns:
            inc_rad   = np.deg2rad(df["Inc_deg"])
            e_inc_rad = np.deg2rad(df["e_Inc_deg"])
            cot_i     = np.clip(np.cos(inc_rad) / np.sin(inc_rad), -10.0, 10.0)
            delta_v   = np.abs(df["Vobs_km/s"] * cot_i * e_inc_rad)
            errV_safe = np.sqrt(errV_safe**2 + delta_v**2)
        weights = 1.0 / (2 * df["Vobs_km/s"] * errV_safe) ** 2
    else:
        weights = np.ones(len(df))
    model.fit(X, y, weights=weights)


def recover_parameters_density(df: pd.DataFrame,
                                h_callable,
                                n_d_grid: int = 200,
                                upsilon_weight: float = 0.0) -> pd.DataFrame:
    """
    Post-hoc parameter recovery for the best density shape found by
    fit_density_inner_opt.

    Given the discovered DM density shape as a Python callable h(x) (where
    x = r/d), re-runs the same inner optimisation (log-spaced d-grid + weighted
    OLS) in Python to find the optimal (upsilon_disk, upsilon_bulge, c_DM, d_kpc)
    per galaxy.

    The DM column is computed via cumulative trapezoid (matching the Julia loss):
        dm_col[i] = (1/r_i) ∫₀^{r_i} h(r'/d) r'² dr'

    Returns a DataFrame with columns galaxy, upsilon_disk, upsilon_bulge,
    c_DM, d_kpc, wls.  c_DM absorbs 4πG·ρ₀·d³ (units: (km/s)²/kpc²).

    Example
    -------
    >>> h = lambda x: 1.0 / (1.0 + x**2)   # pISO density shape
    >>> params = recover_parameters_density(df, h)
    """
    results = []
    for g, gdf in df.groupby("galaxy"):
        gdf = gdf.sort_values("Rad_kpc")   # must be r-sorted for trapezoid
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
            h_vals = h_callable(r / d)
            if not np.all(np.isfinite(h_vals)):
                continue
            integrand = h_vals * r ** 2
            # Prepend virtual origin (r=0, integrand=0) for lower bound
            r_nodes   = np.concatenate([[0.0], r])
            ig_nodes  = np.concatenate([[0.0], integrand])
            cum_I     = cumulative_trapezoid(ig_nodes, r_nodes, initial=0.0)[1:]
            dm_col    = cum_I / r
            if not np.all(np.isfinite(dm_col)):
                continue
            A = np.column_stack([vd2, vb2, dm_col])
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
    fit_density_inner_opt(
        df,
        output_directory="outputs/SPARC/production/density1param",
        iterations=99999,
        n_galaxies=None,
        n_d_grid=100,
        populations=20,
        population_size=40,
        ncycles_per_iteration=100,
        weight_optimize=0.1,
        optimizer_iterations=8,
    )
