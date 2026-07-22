from pysr import PySRRegressor, TemplateExpressionSpec
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.julia_losses import load_loss


def fit_density_2param(df: pd.DataFrame,
                       output_directory: str = "outputs",
                       error_weighting: bool = True,
                       iterations: int = 99999,
                       n_galaxies: int | None = 5,
                       n_d_grid: int = 15,
                       n_gamma_grid: int = 8,
                       gamma_range: tuple = (0.0, 1.0),
                       populations: int = 20,
                       population_size: int = 40,
                       ncycles_per_iteration: int = 100,
                       weight_optimize: float = 0.1,
                       optimizer_iterations: int = 8,
                       upsilon_weight: float = 1.0,
                       nu_t: float = 3.0,
                       n_irls: int = 5,
                       min_points: int = 5,
                       unary_operators: list | None = None,
                       guesses: list | None = None,
                       fraction_replaced_guesses: float = 0.001,
                       weight_nonneg: float = 0.1,
                       weight_slope: float = 0.1,
                       n_gs_iter: int = 20,
                       timeout_in_seconds: float | None = None,
                       procs: int = 0):
    """
    Two-parameter symbolic regression for galaxy rotation curves: learns the DM
    density shape h(x, γ) directly, where x = r/d and γ is a per-galaxy shape
    parameter optimised jointly with d via Nelder-Mead.

    h(x, γ) is evaluated directly from the SR expression; negative values are
    clamped to zero with a penalty (weight_nonneg) at physics test points.
    An inner log-slope penalty (weight_slope) enforces 0 ≤ s ≤ 2 at x ∈ [0.001, 0.01].

    pISO target:  h(x) = 1/(1 + x²)         (γ-independent)
    NFW target:   h(x) = 1/(x·(1 + x)²)     (γ-independent)
    gNFW target:  h(x, γ) = x^(-γ) / (1+x)^(3-γ)

    Model structure (per data point i in galaxy g):
        V²_obs = sign(Vgas)·V²_gas + a[g]·V²_disk + b[g]·V²_bul + c[g]·DM_col[i]

    where:
        DM_col[i] = (1/r_i) · ∫₀^{r_i} h_rec(r'/d[g], γ[g]) · r'² dr'

    Per-galaxy parameters:
        a, b  — M/L ratios (solved analytically via WLS/IRLS)
        c     — DM amplitude (solved analytically)
        d     — scale radius (per-galaxy adaptive range; Nelder-Mead from top-3 coarse starts)
        γ     — per-galaxy shape parameter

    The SR expression h(#1, #2) has:
        #1 = r/d  (updated per d-iteration)
        #2 = γ    (updated per γ-iteration; overwrites the galaxy-code row in X_eval)
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

    # Sort by (galaxy_code, r) — required for the cumulative trapezoid in Julia.
    df = df.sort_values(["galaxy_code", "Rad_kpc"]).reset_index(drop=True)

    y = df["Vobs_km/s"] ** 2
    X = pd.DataFrame({
        "r":       df["Rad_kpc"],
        "galaxy":  df["galaxy_code"],    # col 2 in Julia dataset.X (galaxy ID)
        "Vgas2":   np.sign(df["Vgas_km/s"]) * df["Vgas_km/s"] ** 2,
        "Vdisk2":  df["Vdisk_km/s"] ** 2,
        "Vbulge2": df["Vbul_km/s"] ** 2,
    })

    gamma_lo, gamma_hi = gamma_range

    # Custom loss (models/losses/density2param.jl): h(x, γ) integrated to a DM
    # column; (log d, γ) optimised by coarse 2-D grid (top-3 basins) +
    # Nelder-Mead refinement.  Data must be sorted by (galaxy_code, r).
    inner_opt_loss = load_loss(
        "density2param",
        nu_t=nu_t,
        upsilon_weight=upsilon_weight,
        gamma_lo=gamma_lo,
        gamma_hi=gamma_hi,
        n_gamma_grid=n_gamma_grid,
        n_irls=n_irls,
        n_gs_iter=n_gs_iter,
        n_d_grid=n_d_grid,
        weight_nonneg=weight_nonneg,
        weight_slope=weight_slope,
    )

    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["r", "gamma", "Vgas2", "Vdisk2", "Vbulge2", "xpow"],
        combine="f(r, gamma, xpow)",   # xpow = (r/d)^γ, precomputed per-galaxy
    )

    # xpow = (r/d)^γ is precomputed as variable 6 in X_eval — provides x^γ power-law
    # access without needing ^ as a binary operator (which causes x^x junk).
    # Algebraic operators only: +,-,*,/ over {r, gamma, xpow}.
    _unary = unary_operators if unary_operators is not None else []
    nested = {}

    model = PySRRegressor(
        expression_spec=template,
        output_directory=output_directory,
        niterations=iterations,
        binary_operators=["*", "/", "-", "+"],
        unary_operators=_unary,
        nested_constraints=nested,
        maxsize=28,
        populations=populations,
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        weight_optimize=weight_optimize,
        optimizer_iterations=optimizer_iterations,
        batching=False,
        turbo=True,
        complexity_of_constants=3,
        timeout_in_seconds=timeout_in_seconds,
        procs=procs,
        loss_function_expression=inner_opt_loss,
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


def recover_parameters_density_2param(df: pd.DataFrame,
                                       h_callable,
                                       n_d_grid: int = 200,
                                       n_gamma_grid: int = 50,
                                       gamma_range: tuple = (0.0, 1.0),
                                       upsilon_weight: float = 0.0) -> pd.DataFrame:
    """
    Post-hoc parameter recovery for a density shape h(x, gamma) discovered by
    fit_density_2param.

    Given h as a callable h(x: np.ndarray, gamma: float) -> np.ndarray, recovers
    optimal (upsilon_disk, upsilon_bulge, c_DM, d_kpc, gamma) per galaxy via a
    2D grid search over (d, gamma) followed by WLS.

    Returns a DataFrame with columns:
        galaxy, upsilon_disk, upsilon_bulge, c_DM, d_kpc, gamma, wls

    Example
    -------
    >>> # generalised NFW
    >>> h = lambda x, g: x**(-g) / (1.0 + x)**(3.0 - g)
    >>> params = recover_parameters_density_2param(df, h)
    """
    results = []
    gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_gamma_grid)

    for g, gdf in df.groupby("galaxy"):
        gdf = gdf.sort_values("Rad_kpc")
        r     = gdf["Rad_kpc"].values.astype(float)
        y_obs = gdf["Vobs_km/s"].values ** 2
        vgas2 = np.sign(gdf["Vgas_km/s"].values) * gdf["Vgas_km/s"].values ** 2
        vd2   = gdf["Vdisk_km/s"].values ** 2
        vb2   = gdf["Vbul_km/s"].values ** 2
        errV  = np.maximum(gdf["errV_km/s"].values, 0.5)
        w     = 1.0 / (2 * gdf["Vobs_km/s"].values * errV) ** 2
        resid = y_obs - vgas2

        r_max_g = r.max()
        d_min_g = r_max_g / 20
        d_max_g = max(r_max_g * 10, 50.0)
        best = (np.inf, None)
        for gamma in gamma_vals:
            for d in np.exp(np.linspace(np.log(d_min_g), np.log(d_max_g), n_d_grid)):
                try:
                    h_vals = h_callable(r / d, gamma)
                except Exception:
                    continue
                if not np.all(np.isfinite(h_vals)):
                    continue
                integrand = h_vals * r ** 2
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
                                          d_kpc=d, gamma=gamma, wls=wls))
                except Exception:
                    pass

        if best[1] is not None:
            results.append(best[1])

    return pd.DataFrame(results)


if __name__ == "__main__":
    from datawrangling import produce_SPARC_df
    df = produce_SPARC_df("data/SPARC", selected=True)
    fit_density_2param(
        df,
        output_directory="outputs/SPARC/production/density2param",
        iterations=99999,
        n_galaxies=None,
        n_d_grid=15,
        n_gamma_grid=8,
        populations=20,
        population_size=40,
        ncycles_per_iteration=100,
        weight_optimize=0.1,
        optimizer_iterations=8,
    )
