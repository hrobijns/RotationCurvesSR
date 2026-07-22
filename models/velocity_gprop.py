import json
import os

from pysr import PySRRegressor, TemplateExpressionSpec
import pandas as pd
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.julia_losses import load_loss
from models.gprop_features import galaxy_features, PROP_NAMES, N_PROPS


def fit_vr_gprop(df: pd.DataFrame,
                 output_directory: str = "outputs",
                 params_csv: str = "data/SPARC/galaxy_parameters.csv",
                 error_weighting: bool = True,
                 iterations: int = 99999,
                 n_galaxies: int | None = 5,
                 n_start_d: int = 12,
                 n_start_gamma: int = 5,
                 coarse_irls: int = 1,
                 gamma_range: tuple = (0.0, 1.0),
                 sigma_int: float = 0.05,
                 weight_gamma_range: float = 1.0,
                 gvar_min: float = 0.05,
                 weight_gvar: float = 100.0,
                 populations: int = 20,
                 population_size: int = 40,
                 ncycles_per_iteration: int = 200,
                 weight_optimize: float = 0.0,
                 optimizer_iterations: int = 8,
                 upsilon_weight: float = 1.0,
                 origin_weight: float = 1.0,
                 origin_x: float = 1e-3,
                 weight_nonneg: float = 0.1,
                 weight_mono: float = 0.1,
                 binary_operators: list = ["/", "-", "+", "*"],
                 nu_t: float = 3.0,
                 n_irls: int = 5,
                 min_points: int = 5,
                 unary_operators: list = None,
                 maxsize: int = 30,
                 optimizer_niter: int = 100,
                 timeout_in_seconds: float | None = None,
                 guesses: list | None = None,
                 procs: int = 0,
                 run_id: str | None = None,
                 warm_start: bool = False,
                 checkpoint_every: int | None = None):
    """
    Joint SR: learn f(r/d, γ) AND γ = g(galaxy properties) simultaneously.

    Model (per data point i in galaxy g), v3:
        V²_obs = sign(Vgas)·V²_gas + a[g]·V²_disk + b[g]·V²_bul + c[g]·f(r/d[g], γ[g])
        γ[g]   ~ LogNormal( g(x_M, x_f, x_S, x_R)[g], sigma_int )

    The lognormal population prior is implemented in the loss: per galaxy, γ is
    optimised inside γ_pred·10^{±3·sigma_int} with the smooth penalty
    ((log10 γ − log10 γ_pred)/sigma_int)² added to the NLL (replaces the v2
    hard band). Features come from models/gprop_features.py (photometric/gas
    only, fixed anchors — see that module).

    Per-galaxy parameters:
      a, b — upsilon_disk, upsilon_bulge (linear, WLS → Student-t IRLS solve)
      c    — DM amplitude (linear)
      d    — DM scale radius (nonlinear; per-galaxy adaptive range)
      γ    — within the lognormal prior around g(props)
             (NOT PySR parameters — avoids SVector{N} compile bottleneck)

    Inner-opt settings (n_start_d=12, n_start_gamma=5 within the prior box at
    coarse_irls=1, top-2 Nelder-Mead restarts on the FULL box, n_irls=5) are
    accuracy-gated by analysis/diagnose_inner_opt.py — v2's 5×3 grid +
    margin-shrunk NM box misranked candidates (Spearman 0.83 → 1.00 with these
    settings, at 2.6× the throughput of the maximal-accuracy config).

    PySR-side constants are DISABLED (weight_optimize=0, complexity_of_constants
    =99, should_optimize_constants=False — the velocity2param recipe): BFGS
    constant tuning nests full inner optimisations inside finite-difference
    gradients (dozens of full-dataset evals per call) and starved the search.
    Rational coefficients are built from the one/two/three atoms instead
    (e.g. 1/2 − x_M/5 = #5/#6 − #1/(#6+#7)); seeds carry such forms.

    Guards (in models/losses/velocity_gprop.jl):
      - sterile-member guard (arity only): hard-fail out-of-arity features
      - constant-g guard: std of clamped γ_pred across galaxies must exceed
        gvar_min, else a smooth penalty up to weight_gvar — g must genuinely
        vary with galaxy properties

    Sub-expression feature indices are LOCAL to each combine argument list
    (check_constraints rejects trees with g features >n_props+3 / f features >6):
      f(x, ·, xpow, one, two, three): local 1=x, 2=gamma, 3=xpow, 4=one, 5=two, 6=three
      g(<props>, one, two, three):    local 1..n_props = props, then one, two, three
    BUT randomly initialised trees may reference any dataset feature before
    constraint checks prune them, so all eval matrices must have n_features rows
    (extra rows zero → garbage loss → pruned).
    dataset.X layout (10 + n_props rows):
      rows 1-6:   x, γ, xpow, one, two, three (f's args; 1-3 overwritten per-galaxy)
      rows 7-10:  galaxy_code, Vgas2, Vdisk2, Vbulge2 (aux)
      rows 11+:   anchored galaxy properties (constant within galaxy block;
                  fixed a-priori anchors, no sample statistics — see PROPS below)
    """
    if n_galaxies is not None:
        available = df["galaxy"].unique()
        rng = np.random.default_rng(seed=42)
        selected = rng.choice(available, size=min(n_galaxies, len(available)), replace=False)
        df = df[df["galaxy"].isin(selected)].copy()

    counts = df.groupby("galaxy")["galaxy"].transform("count")
    df = df[counts >= min_points].copy()

    # ---- Galaxy properties (g's features) — see models/gprop_features.py ----
    from models.gprop_features import PROPS
    n_props = N_PROPS
    prop_names = list(PROP_NAMES)
    props = galaxy_features(pd.read_csv(params_csv))

    n_before = df["galaxy"].nunique()
    df = df.drop(columns=[c for c in ("log_Mstar", "log_fgas", *prop_names)
                          if c in df.columns])
    df = df.merge(props, on="galaxy", how="inner")
    n_after = df["galaxy"].nunique()
    if n_after < n_before:
        print(f"Dropped {n_before - n_after} galaxies missing properties in {params_csv}")

    scaler = dict(PROPS)

    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, "feature_scaler.json"), "w") as fh:
        json.dump(scaler, fh, indent=2)

    galaxy_codes, galaxies = pd.factorize(df["galaxy"])
    df["galaxy_code"] = galaxy_codes + 1
    df = df.sort_values("galaxy_code").reset_index(drop=True)

    y = df["Vobs_km/s"] ** 2
    X = pd.DataFrame({
        "r":           df["Rad_kpc"],           # global 1 = x
        "gamma_dummy": np.zeros(len(df)),       # global 2 = gamma  (overwritten per-galaxy in Julia)
        "xpow_dummy":  np.zeros(len(df)),       # global 3 = xpow   (overwritten per-galaxy in Julia)
        "one":         np.ones(len(df)),        # global 4 = one
        "two":         np.full(len(df), 2.0),   # global 5 = two
        "three":       np.full(len(df), 3.0),   # global 6 = three
        "galaxy":      df["galaxy_code"],       # global 7 — galaxy index scan
        "Vgas2":       np.sign(df["Vgas_km/s"]) * df["Vgas_km/s"] ** 2,  # global 8
        "Vdisk2":      df["Vdisk_km/s"] ** 2,   # global 9
        "Vbulge2":     df["Vbul_km/s"] ** 2,    # global 10
        # globals 11..(10+n_props) — g's locals 1..n_props
        **{name: df[name] for name in prop_names},
    })

    gamma_lo, gamma_hi = gamma_range

    # Custom loss (models/losses/velocity_gprop.jl): γ ~ LogNormal(g(props),
    # sigma_int); (a, b, c) analytic, (log d, γ) by dense coarse grid + top-3
    # Fminbox Nelder-Mead restarts; sterile-arity and constant-g guards.
    inner_opt_loss = load_loss(
        "velocity_gprop",
        nu_t=nu_t,
        upsilon_weight=upsilon_weight,
        gamma_lo=gamma_lo,
        gamma_hi=gamma_hi,
        sigma_int=sigma_int,
        weight_gamma_range=weight_gamma_range,
        n_props=n_props,
        n_features=10 + n_props,
        g_max_feature=n_props + 3,
        gvar_min=gvar_min,
        weight_gvar=weight_gvar,
        origin_x=origin_x,
        n_irls=n_irls,
        coarse_irls=coarse_irls,
        n_start_d=n_start_d,
        n_start_gamma=n_start_gamma,
        optimizer_niter=optimizer_niter,
        origin_weight=origin_weight,
        weight_nonneg=weight_nonneg,
        weight_mono=weight_mono,
    )

    template = TemplateExpressionSpec(
        expressions=["f", "g"],
        variable_names=["x", "gamma", "xpow", "one", "two", "three",
                        "galaxy_code", "Vgas2", "Vdisk2", "Vbulge2",
                        *prop_names],
        combine=f"f(x, g({', '.join(prop_names)}, one, two, three), "
                "xpow, one, two, three)",
    )

    # Seed populations with feature-valid members. Random init trees sample all
    # dataset features, but valid members need f-features ≤6 AND g-features
    # ≤n_props+3 (joint check_constraints), so unseeded populations start
    # ~all-invalid and take a long time to bootstrap.
    # f seeds: physically motivated profiles (see profiles.py) — pISO, NFW,
    # SR-A (C11), SR-C9, SR-G23. g seeds: linear laws along the known post-hoc
    # relation (γ decreasing with M*, slope ≈ −0.2) + variants. PySR constants
    # are disabled, so coefficients are rationals built from the atoms.
    # Placeholders are LOCAL: f: #1=x #2=γ #3=xpow=x^(−γ) #4=1 #5=2 #6=3;
    #   g: #1=x_M #2=x_f #3=x_S #4=x_R #5=1 #6=2 #7=3.
    if guesses is None:
        g_lin_M = "#5/#6 - #1/(#6 + #7)"        # 1/2 − x_M/5 ≈ known law
        g_lin_F = "#5/#7 + #2/(#6 + #7)"        # 1/3 + x_f/5
        g_lin_S = "#5/#6 - #3/(#6 * #7)"        # 1/2 − x_S/6
        g_lin_MS = "#5/#6 - (#1 + #3)/(#6 * #7)"  # 1/2 − (x_M+x_S)/6
        guesses = [
            {"f": "#4 - atan(#1)/#1",              "g": g_lin_M},   # pISO + M* law
            {"f": "#4 - atan(#1)/#1",              "g": g_lin_S},   # pISO + Σ* law
            {"f": "(log1p(#1) - #1/(#4+#1))/#1",   "g": g_lin_M},   # NFW
            {"f": "#3*atan(#6*log1p((#1/#3)/#3))", "g": g_lin_M},   # SR-A (C11)
            {"f": "#3*atan(#6*log1p((#1/#3)/#3))", "g": g_lin_MS},  # SR-A + M,Σ
            {"f": "#1/(#1 + #3 - #2*#6)",          "g": g_lin_F},   # SR-C9 + fgas
            {"f": "atan(#3*(#1 - log1p(#1))/(#2*(#1 + log1p(#6))))",
                                                    "g": g_lin_M},  # SR-G19 family
        ]

    active_unary = unary_operators if unary_operators is not None else ["atan", "log1p"]
    nested = {op: {op: 0} for op in active_unary}
    if "exp" in active_unary:
        nested["exp"]["log1p"] = 0  # exp(log1p(x)) ≈ x, trivial
        if "log" in active_unary:
            nested["exp"]["log"] = 0   # exp(log(x)) = x, trivial
            nested["log"]["exp"] = 0   # log(exp(x)) = x, trivial
        if "log1p" in active_unary:
            nested["log1p"]["exp"] = 0 # log1p(exp(x)) ≈ x for large x
    # Resume from a prior checkpoint (SL3-chained run) if one exists; otherwise
    # build fresh. run_id must be fixed (not the PySR-default timestamp) for
    # the checkpoint path to be findable across separate sbatch submissions.
    checkpoint = Path(output_directory) / str(run_id) / "checkpoint.pkl" if run_id else None
    if warm_start and checkpoint is not None and checkpoint.exists():
        # Not PySRRegressor.from_file(): that eagerly calls model.refresh(),
        # which reconstructs the hall of fame from Julia state that isn't
        # valid for a run interrupted mid-search (crashes with "cannot unpack
        # non-iterable NoneType" in get_hof()). The actual resume mechanism
        # lives in .fit() itself (keyed on warm_start + julia_state_stream_,
        # sr.py ~line 2458) and doesn't need refresh() first, so just unpickle
        # and let fit() below do the resume.
        import pickle
        print(f"[resume] loading checkpoint from {checkpoint} (skipping PySR's own refresh())")
        with open(checkpoint, "rb") as f:
            model = pickle.load(f)
        model.set_params(warm_start=True)
    else:
        model = PySRRegressor(
            expression_spec=template,
            output_directory=output_directory,
            run_id=run_id,
            warm_start=warm_start,
            niterations=iterations,
            binary_operators=binary_operators,
            unary_operators=active_unary,
            nested_constraints=nested,
            maxsize=maxsize,
            populations=populations,
            population_size=population_size,
            ncycles_per_iteration=ncycles_per_iteration,
            weight_optimize=weight_optimize,
            optimizer_iterations=optimizer_iterations,
            should_optimize_constants=False,
            complexity_of_constants=99,
            batching=False,
            turbo=True,
            timeout_in_seconds=timeout_in_seconds,
            procs=procs,
            guesses=guesses,
            loss_function_expression=inner_opt_loss,
        )

    if error_weighting:
        errV_safe = np.maximum(df["errV_km/s"], 0.5)
        weights = 1.0 / (2 * df["Vobs_km/s"] * errV_safe) ** 2
    else:
        weights = np.ones(len(df))

    if checkpoint_every is not None:
        # PySR only checkpoints before _run() starts and after it returns
        # normally (sr.py fit()) — a niterations=99999 run killed by SLURM's
        # wall-clock never reaches the "after" checkpoint, so nothing from
        # this job's search survives to disk. Looping fit() in small chunks
        # means every chunk completes normally, so a real, resumable
        # checkpoint is written every `checkpoint_every` iterations instead
        # of only (never) at the end.
        model.set_params(niterations=checkpoint_every, warm_start=True)
        while True:
            model.fit(X, y, weights=weights)
    else:
        model.fit(X, y, weights=weights)


if __name__ == "__main__":
    from datawrangling import produce_SPARC_df
    df = produce_SPARC_df("data/SPARC", selected=True)
    fit_vr_gprop(
        df,
        output_directory="outputs/toydatasets/velocity_gprop_smoke",
        iterations=99999,
        n_galaxies=5,
        populations=10,
        population_size=20,
        ncycles_per_iteration=100,
        unary_operators=["atan", "log1p"],
        maxsize=28,
    )
