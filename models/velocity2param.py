from pysr import PySRRegressor, TemplateExpressionSpec
import pandas as pd
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.julia_losses import load_loss


def fit_vr_2param(df: pd.DataFrame,
                  output_directory: str = "outputs",
                  error_weighting: bool = True,
                  iterations: int = 99999,
                  n_galaxies: int | None = 5,
                  n_start_d: int = 4,
                  n_start_gamma: int = 4,
                  gamma_range: tuple = (0.0, 1.0),
                  populations: int = 20,
                  population_size: int = 40,
                  ncycles_per_iteration: int = 200,
                  weight_optimize: float = 0.0,
                  optimizer_iterations: int = 0,
                  upsilon_weight: float = 1.0,
                  origin_weight: float = 1.0,
                  origin_x: float = 1e-3,
                  weight_nonneg: float = 0.1,
                  weight_mono: float = 0.1,
                  binary_operators: list = ["/", "-", "+", "*"],
                  nu_t: float = 3.0,
                  n_irls: int = 3,
                  min_points: int = 5,
                  unary_operators: list = None,
                  maxsize: int = 18,
                  optimizer_niter: int = 50,
                  timeout_in_seconds: float | None = None,
                  procs: int = 0,
                  run_id: str | None = None,
                  warm_start: bool = False,
                  checkpoint_every: int | None = None):
    """
    2D symbolic regression: learn f(r/d, γ) where γ is a per-galaxy shape parameter
    optimised jointly with scale radius d.

    Model (per data point i in galaxy g):
        V²_obs = sign(Vgas)·V²_gas + a[g]·V²_disk + b[g]·V²_bul + c[g]·f(r/d[g], γ[g])

    Per-galaxy parameters:
      a, b — upsilon_disk, upsilon_bulge (linear, WLS solve; softplus positivity)
      c    — DM amplitude (linear)
      d    — DM scale radius (nonlinear; per-galaxy adaptive range, L-BFGS-B)
      γ    — shape parameter ∈ gamma_range (nonlinear, jointly optimised with d)

    f nodes use GLOBAL variable indices matching variable_names order.
    dataset.X / X_loc layout (10 rows):
      row 1:  x = r/d          (global #1; updated per loss_g call)
      row 2:  γ                (global #2; set per-galaxy)
      row 3:  xpow = (r/d)^-γ  (global #3; set per-galaxy)
      row 4:  1.0              (global #4 "one")
      row 5:  2.0              (global #5 "two")
      row 6:  3.0              (global #6 "three")
      rows 7-10: galaxy_code, Vgas2, Vdisk2, Vbulge2 (zeros in X_loc; used from dataset.X)
    """
    if n_galaxies is not None:
        available = df["galaxy"].unique()
        rng = np.random.default_rng(seed=42)
        selected = rng.choice(available, size=min(n_galaxies, len(available)), replace=False)
        df = df[df["galaxy"].isin(selected)].copy()

    counts = df.groupby("galaxy")["galaxy"].transform("count")
    df = df[counts >= min_points].copy()

    galaxy_codes, galaxies = pd.factorize(df["galaxy"])
    df["galaxy_code"] = galaxy_codes + 1
    df = df.sort_values("galaxy_code").reset_index(drop=True)

    y = df["Vobs_km/s"] ** 2
    # Column order puts f's variables at global indices 1-6 so #1=#x, #2=#gamma, etc.
    # Auxiliary data (galaxy_code, Vgas2, Vdisk2, Vbulge2) are at indices 7-10;
    # X_loc zeros these out so f expressions using #7-#10 evaluate to 0 → bad fit.
    X = pd.DataFrame({
        "r":           df["Rad_kpc"],           # global 1 = x
        "gamma_dummy": np.zeros(len(df)),       # global 2 = gamma  (overwritten per-galaxy in Julia)
        "xpow_dummy":  np.zeros(len(df)),       # global 3 = xpow   (overwritten per-galaxy in Julia)
        "one":         np.ones(len(df)),        # global 4 = one
        "two":         np.full(len(df), 2.0),   # global 5 = two
        "three":       np.full(len(df), 3.0),   # global 6 = three
        "galaxy":      df["galaxy_code"],       # global 7 — galaxy index scan
        "Vgas2":       np.sign(df["Vgas_km/s"]) * df["Vgas_km/s"] ** 2,  # global 8
        "Vdisk2":      df["Vdisk_km/s"] ** 2,  # global 9
        "Vbulge2":     df["Vbul_km/s"] ** 2,   # global 10
    })

    gamma_lo, gamma_hi = gamma_range

    # Custom loss (models/losses/velocity2param.jl): per galaxy, (a, b, c)
    # solved analytically; (log d, γ) jointly optimised by coarse grid +
    # Fminbox Nelder-Mead.
    inner_opt_loss = load_loss(
        "velocity2param",
        nu_t=nu_t,
        upsilon_weight=upsilon_weight,
        gamma_lo=gamma_lo,
        gamma_hi=gamma_hi,
        origin_x=origin_x,
        n_irls=n_irls,
        n_start_d=n_start_d,
        n_start_gamma=n_start_gamma,
        optimizer_niter=optimizer_niter,
        origin_weight=origin_weight,
        weight_nonneg=weight_nonneg,
        weight_mono=weight_mono,
    )

    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["x", "gamma", "xpow", "one", "two", "three", "galaxy_code", "Vgas2", "Vdisk2", "Vbulge2"],
        combine="f(x, gamma, xpow, one, two, three)",
    )

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
    fit_vr_2param(
        df,
        output_directory="outputs/toydatasets/velocity2param_toy",
        iterations=99999,
        n_galaxies=5,
        n_start_d=6,
        n_start_gamma=6,
        populations=10,
        population_size=20,
        ncycles_per_iteration=100,
        weight_optimize=0.0,
        optimizer_iterations=0,
        unary_operators=["atan", "log1p"],
        maxsize=25,
    )
