import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

###################################################################################################
# SPARC INGESTION:

def date_file_to_df(path: str) -> pd.DataFrame:
    """Load in a SPARC .dat file from a given file path into a dataframe."""
    columns = [
        "Rad_kpc", "Vobs_km/s", "errV_km/s", "Vgas_km/s",
        "Vdisk_km/s", "Vbul_km/s", "SBdisk_L/pc^2", "SBbul_L/pc^2"
    ]
    df = pd.read_csv(path, comment="#", sep=r'\s+', names = columns)
    df["galaxy"] = os.path.basename(path).replace("_rotmod.dat", "")
    return df

def produce_SPARC_df(
    data_dir: str,
    reduced_load: bool = False,
    n: int = 50,
    random_state: int = 42,
    quality: int | None = None,
    galaxies: list[str] | None = None,
    selected: bool | None = None,
) -> pd.DataFrame:
    """
    Combine all SPARC .dat files from the data folder into one DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to directory holding all the SPARC _rotmod.dat files.
    reduced_load : bool
        If True, randomly sample at most n galaxies from the (filtered) set.
    n : int
        Number of galaxies to sample when reduced_load=True.
    random_state : int
        RNG seed for reproducible sampling.
    quality : int | None
        If provided, keep only galaxies whose Q column in galaxy_parameters.csv
        equals this value (e.g. quality=1 for highest-quality curves).
        Expects galaxy_parameters.csv one directory above data_dir.
    galaxies : list[str] | None
        If provided, keep only the named galaxies (e.g. ["NGC2403", "DDO154"]).
        Can be combined with quality; the intersection is used.
    selected : bool | None
        If True, keep only galaxies flagged selected=True in galaxy_parameters.csv
        (Q≤2, T≥1, 30°≤i≤80°, n≥10, non-AGN).  If False, keep only the excluded
        ones.  Can be combined with quality/galaxies; the intersection is used.

    Returns
    -------
    pd.DataFrame with columns:
        'Rad_kpc', 'Vobs_km/s', 'errV_km/s', 'Vgas_km/s', 'Vdisk_km/s',
        'Vbul_km/s', 'SBdisk_L/pc^2', 'SBbul_L/pc^2', 'galaxy'
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith("_rotmod.dat")]
    allowed = {f.replace("_rotmod.dat", "") for f in all_files}

    params_path = os.path.join(
        os.path.dirname(os.path.normpath(data_dir)), "SPARC/galaxy_parameters.csv"
    )

    if quality is not None:
        galaxy_params = pd.read_csv(params_path)
        q_names = set(galaxy_params.loc[galaxy_params["Q"] == quality, "galaxy"])
        allowed &= q_names

    if selected is not None:
        galaxy_params = pd.read_csv(params_path)
        sel_names = set(galaxy_params.loc[galaxy_params["selected"] == selected, "galaxy"])
        allowed &= sel_names

    if galaxies is not None:
        allowed &= set(galaxies)

    selected_files = [f for f in all_files if f.replace("_rotmod.dat", "") in allowed]

    if reduced_load:
        rng = np.random.default_rng(random_state)
        selected_files = list(
            rng.choice(selected_files, size=min(n, len(selected_files)), replace=False)
        )

    dfs = [date_file_to_df(os.path.join(data_dir, f)) for f in selected_files]

    out = pd.concat(dfs, ignore_index=True)

    # Join inclination and distance columns from galaxy_parameters.csv
    params_path = os.path.join(
        os.path.dirname(os.path.normpath(data_dir)), "SPARC", "galaxy_parameters.csv"
    )
    if os.path.exists(params_path):
        galaxy_params = pd.read_csv(params_path)
        inc_cols = galaxy_params[["galaxy", "Inc_deg", "e_Inc_deg", "D_Mpc", "e_D_Mpc"]]
        out = out.merge(inc_cols, on="galaxy", how="left")

    print(f"Taken data from {len(out['galaxy'].unique())} galaxies, making one DataFrame with {len(out)} rows.")
    return out


###################################################################################################
# GALAXY SELECTION CRITERIA:

def _excl_reason(row):
    """Return the first failing criterion, or '' if the galaxy passes all cuts."""
    if row["Q"] == 3:                return "Q=3"
    if row["Inc_deg"] < 30:          return "i<30 (face-on)"
    if row["n_data"] < 10:           return "n<10"
    return ""


def update_galaxy_selection(
    data_dir: str = "data/SPARC",
    gp_path: str = "data/galaxy_parameters.csv",
) -> pd.DataFrame:
    """
    Compute n_data, selected, and excl_reason for every galaxy and write them
    back into galaxy_parameters.csv.

    Selection criteria:
      - Q <= 2
      - 30 <= Inc_deg  (exclude face-on)
      - n_data >= 10

    Returns the updated DataFrame.
    """
    gp = pd.read_csv(gp_path)

    # Drop any previously computed columns so we can recompute cleanly
    gp = gp.drop(columns=[c for c in ("n_data", "selected", "excl_reason") if c in gp.columns])

    # Compute data-point counts from the rotation-curve files
    df = produce_SPARC_df(data_dir)
    n_pts = df.groupby("galaxy").size().rename("n_data")
    gp = gp.set_index("galaxy").join(n_pts).reset_index()

    gp["excl_reason"] = gp.apply(_excl_reason, axis=1)
    gp["selected"]    = gp["excl_reason"] == ""

    # Preserve column order: insert new columns before Ref
    cols = [c for c in gp.columns if c not in ("n_data", "selected", "excl_reason", "Ref")]
    ref_idx = cols.index("Q") + 1   # put them right after Q
    cols = cols[:ref_idx] + ["n_data", "selected", "excl_reason"] + cols[ref_idx:] + ["Ref"]
    gp = gp[[c for c in cols if c in gp.columns]]

    gp.to_csv(gp_path, index=False)
    print(f"Updated {gp_path}: {gp['selected'].sum()} selected, {(~gp['selected']).sum()} excluded.")
    return gp


###################################################################################################
# SPARC TOY DATASET GENERATION:

def make_pISO_toy(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Replace Vobs with the exact pISO total velocity for each galaxy, using
    per-galaxy MCMC-fitted parameters from results (e.g. outputs/fits/mcmc_pISO.csv).
    Baryonic components are unchanged; upsilon columns are appended.

    Parameters
    ----------
    df : pd.DataFrame
        SPARC dataframe from produce_SPARC_df.
    results : pd.DataFrame
        Per-galaxy fitted parameters with columns:
        galaxy, upsilon_disk, upsilon_bulge, a (km²/s²), b_kpc.

    Returns
    -------
    pd.DataFrame matching df's columns plus upsilon_disk, upsilon_bulge.
    """
    rows = []
    for _, res in results.iterrows():
        gdf = df[df["galaxy"] == res["galaxy"]].copy()
        if gdf.empty:
            continue
        r      = gdf["Rad_kpc"].values.astype(float)
        vgas   = gdf["Vgas_km/s"].values.astype(float)
        vdisk  = gdf["Vdisk_km/s"].values.astype(float)
        vbulge = gdf["Vbul_km/s"].values.astype(float)

        x     = r / res["b_kpc"]
        vdm2  = res["a"] * (1.0 - np.arctan(x) / x)
        v2    = (np.sign(vgas) * vgas**2
                 + res["upsilon_disk"]  * vdisk**2
                 + res["upsilon_bulge"] * vbulge**2
                 + vdm2)
        gdf["Vobs_km/s"]    = np.sqrt(np.maximum(v2, 0.0))
        gdf["upsilon_disk"]  = res["upsilon_disk"]
        gdf["upsilon_bulge"] = res["upsilon_bulge"]
        rows.append(gdf)
    return pd.concat(rows, ignore_index=True)


def add_velocity_noise(df: pd.DataFrame, sigma: float = 0.05,
                       seed: int | None = None) -> pd.DataFrame:
    """
    Add multiplicative Gaussian noise to Vobs and set errV accordingly.

    Vobs → Vobs · (1 + ε),  ε ~ N(0, sigma²)
    errV_km/s = sigma · Vobs  (before noise)

    Parameters
    ----------
    df : pd.DataFrame
        Toy dataset from make_pISO_toy (or any SPARC-format DataFrame).
    sigma : float
        Fractional noise level (0.02 = small, 0.05 = mid, 0.10 = strong).
    seed : int | None
        RNG seed for reproducibility.
    """
    rng      = np.random.default_rng(seed)
    out      = df.copy()
    eps      = rng.normal(loc=0.0, scale=sigma, size=len(df))
    out["errV_km/s"]  = sigma * out["Vobs_km/s"]
    out["Vobs_km/s"] *= (1.0 + eps)
    return out