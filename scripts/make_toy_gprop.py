"""
make_toy_gprop.py — toy datasets for the joint f+g model (models/velocity_gprop.py).

Takes real SPARC baryonic curves (selected Q=1) and injects synthetic DM:
    V²_obs = sign(Vgas)·Vgas² + a·Vdisk² + b·Vbul² + c·f(r/d, γ)
    f(x,γ) = x^(-γ) · (x - atan(x))/x          (γ-modulated pISO velocity shape)

Two outputs:
  data/toydatasets/gprop_midnoise.csv    (v2 toy, kept for provenance)
    γ_true = 0.5 - 0.2·z_Mstar (z-scored), no intrinsic scatter
  data/toydatasets/gprop_v3_scatter.csv  (v3 toy — pass --v3)
    γ_pred = 0.5 - 0.2·x_M   (anchored feature, models/gprop_features.py)
    γ_true = clip(γ_pred · 10^N(0, 0.05 dex), 0.01, 0.99)
    → PASS gate for the v3 pipeline: recover g's coefficients within ~±30%
      and Spearman(γ_planted, γ_recovered) ≥ 0.9 despite the planted scatter.

5% multiplicative noise on Vobs. Truth columns appended for recovery checks.
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datawrangling import produce_SPARC_df
from models.gprop_features import galaxy_features

PARAMS_CSV = "data/SPARC/galaxy_parameters.csv"
NOISE = 0.05
SIGMA_INT = 0.05  # dex, v3 planted lognormal scatter

ap = argparse.ArgumentParser()
ap.add_argument("--v3", action="store_true",
                help="planted anchored-x_M law + lognormal scatter (v3 PASS gate)")
args = ap.parse_args()
OUT = "data/toydatasets/gprop_v3_scatter.csv" if args.v3 else "data/toydatasets/gprop_midnoise.csv"

rng = np.random.default_rng(seed=7)

df = produce_SPARC_df("data/SPARC", selected=True)

params = pd.read_csv(PARAMS_CSV)
props = params[["galaxy", "M_star_1e9_Msun"]].dropna().copy()
props["log_Mstar"] = np.log10(props["M_star_1e9_Msun"].clip(lower=1e-4))
df = df.merge(props[["galaxy", "log_Mstar"]], on="galaxy", how="inner")

feats = galaxy_features(params).set_index("galaxy")
gal_props = df.groupby("galaxy")["log_Mstar"].first()
mu, sd = gal_props.mean(), gal_props.std()
print(f"Galaxies: {len(gal_props)}  log_Mstar mean={mu:.3f} std={sd:.3f}")


def f_true(x, gamma):
    x = np.clip(x, 1e-3, 1e3)
    return x ** (-gamma) * (x - np.arctan(x)) / x


rows = []
for gal, sub in df.groupby("galaxy", sort=False):
    sub = sub.sort_values("Rad_kpc").copy()
    r = sub["Rad_kpc"].values
    if args.v3:
        gamma_pred = float(np.clip(0.5 - 0.2 * feats.loc[gal, "x_M"], 0.02, 0.98))
        gamma = float(np.clip(gamma_pred * 10 ** rng.normal(0.0, SIGMA_INT), 0.01, 0.99))
    else:
        z = (gal_props[gal] - mu) / sd
        gamma_pred = gamma = float(np.clip(0.5 - 0.2 * z, 0.05, 0.95))

    a = float(np.exp(rng.normal(np.log(0.5), 0.1 * np.log(10))))   # upsilon_disk
    b = float(np.exp(rng.normal(np.log(0.7), 0.1 * np.log(10))))   # upsilon_bulge
    d = float(np.exp(rng.uniform(np.log(r.max() / 4), np.log(r.max()))))
    c = float(rng.uniform(60.0, 150.0)) ** 2                       # km²/s²

    v2 = (np.sign(sub["Vgas_km/s"]) * sub["Vgas_km/s"] ** 2
          + a * sub["Vdisk_km/s"] ** 2
          + b * sub["Vbul_km/s"] ** 2
          + c * f_true(r / d, gamma))
    v2 = np.maximum(v2.values, 1.0)
    vobs = np.sqrt(v2) * (1 + NOISE * rng.standard_normal(len(r)))

    sub["Vobs_km/s"] = vobs
    sub["errV_km/s"] = NOISE * np.sqrt(v2)
    sub["upsilon_disk"] = a
    sub["upsilon_bulge"] = b
    sub["c_dm"] = c
    sub["d_kpc"] = d
    sub["gamma_true"] = gamma
    sub["gamma_pred_true"] = gamma_pred
    if not args.v3:
        sub["z_Mstar"] = (gal_props[gal] - mu) / sd
    rows.append(sub)

out = pd.concat(rows, ignore_index=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_csv(OUT, index=False)
print(f"Saved {len(out)} rows, {out['galaxy'].nunique()} galaxies -> {OUT}")
print(f"gamma_true range: [{out['gamma_true'].min():.3f}, {out['gamma_true'].max():.3f}]")
