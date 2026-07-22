"""
gprop_features.py — the galaxy-property features used by g(γ | observables).

Single source of truth for the gprop pipeline (models/velocity_gprop.py, the MCMC
verification runners, and any post-hoc analysis). Design rules:

- Photometric/gas observables ONLY. Vflat, M_vir, E are derived from the rotation
  curve being fitted (circular) and Vflat has 12 gaps among the selected sample.
- Fixed a-priori anchors, NOT sample statistics — a discovered g applies to any
  single galaxy without reference to the fitted sample.
- Every feature is a dimensionless log10 ratio, O(1) over SPARC.

Features (columns of galaxy_parameters.csv → anchored logs):

  x_M  = log10( M* / 10^9.5 Msun )                     stellar mass
         M_star_1e9_Msun; sample range ≈ [−2.5, 2.0]
  x_f  = log10( f_gas / 0.1 ),  f_gas = M_gas / M_b    gas richness
         sample range ≈ [−3.0, 1.0]
  x_S  = log10( Σ* / 100 Msun pc⁻² )                   stellar surface density
         Σ* = 0.5·M* / (π R_eff²)  [M* in Msun, R_eff in pc]
         (×10³ unit factor from 1e9 Msun / kpc²); replaces Hubble type T and the
         luminosity-based SBeff; sample range ≈ [−2.5, 2.5]
  x_R  = log10( R_disk / 2 kpc )                       disk scale length
         sample range ≈ [−1.0, 1.0]
"""
import numpy as np
import pandas as pd

# (name, human-readable definition) — order defines the g feature indices 1..n
PROPS = [
    ("x_M", "log10(M_star / 10^9.5 Msun)"),
    ("x_f", "log10(f_gas / 0.1), f_gas = M_gas/M_b"),
    ("x_S", "log10(Sigma_star / 100 Msun pc^-2), Sigma_star = 0.5 M*/(pi Reff^2)"),
    ("x_R", "log10(R_disk / 2 kpc)"),
]

PROP_NAMES = [name for name, _ in PROPS]
N_PROPS = len(PROPS)

REQUIRED_COLUMNS = ["galaxy", "M_star_1e9_Msun", "M_gas_1e9_Msun", "M_b_1e9_Msun",
                    "Reff_kpc", "Rdisk_kpc"]


def galaxy_features(params: pd.DataFrame, diagnostics: bool = False) -> pd.DataFrame:
    """Anchored feature table (galaxy + x_M, x_f, x_S, x_R) from galaxy_parameters.csv.

    Rows with any missing required column are dropped. Set diagnostics=True to print
    ranges, pairwise correlations, and VIFs (the "used properly" check).
    """
    p = params[REQUIRED_COLUMNS].dropna().copy()

    out = pd.DataFrame({"galaxy": p["galaxy"]})
    out["x_M"] = np.log10(p["M_star_1e9_Msun"].clip(lower=1e-4)) - 0.5
    out["x_f"] = np.log10(
        (p["M_gas_1e9_Msun"] / p["M_b_1e9_Msun"].clip(lower=1e-6)).clip(1e-4, 1.0)
    ) + 1.0
    # Σ* [Msun/pc²] = 0.5 · (M*·1e9 Msun) / (π · (Reff·1e3 pc)²) = 1e3·0.5·M*/(π·Reff²)
    sigma_star = 1e3 * 0.5 * p["M_star_1e9_Msun"] / (np.pi * p["Reff_kpc"].clip(lower=1e-3) ** 2)
    out["x_S"] = np.log10(sigma_star.clip(lower=1e-2)) - 2.0
    out["x_R"] = np.log10(p["Rdisk_kpc"].clip(lower=1e-2)) - np.log10(2.0)

    if diagnostics:
        X = out[PROP_NAMES]
        print("gprop features:")
        print(X.agg(["min", "median", "max", "std"]).round(3).to_string())
        print("\npairwise correlations:")
        print(X.corr().round(3).to_string())
        corr_inv = np.linalg.inv(X.corr().values)
        vif = {n: corr_inv[i, i] for i, n in enumerate(PROP_NAMES)}
        print("\nVIF: " + "  ".join(f"{n}={v:.2f}" for n, v in vif.items()))
    return out
