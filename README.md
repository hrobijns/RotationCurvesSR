# rotationcurvesSR

Symbolic regression on SPARC galaxy rotation curves — rediscovering dark
matter density profiles from data. Lean pipeline code; figures/analysis are
tracked separately until publication.

## Layout

- `models/velocity2param.py` — f(x, γ), per-galaxy free shape parameter γ.
- `models/velocity_gprop.py` — joint f(x, γ) + γ = g(galaxy properties).
- `models/velocity.py`, `models/density*.py` — earlier 1-param variants.
- `models/losses/*.jl` — Julia loss kernels, loaded via `models/julia_losses.py`.
- `datawrangling.py` — SPARC `.dat` ingestion (`produce_SPARC_df`).
- `data/SPARC/` — SPARC rotation curve data + `galaxy_parameters.csv`.
- `scripts/run_v2p_toy.py`, `scripts/run_gprop_toy.py` — small smoke-test runs.
- `scripts/hpc/` — CSD3 (Cambridge HPC) setup + SLURM job scripts.

## Local setup

```
pip install -e .
python scripts/run_v2p_toy.py
python scripts/run_gprop_toy.py
```

## Cluster (CSD3, SL3)

```
ssh <user>@login-cpu.hpc.cam.ac.uk
git clone --depth 1 https://github.com/hrobijns/rotationcurvesSR.git ~/rds/hpc-work/rotationcurvesSR
cd ~/rds/hpc-work/rotationcurvesSR
bash scripts/hpc/setup_env.sh          # edit -A ACCOUNT_CODE in slurm_*.sh first
bash scripts/hpc/submit_chain.sh scripts/hpc/slurm_velocity2param.sh 4
bash scripts/hpc/submit_chain.sh scripts/hpc/slurm_velocity_gprop.sh 4
```

SL3 caps jobs at 12h / 448 cores cluster-wide — `submit_chain.sh` links
multiple jobs via SLURM dependencies, resuming from PySR's checkpoint
(`warm_start=True`, fixed `run_id`) across the chain.
