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

## Cluster (any SLURM cluster with a similarly-sized CPU partition; written for CSD3/SL3)

```
ssh <user>@login-cpu.hpc.cam.ac.uk
git clone --depth 1 https://github.com/hrobijns/rotationcurvesSR.git ~/rds/hpc-work/rotationcurvesSR
cd ~/rds/hpc-work/rotationcurvesSR
bash scripts/hpc/setup_env.sh          # one-time: conda env + Julia depot precompile

# edit -A ACCOUNT_CODE -> your own project account in both scripts first
# (see `mybalance` for CSD3 project codes), and -p icelake -> your cluster's
# equivalent CPU partition name if not on CSD3
sbatch scripts/hpc/slurm_velocity2param.sh
sbatch scripts/hpc/slurm_velocity_gprop.sh
```

Each is a single, non-chained job requesting one full 76-core node for up
to ~12h (SL3's wall-clock cap; `--time=11:45:00` to leave startup margin).
They don't resume across separate submissions — if you want more search
time, just increase `--time` and/or `--cpus-per-task` / `procs=` in the
Python call to match your cluster's node size, rather than re-submitting.
Safe to run both at once (152 cores total, separate nodes, no shared state).

Check progress any time with `squeue --me` and by tailing
`outputs/production/hpc_v2param_<jobid>.log` /
`outputs/production/hpc_gprop_<jobid>.log`, or by reading the
`hall_of_fame.csv` PySR maintains under each run's output directory.
