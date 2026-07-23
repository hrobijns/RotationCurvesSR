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

### Adapting to a different SLURM cluster

Everything here (the model code, the requirement for `Optim.jl`, the
general env setup) is cluster-agnostic. Only a handful of values are
CSD3/SL3-specific and need swapping for your own cluster:

- **`scripts/hpc/slurm_velocity2param.sh` / `slurm_velocity_gprop.sh`**:
  `-A ACCOUNT_CODE` (your project/account code — `mybalance`-equivalent, or
  omit `-A` if your cluster doesn't require one), `-p icelake` (your
  cluster's CPU partition name — check `sinfo`), `--cpus-per-task=76`
  (match a full node's core count on your cluster, or whatever chunk you
  want), and `--time=11:45:00` (your queue's max wall-clock, minus a bit of
  margin). **Whatever you set `--cpus-per-task` to must also be set as
  `procs=` in the Python call inside the same file** — the two aren't
  linked automatically.
- **`scripts/hpc/setup_env.sh`**: `module load miniconda/3` — module names
  are cluster-specific; run `module avail conda` (or similar) to find the
  equivalent, or skip module-loading entirely if conda/mamba is already on
  `PATH`. The `conda-forge --override-channels`, `LD_LIBRARY_PATH`, and
  Julia/`libstdc++` workarounds in that script were fixes for CSD3-specific
  quirks (a broken default conda channel, a git/OpenSSL library clash, an
  outdated system `libstdc++`) — they're harmless to leave in even if your
  cluster doesn't have the same issues, but if `pip install -e .` or the
  Julia precompile step fails in a new way, that's the likely place to
  look, and the comments there explain each workaround's root cause.
- **Filesystem**: CSD3 has a small backed-up home dir and a large
  non-backed-up `~/rds/hpc-work` for active job I/O — clone the repo
  wherever your cluster's equivalent scratch/project storage is, not a
  small home quota.

Not chained/resumable on any cluster — each `sbatch` is a single run up to
its time limit; there's no automatic continuation between submissions
(PySR's own checkpoint resume for this project's `TemplateExpressionSpec`
models turned out not to work reliably, so it was deliberately removed
rather than shipped half-working).
