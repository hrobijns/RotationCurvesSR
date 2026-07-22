#!/bin/bash
# Chain N SL3 jobs (12h cap each) via SLURM dependency, so a PySR search
# continues across wall-clock blocks using warm_start + checkpoint.pkl
# (see run_id in slurm_velocity2param.sh / slurm_velocity_gprop.sh).
#
# Usage:
#   bash scripts/hpc/submit_chain.sh scripts/hpc/slurm_velocity2param.sh 4
#
# Each job is submitted --dependency=afterany on the previous one, so job 2
# starts once job 1 ends (success, timeout, or crash) rather than waiting on
# success only — a 12h timeout is the expected, not the failure, case here.
set -euo pipefail

SCRIPT="${1:?usage: submit_chain.sh <slurm_script> [n_jobs=3]}"
N="${2:-3}"

jobid=""
all_ids=()
for i in $(seq 1 "$N"); do
  if [ -z "$jobid" ]; then
    jobid=$(sbatch --parsable "$SCRIPT")
  else
    jobid=$(sbatch --parsable --dependency=afterany:"$jobid" "$SCRIPT")
  fi
  all_ids+=("$jobid")
  echo "chain $i/$N -> job $jobid"
done

echo "Watch with: squeue --me"
echo "Cancel whole chain with: scancel ${all_ids[*]}"
