#!/bin/bash -l
#
# The allocation wrapper. tools/h800/dispatch.sh ships this file next to
# src.tar.gz and run.env, and overrides every directive below on the sbatch
# command line, so the values here are the record of what a hand submission
# would ask for rather than the settings a dispatched run actually uses.
#
#SBATCH --job-name=omni-h800
#SBATCH --partition=training
#SBATCH --account=llm
#SBATCH --qos=for_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through sbatch}"

run_dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
. "$run_dir/run.env"

# The archive is packed at the repository root, so it has to unpack into src/
# for SRC_DIR to really be the repository root.
src_dir=${SRC_DIR:-$run_dir/src}
mkdir -p "$src_dir" "$OUT_DIR"
tar -xzf "$run_dir/src.tar.gz" -C "$src_dir"

# run.env defines these; export them so the summary can record the same identity.
export RUN_ID=${RUN_ID:-} JOB_NAME=${JOB_NAME:-}

entry=$src_dir/tools/h800/jobs/run_h800.sh
if [ ! -x "$entry" ]; then
    printf 'run_h800: entry missing: %s (was it committed before dispatch?)\n' "$entry" >&2
    exit 2
fi
exec "$entry"
