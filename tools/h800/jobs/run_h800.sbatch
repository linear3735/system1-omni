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

# Slurm 把批处理脚本拷进自己的 spool 再执行，$0 因此不是 run 目录：
# run 目录由 dispatch.sh 作为第一个参数传入，缺省回落到 SLURM_SUBMIT_DIR。
run_dir=${1:-${SLURM_SUBMIT_DIR:-}}
[ -n "$run_dir" ] || { printf 'usage: run_h800.sbatch <run-dir>\n' >&2; exit 2; }
run_dir=$(cd "$run_dir" && pwd)
# run.env 里是普通赋值，不导出的话 exec 之后的 run_h800.sh 一个都收不到：
# 实测作业会在分配里立刻以 "RUN_DIR: RUN_DIR is required" 死掉。set -a 覆盖全部键，
# 包括 STAGES——它若丢失，作业会静默退回脚本自己的默认阶段集。
set -a
# shellcheck source=/dev/null
. "$run_dir/run.env"
set +a

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
