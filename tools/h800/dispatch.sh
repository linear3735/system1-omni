#!/usr/bin/env bash
#
# Run one job on the H800 cluster and bring its evidence back.
#
# This runs on the machine that can reach the cluster over SSH: the self-hosted
# runner behind .github/workflows/gpu-h800.yml, or a laptop. It ships a
# `git archive` of the current checkout to the login node, submits exactly one
# Slurm job and waits for it. Only file transfer and scheduler commands happen
# on the login node; everything that compiles or touches a GPU runs inside the
# allocation (jobs/run_h800.sh).
#
#   tools/h800/dispatch.sh --check    read-only: SSH, Slurm, modules, space
#   tools/h800/dispatch.sh --plan     print what a run would do; touch nothing
#   tools/h800/dispatch.sh --submit   ship, submit, wait, fetch (explicit)
#
# Settings come from tools/h800/h800.env when it exists (copy h800.env.example);
# variables already set in the environment win over that file.
#
# Exit codes: 0 the job ran and its own status decided, 2 refused before submitting,
# 3 the job is still running or its accounting is not readable yet (unknown, not a
# failure: keep the JobID in artifacts/h800/submit-receipts/ and query later),
# 4 the evidence is larger than the cap.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$here/../.." && pwd)

if [ -f "$here/h800.env" ]; then
    # shellcheck source=/dev/null
    . "$here/h800.env"
fi

# Default to printing the plan: a real submission has to say --submit.
mode=${1:---plan}
case "$mode" in
    --check | --plan | --submit) ;;
    -h | --help)
        sed -n '2,17p' "${BASH_SOURCE[0]}" | cut -c3-
        exit 0
        ;;
    *)
        printf 'dispatch: unknown argument %s\n' "$mode" >&2
        exit 2
        ;;
esac

host=${H800_HOST:-h800-login}
root=${H800_ROOT:-}
partition=${H800_PARTITION:-training}
account=${H800_ACCOUNT:-llm}
qos=${H800_QOS:-for_llm}
time_limit=${H800_TIME:-00:30:00}
gpus=${H800_GPUS:-1}
cpus=${H800_CPUS:-8}
mem=${H800_MEM:-64G}
cuda_module=${H800_CUDA_MODULE:-cuda/13.0}
python_bin=${H800_PYTHON:-python3}
shared_tools=${H800_SHARED_TOOLS:-}
stages=${H800_STAGES:-env,build}
test_args=${H800_TEST_ARGS:-}
test_env=${H800_TEST_ENV:-}
wait_seconds=${H800_WAIT_SECONDS:-5400}
artifacts=${H800_ARTIFACT_DIR:-$repo_root/artifacts/h800}
job_name=${H800_JOB_NAME:-omni-h800}
# The build stage ships real compiler output (out/cuda-build), so the cap is
# larger here than for a text-only evidence set.
evidence_max_mb=${H800_EVIDENCE_MAX_MB:-512}
cancel_on_timeout=${H800_CANCEL_ON_TIMEOUT:-0}
allow_dirty=${H800_ALLOW_DIRTY:-0}

# Runs may only land under the personal root: a shared root mixes two projects'
# ledgers and evidence together.
case "$root" in
    '') ;;
    /aifs4su/shiheming/*) ;;
    *)
        printf 'dispatch: refusing H800_ROOT=%s; runs must land under /aifs4su/shiheming/\n' "$root" >&2
        exit 2
        ;;
esac

ssh_run() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" "$@"; }

# Slurm is not on PATH in a non-interactive SSH session on this cluster.
prologue='[ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; command -v sbatch >/dev/null 2>&1 || module load slurm/slurm >/dev/null 2>&1 || true;'

run_check() {
    ssh_run bash -s -- "$partition" "$root" <<'REMOTE'
set -u
[ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
command -v sbatch >/dev/null 2>&1 || module load slurm/slurm >/dev/null 2>&1 || true
partition=$1
root=$2
printf 'host       %s\n' "$(hostname -s)"
printf 'user       %s\n' "$(id -un)"
printf 'sbatch     %s\n' "$(command -v sbatch || printf 'missing')"
printf '\n-- partition %s\n' "$partition"
sinfo -p "$partition" -o '%P %a %D %t %G' 2>&1 | head -6
printf '\n-- own queue\n'
squeue -u "$(id -un)" -o '%i %T %M %j' 2>&1 | head -8
printf '\n-- cuda modules\n'
module avail cuda 2>&1 | tail -4
printf '\n-- space\n'
if [ -n "$root" ]; then
    df -h "$(dirname "$root")" 2>&1 | tail -1
else
    printf 'H800_ROOT is unset: a run would have nowhere to land\n'
fi
REMOTE
}

run_plan() {
    run_id=${H800_RUN_ID:-'<utc>-<sha7>'}
    remote_root=${root:-'<H800_ROOT>'}
    printf 'plan only: nothing is executed and nothing is written\n\n'
    printf '  1. git -C %s archive --format=tar.gz HEAD -> src.tar.gz\n' "$repo_root"
    printf '  2. ssh %s: umask 077; mkdir -p %s/%s\n' "$host" "$remote_root" "$run_id"
    printf '  3. ship src.tar.gz, run_h800.sbatch and run.env into that directory\n'
    printf '  4. ssh %s: %ssbatch --parsable --job-name=omni-h800 \\\n' "$host" "$prologue "
    printf '       --partition=%s --account=%s --qos=%s --gres=gpu:%s \\\n' \
        "$partition" "$account" "$qos" "$gpus"
    printf '       --cpus-per-task=%s --mem=%s --time=%s \\\n' "$cpus" "$mem" "$time_limit"
    printf '       --chdir=%s/%s %s/%s/run_h800.sbatch\n' "$remote_root" "$run_id" "$remote_root" "$run_id"
    printf '  5. poll squeue/sacct until the job leaves the queue (max %ss)\n' "$wait_seconds"
    printf '  6. fetch %s/%s/out into %s\n' "$remote_root" "$run_id" "$artifacts"
    printf '\nstages     %s\n' "$stages"
    printf 'python     %s\n' "$python_bin"
    printf 'received   %s\n' "$shared_tools"
}

run_job() {
    : "${root:?set H800_ROOT or copy tools/h800/h800.env.example to tools/h800/h800.env}"

    revision=$(git -C "$repo_root" rev-parse HEAD)
    short=$(git -C "$repo_root" rev-parse --short HEAD)
    run_id=${H800_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$short}
    run_dir=$root/$run_id

    # git archive only packs committed content, so an untracked entry point never
    # reaches the job. Shipping a run that cannot start costs a queue slot and an
    # allocation, so refuse by default.
    dirty=$(git -C "$repo_root" status --porcelain --untracked-files=all || true)
    if [ -n "$dirty" ]; then
        printf 'dispatch: refusing to ship a dirty checkout: %s changed paths\n' \
            "$(printf '%s\n' "$dirty" | wc -l | tr -d ' ')" >&2
        printf '%s\n' "$dirty" >&2
        if [ "$allow_dirty" = 1 ]; then
            printf 'dispatch: H800_ALLOW_DIRTY=1, shipping commit %s only\n' "$short" >&2
        else
            printf 'dispatch: commit first, or set H800_ALLOW_DIRTY=1 to accept commit %s\n' "$short" >&2
            exit 2
        fi
    fi

    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    git -C "$repo_root" archive --format=tar.gz --output="$tmp/src.tar.gz" HEAD
    # The entry point has to be inside the archive, or the failure moves into the
    # allocation instead of staying here.
    for required in tools/h800/jobs/run_h800.sh Cargo.toml; do
        tar -tzf "$tmp/src.tar.gz" | grep -qx "$required" || {
            printf 'dispatch: %s is missing from the archive (untracked? commit it first)\n' "$required" >&2
            exit 2
        }
    done
    cp "$here/jobs/run_h800.sbatch" "$tmp/run_h800.sbatch"

    {
        printf 'RUN_DIR=%q\n' "$run_dir"
        printf 'SRC_DIR=%q\n' "$run_dir/src"
        printf 'OUT_DIR=%q\n' "$run_dir/out"
        printf 'REVISION=%q\n' "$revision"
        printf 'STAGES=%q\n' "$stages"
        printf 'PYTHON=%q\n' "$python_bin"
        printf 'CUDA_MODULE=%q\n' "$cuda_module"
        printf 'SHARED_TOOLS=%q\n' "$shared_tools"
        printf 'TEST_ARGS=%q\n' "$test_args"
        printf 'TEST_ENV_FILE=%q\n' "$run_dir/gpu-test.env"
        printf 'JOB_NAME=%q\n' "$job_name-$run_id"
        printf 'RUN_ID=%q\n' "$run_id"
    } >"$tmp/run.env"

    if [ -n "$test_env" ]; then
        [ -f "$test_env" ] || { printf 'dispatch: H800_TEST_ENV %s does not exist\n' "$test_env" >&2; exit 2; }
        cp "$test_env" "$tmp/gpu-test.env"
    fi

    bundle_files=(src.tar.gz run.env run_h800.sbatch)
    [ -f "$tmp/gpu-test.env" ] && bundle_files+=(gpu-test.env)
    (cd "$tmp" && tar -czf bundle.tar.gz "${bundle_files[@]}")

    printf 'run        %s\n' "$run_id"
    printf 'revision   %s\n' "$short"
    printf 'stages     %s\n' "$stages"

    ssh_run "umask 077; mkdir -p '$run_dir'"
    ssh_run "umask 077; cat > '$run_dir/bundle.tar.gz'" <"$tmp/bundle.tar.gz"
    ssh_run "cd '$run_dir' && tar -xzf bundle.tar.gz && rm -f bundle.tar.gz" >/dev/null

    job_id=$(ssh_run "$prologue cd '$run_dir' && sbatch --parsable \
        --job-name='$job_name-$run_id' \
        --partition='$partition' --account='$account' --qos='$qos' \
        --gres='gpu:$gpus' --cpus-per-task='$cpus' --mem='$mem' --time='$time_limit' \
        --chdir='$run_dir' --output='$run_dir/slurm-%j.out' --error='$run_dir/slurm-%j.err' \
        '$run_dir/run_h800.sbatch'" | tail -1 | cut -d';' -f1)
    case "$job_id" in
        '' | *[!0-9]*) printf 'dispatch: sbatch did not return a job id (%s)\n' "$job_id" >&2; exit 1 ;;
    esac
    printf 'job        %s\n' "$job_id"

    # Submit receipt: the single observer resumes this JobID from here, and any
    # resubmit has to look at it first.
    mkdir -p "$artifacts/submit-receipts"
    receipt=$artifacts/submit-receipts/$run_id.json
    {
        printf '{\n'
        printf '  "run_id": "%s",\n' "$run_id"
        printf '  "job_id": "%s",\n' "$job_id"
        printf '  "job_name": "%s",\n' "$job_name-$run_id"
        printf '  "host": "%s",\n' "$host"
        printf '  "remote_dir": "%s",\n' "$run_dir"
        printf '  "revision": "%s",\n' "$revision"
        printf '  "stages": "%s",\n' "$stages"
        printf '  "submitted_utc": "%s"\n' "$(date -u +%FT%TZ)"
        printf '}\n'
    } >"$receipt"
    printf 'receipt    %s\n' "$receipt"

    deadline=$(( $(date +%s) + wait_seconds ))
    state=
    while :; do
        state=$(ssh_run "$prologue squeue -h -j '$job_id' -o %T" | tr -d '[:space:]')
        [ -n "$state" ] || state=PENDING_OR_GONE
        case "$state" in
            COMPLETED | FAILED | CANCELLED | TIMEOUT | NODE_FAIL | OUT_OF_MEMORY | PREEMPTED | BOOT_FAIL | DEADLINE) break ;;
        esac
        if [ "$(date +%s)" -ge "$deadline" ]; then
            printf 'dispatch: job %s is still %s after %ss; leaving it running\n' "$job_id" "$state" "$wait_seconds" >&2
            if [ "$cancel_on_timeout" = 1 ]; then
                printf 'dispatch: H800_CANCEL_ON_TIMEOUT=1, cancelling %s\n' "$job_id" >&2
                ssh_run "$prologue scancel '$job_id'" || true
            else
                printf 'dispatch: not cancelling (H800_CANCEL_ON_TIMEOUT=1 to cancel); receipt %s\n' "$receipt" >&2
            fi
            exit 3
        fi
        sleep 20
    done

    # Leaving the queue is not the same as accounting being readable: a missing
    # sacct record is not failure evidence, and not a reason to resubmit.
    final=
    grace_deadline=$(( $(date +%s) + 120 ))
    while :; do
        final=$(ssh_run "$prologue sacct -j '$job_id' -X -n -P -o State,ExitCode" | head -1 | tr -d '[:space:]')
        [ -n "$final" ] && break
        if [ "$(date +%s)" -ge "$grace_deadline" ]; then
            printf 'dispatch: no sacct record for job %s yet (last squeue state %s): unknown, not failed\n' "$job_id" "$state" >&2
            printf 'dispatch: keep this JobID and query again later; receipt %s\n' "$receipt" >&2
            exit 3
        fi
        sleep 10
    done
    printf 'slurm      %s\n' "$final"

    mkdir -p "$artifacts"
    # Evidence only: out/cuda-build is real compiler output and is kept, caches
    # and model weights are not.
    remote_kb=$(ssh_run "cd '$run_dir' && du -sk out 2>/dev/null | cut -f1" | tail -1 | tr -d '[:space:]')
    remote_mb=$(( ${remote_kb:-0} / 1024 ))
    printf 'evidence   %s MB (cap %s MB)\n' "$remote_mb" "$evidence_max_mb"
    if [ "$remote_mb" -gt "$evidence_max_mb" ]; then
        printf 'dispatch: evidence is %s MB, over the %s MB cap; refusing to pull it back.\n' \
            "$remote_mb" "$evidence_max_mb" >&2
        printf 'dispatch: narrow the stages, or raise H800_EVIDENCE_MAX_MB deliberately.\n' >&2
        exit 4
    fi
    ssh_run "cd '$run_dir' && tar --ignore-failed-read --exclude=target --exclude=__pycache__ \
        --exclude='*.gguf' --exclude='*.safetensors' \
        -czf - out run.env slurm-*.out slurm-*.err" >"$artifacts/$run_id.tar.gz"
    mkdir -p "$artifacts/$run_id"
    tar -xzf "$artifacts/$run_id.tar.gz" -C "$artifacts/$run_id"
    printf 'evidence   %s\n' "$artifacts/$run_id"

    if [ -f "$artifacts/$run_id/out/stage-status.tsv" ]; then
        printf '\n-- stages\n'
        cat "$artifacts/$run_id/out/stage-status.tsv"
    fi
    if [ -f "$artifacts/$run_id/out/exit-code" ]; then
        code=$(cat "$artifacts/$run_id/out/exit-code")
        printf 'exit       %s\n' "$code"
        exit "$code"
    fi
    printf 'exit       1 (the job left no out/exit-code; slurm said %s)\n' "${final:-$state}" >&2
    exit 1
}

case "$mode" in
    --check) run_check ;;
    --plan) run_plan ;;
    --submit) run_job ;;
esac
