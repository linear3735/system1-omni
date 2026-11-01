#!/usr/bin/env bash
#
# The stages that run inside an H800 allocation.
#
# Submitted by tools/h800/dispatch.sh through jobs/run_h800.sbatch. It can also
# be run by hand from an interactive allocation:
#
#   srun --partition=training --account=llm --qos=for_llm --gres=gpu:1 \
#        --cpus-per-task=8 --mem=64G --time=00:30:00 --pty bash -l
#   RUN_DIR=$PWD ... tools/h800/jobs/run_h800.sh
#
# Stages
#   env     the hardware, driver, toolkit and revision this run actually saw
#   build   build every declared CUDA backend, then list the ELF images inside
#   tier3   report what the Tier 3 contract would run; executes nothing
#   tests   run the GPU-gated tests named by TEST_ARGS
#
# Every stage runs even when an earlier one failed: the evidence of a partial
# run is worth keeping. The exit status is 1 if any stage failed.

set -uo pipefail
umask 077

run_dir=${RUN_DIR:?RUN_DIR is required}
src=${SRC_DIR:-$run_dir/src}
out=${OUT_DIR:-$run_dir/out}
stages=${STAGES:-env,build}
python_bin=${PYTHON:-python3}
cuda_module=${CUDA_MODULE:-cuda/13.0}
shared_tools=${SHARED_TOOLS:-}
revision=${REVISION:-unknown}
test_args=${TEST_ARGS:-}
test_env_file=${TEST_ENV_FILE:-}

case "$run_dir" in
    /aifs4su/*) ;;
    *)
        printf 'refusing to run outside /aifs4su (RUN_DIR=%s)\n' "$run_dir" >&2
        exit 2
        ;;
esac
case "$(hostname -s)" in
    slogin* | login* | master*)
        printf 'refusing to run stages on a login node (%s)\n' "$(hostname -s)" >&2
        exit 2
        ;;
esac

[ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
module load "$cuda_module"

if [ -n "$shared_tools" ]; then
    export CARGO_HOME=$shared_tools/cargo
    export RUSTUP_HOME=$shared_tools/rustup
    # This cluster's rustup has no default toolchain configured.
    export RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-1.98.1}
    export PATH=$CARGO_HOME/bin:$PATH
fi
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$run_dir/target}
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$out"
status_file=$out/stage-status.tsv
: >"$status_file"
failed=0

stage_env() {
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-none}"
        printf 'job_name=%s\n' "${JOB_NAME:-none}"
        printf 'run_id=%s\n' "${RUN_ID:-none}"
        printf 'node=%s\n' "$(hostname -s)"
        printf 'utc=%s\n' "$(date -u +%FT%TZ)"
        printf 'revision=%s\n' "$revision"
        printf 'allocation=%s/%scpu/%sM/gpu:%s\n' \
            "${SLURM_JOB_PARTITION:-none}" "${SLURM_CPUS_PER_TASK:-?}" \
            "${SLURM_MEM_PER_NODE:-?}" "${SLURM_GPUS_ON_NODE:-?}"
        printf '\n-- host\n'
        uname -a
        printf '\n-- gpu\n'
        nvidia-smi -L || true
        nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
            --format=csv,noheader || true
        printf '\n-- cuda\n'
        nvcc --version || true
        cuobjdump --version | head -2 || true
        printf '\n-- python and rust\n'
        "$python_bin" -V || true
        cargo --version || true
        rustc --version || true
    } 2>&1 | tee "$out/env.txt"
}

stage_build() {
    local code=0
    local build_dir=$out/cuda-build
    printf 'building declared CUDA backends into %s\n' "$build_dir"
    "$python_bin" "$src/src/backends/cuda/build_backends.py" \
        --repo-root "$src" --out "$build_dir" 2>&1 | tee "$out/build.log"
    code=${PIPESTATUS[0]}
    printf '\n-- ELF images inside every built library\n' | tee "$out/elf.txt"
    local lib
    for lib in "$build_dir"/*/*.so; do
        [ -f "$lib" ] || continue
        printf '== %s\n' "$lib" | tee -a "$out/elf.txt"
        cuobjdump --list-elf "$lib" 2>&1 | tee -a "$out/elf.txt"
    done
    return "$code"
}

stage_tier3() {
    "$python_bin" - "$src" <<'PY' 2>&1 | tee "$out/tier3.txt"
import glob, json, os, sys

root = sys.argv[1]
patterns = ["src/backends/cuda/*.backend.json", "src/backends/cuda/*/*.backend.json"]
rows = []
for pattern in patterns:
    for path in sorted(glob.glob(os.path.join(root, pattern))):
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        numerics = manifest.get("numerics")
        reference = manifest.get("reference")
        rows.append(
            {
                "path": os.path.relpath(path, root),
                "name": manifest.get("name") or manifest.get("label"),
                "status": manifest.get("status", "unspecified"),
                "tolerance": (numerics or {}).get("tolerance") if isinstance(numerics, dict) else None,
                "entrypoint": (reference or {}).get("entrypoint") if isinstance(reference, dict) else None,
            }
        )

print("declared CUDA backends: %d" % len(rows))
for row in rows:
    print(
        "  %-28s status=%-12s tolerance=%-10s entrypoint=%s"
        % (row["name"], row["status"], row["tolerance"], row["entrypoint"])
    )

ready = [row for row in rows if row["status"] == "validated" and row["entrypoint"]]
print()
if not ready:
    print(
        "Tier 3 has nothing to run: no backend declares status=validated with a\n"
        "reference.entrypoint. This stage reports the declarations instead of\n"
        "inventing a parity harness."
    )
    sys.exit(0)

print("Tier 3 material is declared but this harness cannot execute it:")
for row in ready:
    print("  %s -> %s" % (row["path"], row["entrypoint"]))
print(
    "\nsrc/backends/cuda/contract.md does not define how a reference entrypoint is\n"
    "invoked, which device it runs on, or how its output is compared. Decide that\n"
    "convention (upstream) before wiring the comparison here."
)
sys.exit(1)
PY
}

stage_tests() {
    if [ -z "$test_args" ]; then
        printf 'TEST_ARGS is empty: no GPU test target was named, so none ran.\n'
        return 0
    fi
    if [ -n "$test_env_file" ] && [ -f "$test_env_file" ]; then
        sed "s|{cuda_build}|$out/cuda-build|g" "$test_env_file" >"$out/test-env.sh"
        printf -- '-- test environment\n'
        cat "$out/test-env.sh"
        set -a
        # shellcheck source=/dev/null
        . "$out/test-env.sh"
        set +a
    fi
    # TEST_ARGS is a list of cargo arguments, so word splitting is intended.
    # shellcheck disable=SC2086
    cargo test --locked $test_args 2>&1 | tee "$out/tests.log"
    return "${PIPESTATUS[0]}"
}

run_stage() {
    local name=$1 code=0
    printf '\n===== stage %s =====\n' "$name"
    if "stage_$name"; then
        printf '%s\tok\n' "$name" >>"$status_file"
    else
        code=$?
        printf '%s\tfailed(%s)\n' "$name" "$code" >>"$status_file"
        failed=1
    fi
}

write_summary() {
    "$python_bin" - "$run_dir" "$out" "$revision" "$stages" "$failed" <<'PY'
import json, os, sys

run_dir, out, revision, stages, failed = sys.argv[1:6]
status_path = os.path.join(out, "stage-status.tsv")
stage_rows = []
with open(status_path, encoding="utf-8") as handle:
    for line in handle:
        name, _, state = line.rstrip("\n").partition("\t")
        if name:
            stage_rows.append({"stage": name, "state": state})

summary = {
    "revision": revision,
    "requested_stages": [s for s in stages.split(",") if s],
    "stages": stage_rows,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "run_id": os.environ.get("RUN_ID"),
    "job_name": os.environ.get("JOB_NAME"),
    "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "exit_code": int(failed),
    "artifacts": sorted(
        os.path.relpath(os.path.join(root, name), out)
        for root, _, names in os.walk(out)
        for name in names
        if name != "summary.json"
    ),
}
with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

for stage in ${stages//,/ }; do
    run_stage "$stage"
done

write_summary | tee "$out/summary.txt"
printf '%s' "$failed" >"$out/exit-code"
printf '\nexit=%s\n' "$failed"
exit "$failed"
