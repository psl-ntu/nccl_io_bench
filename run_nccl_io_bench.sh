#!/bin/bash
# Submit nccl_io_bench.py on Tuolumne via Flux.
# Compares Lustre, VAST, Rabbit, SHM, and DYAD-staged local storage to
# demonstrate I/O/NCCL fabric interference (and DYAD's mitigation of it).
#
# IMPORTANT: the inner `flux run` calls below must carry --exclusive, even
# though the outer `flux batch`/`flux --parent batch` submission is already
# --exclusive. Without it, every rank silently gets bound to the same tiny
# ~8-core cpuset (confirmed via os.sched_getaffinity) inherited from the
# batch driver process, instead of a proper full-node allocation -- this
# was root-caused by comparing against run_pecan_sair_flux.sh's real
# training invocation (which does carry --exclusive on its inner `flux
# run`) and produced a severe, filesystem-independent CPU-oversubscription
# artifact (every phase including pure-tmpfs shm showed 10-15x spurious
# "NCCL interference") until fixed.
#
# Usage: bash submit_nccl_io_bench.sh [N_NODES]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_NODES=${1:-4}
NUM_GPUS_PER_NODE=4
TOTAL_RANKS=$(( NUM_NODES * NUM_GPUS_PER_NODE ))

# Background I/O readers per rank (nccl_io_bench.py --io-workers). Default 8
# matches PECAN's real num_workers=8.
IO_WORKERS=${IO_WORKERS:-8}

# Distinguishes scratch paths/KVS namespace across concurrently-running
# invocations of this script (e.g. an IO_WORKERS=8 reproducibility run and an
# IO_WORKERS=1 confound-isolation run submitted at the same time) so they
# don't collide on the same Lustre/VAST/shm paths or DYAD KVS namespace.
# RABBIT_DIR/DYAD_DIR need no such tag: DW_JOB_xfssmall is already unique per
# flux batch job (separate Rabbit NVMe reservation each).
RUN_TAG=${RUN_TAG:-default}

source /p/lustre5/wang116/tuolumne/python-venv/pecan-milan-venv/bin/activate

BENCH=${SCRIPT_DIR}/nccl_io_bench.py
STAGE=${SCRIPT_DIR}/dyad_stage.py

export DYAD_KVS_NAMESPACE=nccl_io_dyad_bench_${RUN_TAG}
export DYAD_DTL_MODE=FLUX_RPC

# Scratch dirs — one per filesystem under test
LUSTRE_DIR=/p/lustre5/wang116/tuolumne/bench_scratch_${RUN_TAG}
VAST_DIR=/p/vast1/wang116/bench_scratch_${RUN_TAG}   # adjust to your VAST allocation
RABBIT_DIR=${DW_JOB_xfssmall}/bench_scratch
SHM_DIR=/dev/shm/bench_scratch_${RUN_TAG}
DYAD_DIR=${DW_JOB_xfssmall}/dyad_managed   # DYAD-managed dir; also lives on Rabbit NVMe

mkdir -p "$LUSTRE_DIR" "$VAST_DIR" "$RABBIT_DIR" "$SHM_DIR"
#flux run -N $NUM_NODES -n $NUM_NODES echo "RABBIT DIR:" ${RABBIT_DIR}

echo "=== nccl_io_bench: ${NUM_NODES} nodes / ${TOTAL_RANKS} GPUs, io-workers=${IO_WORKERS}, run-tag=${RUN_TAG} ==="

run_bench() {
    local label=$1
    local fs_path=$2
    local extra_args=$3
    echo ""
    echo "--- $label ($fs_path) ---"
    flux run \
        -N "$NUM_NODES" \
        -n "$TOTAL_RANKS" \
        --gpus-per-task=1 \
        -o mpibind=off \
        --exclusive \
        python "$BENCH" \
            --fs-path    "$fs_path" \
            --label      "$label" \
            --tensor-mb  0.076 \
            --iters      100 \
            --warmup     20 \
            --io-mode    paced \
            --io-workers "$IO_WORKERS" \
            --batch-size 64 \
            --sample-kb  286 \
            --file-mb    512 \
            $extra_args
}

run_bench "shm"    "$SHM_DIR"
run_bench "rabbit" "$RABBIT_DIR"
# --no-cleanup: the dyad phase below reuses these same files as the PFS
# master copy (nccl_io_bench.py deletes its test files by default).
run_bench "lustre" "$LUSTRE_DIR" "--no-cleanup"
run_bench "vast"   "$VAST_DIR"

# --- dyad: same measurement, but files are staged into $DYAD_DIR the way a
# real DataLoader worker would -- on first touch, each rank reads its own
# file from the PFS (reusing the files the "lustre" phase above already
# created at $LUSTRE_DIR) and produce()s the local copy into $DYAD_DIR, so
# this measures genuine post-DYAD-staging local reads rather than files
# written directly in place.
echo ""
echo "--- dyad ($DYAD_DIR) ---"
flux exec -r all mkdir -p "$DYAD_DIR"
flux kvs namespace create "$DYAD_KVS_NAMESPACE"
dyad start -p "$DYAD_DIR"
flux run -N "$NUM_NODES" --tasks-per-node="$NUM_GPUS_PER_NODE" --exclusive \
    python3 "$STAGE" --pfs-path "$LUSTRE_DIR" --fs-path "$DYAD_DIR"
STAGE_RC=$?
if [ "$STAGE_RC" -eq 0 ]; then
    run_bench "dyad" "$DYAD_DIR"
else
    echo "SKIPPING dyad measurement: staging failed (exit ${STAGE_RC})"
fi
dyad stop
flux kvs namespace remove "$DYAD_KVS_NAMESPACE"
flux exec -r all rm -rf "$DYAD_DIR"
rm -rf "$LUSTRE_DIR"
