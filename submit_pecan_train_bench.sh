#!/bin/bash
# Submit pecan_train_bench.py on Tuolumne via Flux -- the real EGNN model
# and real training phases (DataLoader, H2D, forward, backward+all-reduce,
# optimizer step), with synthetic (dependency-free) data. Companion to
# nccl_io_bench.py's isolated all-reduce microbenchmark: this one measures
# the real compute+comm workload the paper's bwd+nccl numbers come from.
#
# Usage: bash submit_pecan_train_bench.sh [N_NODES] [ITERS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_NODES=${1:-4}
NUM_GPUS_PER_NODE=4
TOTAL_RANKS=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
ITERS=${2:-100}

BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}

source /p/lustre5/wang116/tuolumne/python-venv/pecan-milan-venv/bin/activate

echo "=== pecan_train_bench: ${NUM_NODES} nodes / ${TOTAL_RANKS} GPUs, iters=${ITERS}, batch=${BATCH_SIZE}, workers=${NUM_WORKERS} ==="

flux run \
    -N "$NUM_NODES" \
    -n "$TOTAL_RANKS" \
    --gpus-per-task=1 \
    -o mpibind=off \
    --exclusive \
    python "${SCRIPT_DIR}/pecan_train_bench.py" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --iters "$ITERS"
