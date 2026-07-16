#!/bin/bash


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# The "dyad" phase in run_nccl_io_bench.sh calls `flux module load dyad.so`,
# which dlopen's the module inside each job broker at broker startup -- using
# the broker's own process environment, which is inherited from this
# submitting shell's environment, not from exports made later inside
# run_nccl_io_bench.sh. Must be set here, before `flux batch`.
DYAD_INSTALL_PREFIX=/p/lustre5/wang116/tuolumne/sources/dyad-pdsw/install
export PATH=${DYAD_INSTALL_PREFIX}/bin:${DYAD_INSTALL_PREFIX}/sbin:${PATH}
# lib64 has DYAD's own libs; lib has Margo/Mercury/Argobots/json-c (this build
# has DYAD_ENABLE_MARGO_DATA=TRUE, so libdyad_client.so depends on libmercury
# even when the DTL mode actually used at runtime is FLUX_RPC).
export LD_LIBRARY_PATH=${DYAD_INSTALL_PREFIX}/lib64:${DYAD_INSTALL_PREFIX}/lib:/opt/cray/pe/lib64/cce:${LD_LIBRARY_PATH}

NUM_NODES=${1:-4}
flux batch -N ${NUM_NODES} --exclusive -t 20m -q pdebug -S dw=xfs_small ${SCRIPT_DIR}/run_nccl_io_bench.sh ${NUM_NODES}
