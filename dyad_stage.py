#!/usr/bin/env python3
"""
dyad_stage.py — stage per-rank test files for nccl_io_bench.py the way a real
PECAN DataLoader worker does: the worker is both producer and consumer. On
first touch of a file, it checks DYAD locally (a cold miss, since nothing has
been produced yet), falls back to reading the master copy directly from the
shared parallel file system (PFS), writes that data into its own node-local
DYAD-managed directory, and calls dyad_produce() on the local copy -- so any
later consume() call (from this rank in a subsequent epoch, or from another
rank/node that happens to want the same file) can be served from a DYAD
cache instead of going back to the PFS.

There is no fixed "producer node" / "consumer node" split: every rank runs
the identical self-serve staging logic, matching pecan/dataset_dyad.py's
actual get_metadata() -> [miss -> read+produce] -> consume() pattern.

The PFS master copy is expected to already exist (e.g. from an earlier
"lustre" or "vast" phase of the same nccl_io_bench.py run, which creates one
file per rank at --pfs-path under the exact naming nccl_io_bench.py uses).
After this script exits, --fs-path contains, on every node, the local
DYAD-staged copy of that node's own ranks' files -- already present with the
correct size, so nccl_io_bench.py itself needs no modification.

For cheap correctness testing on a single physical node (no real multi-node
allocation needed), set DYAD_LOCAL_TEST=1: each simulated Flux broker (e.g.
under `flux start --test-size=2`) gets its own subdirectory under
--fs-path, mirroring the same convention pecan/dataset_dyad.py uses.

Usage (inside an allocation with dyad.so already loaded on all brokers and a
KVS namespace already created):
    flux run -N <nodes> -n <total-ranks> python dyad_stage.py \\
        --pfs-path /p/lustre5/.../bench_scratch \\
        --fs-path /l/ssd/bench/dyad_managed
"""
import argparse
import os
import shutil
import subprocess

from mpi4py import MPI

from pydyad import Dyad
from pydyad.bindings import DTLMode, DTLCommMode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pfs-path", required=True,
                   help="Shared PFS directory holding each rank's master file "
                        "(same naming as nccl_io_bench.py: _nccl_io_bench_rank{R:04d}.bin)")
    p.add_argument("--fs-path", required=True,
                   help="DYAD-managed (node-local) directory")
    return p.parse_args()


def get_broker_rank():
    return int(subprocess.check_output(["flux", "getattr", "rank"]).decode().strip())


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_local_test = os.getenv("DYAD_LOCAL_TEST", "0") == "1"
    broker_rank = get_broker_rank()
    managed_dir = os.path.join(args.fs_path, str(broker_rank)) if is_local_test else args.fs_path
    os.makedirs(managed_dir, exist_ok=True)

    dyad_io = Dyad()
    dyad_io.init(debug=False, check=False, shared_storage=False, reinit=False,
                 async_publish=True, fsync_write=False, key_depth=3, service_mux=1,
                 key_bins=1024, kvs_namespace=os.environ["DYAD_KVS_NAMESPACE"],
                 prod_managed_path=managed_dir, cons_managed_path=managed_dir,
                 dtl_mode=DTLMode.DYAD_DTL_FLUX_RPC, dtl_comm_mode=DTLCommMode.DYAD_COMM_RECV)

    fname = f"_nccl_io_bench_rank{rank:04d}.bin"
    pfs_fpath = os.path.join(args.pfs_path, fname)
    local_fpath = os.path.join(managed_dir, fname)

    t0 = MPI.Wtime()
    # First touch: ask DYAD whether this file is already known locally. On a
    # cold cache this is a miss (nothing has been produced yet for this
    # file), mirroring dataset_dyad.py's get_metadata() check.
    file_obj = dyad_io.get_metadata(fname=local_fpath, should_wait=False, raw=True)
    if file_obj:
        dyad_io.consume_w_metadata(local_fpath, file_obj)
        dyad_io.free_metadata(file_obj)
        mode = "cache-hit"
    else:
        # Cold miss: read the master copy from the PFS, stage it locally,
        # then produce() it so future consumers (this rank next epoch, or
        # any other rank/node) can be served from DYAD instead of the PFS.
        shutil.copyfile(pfs_fpath, local_fpath)
        dyad_io.produce(local_fpath)
        mode = "staged-from-pfs"
    elapsed = MPI.Wtime() - t0

    size = os.path.getsize(local_fpath)
    print(f"[dyad_stage] rank {rank} (broker {broker_rank}): {mode} "
          f"{local_fpath} ({size/1024/1024:.1f} MB) in {elapsed*1000:.1f} ms", flush=True)

    comm.Barrier()
    dyad_io.finalize()

    if rank == 0:
        print("[dyad_stage] staging complete", flush=True)


if __name__ == "__main__":
    main()
