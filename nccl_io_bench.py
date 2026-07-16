#!/usr/bin/env python3
"""
nccl_io_bench.py — NCCL / filesystem I/O interference microbenchmark.

Measures whether concurrent DataLoader-style I/O degrades NCCL all-reduce latency.

Default mode (--io-mode paced): I/O workers read exactly one batch worth of data
per training iteration, synchronized with the all-reduce, emulating real PyTorch
DataLoader prefetch.  This avoids the CPU/cache thrashing of naive tight-loop I/O
benchmarks that spin 32+ threads continuously regardless of training speed.

Two metrics are reported per iteration:
  nccl_ms  — CUDA-timed all-reduce latency (does I/O slow the collective?)
  dl_wait  — time the main thread blocked waiting for prefetch after nccl completed
              (was I/O too slow to hide behind compute?)

Interpretation:
  nccl_ms increases vs baseline → fabric TC contention (Lustre kfabric vs NCCL on
      same Slingshot traffic class).
  dl_wait > 0                   → I/O bandwidth cannot keep up with iteration rate
      (DataLoader stall; consider node-local NVMe or DYAD caching).
  Both near zero                → filesystem is well-isolated (VAST/NFS over TCP, or
      fast node-local storage with no fabric sharing).

Quick start (mpi4py / Flux):
    flux run -N 4 -n 16 python nccl_io_bench.py --fs-path /p/lustre5/yourdir

Quick start (torchrun):
    torchrun --nproc_per_node=4 nccl_io_bench.py --fs-path /path/to/lustre5

Compare two filesystems:
    flux run -N 4 -n 16 python nccl_io_bench.py --fs-path /p/lustre5/x --label lustre
    flux run -N 4 -n 16 python nccl_io_bench.py --fs-path /p/vast1/x   --label vast

PECAN defaults (--batch-size 64 --sample-kb 286 --io-workers 8):
    Each worker reads ceil(64/8)=8 samples x 286 KB = ~2.3 MB per iteration.
    This matches PECAN's graph-cache DataLoader with prefetch_factor=2.
"""

import os
import time
import math
import random
import multiprocessing as mp
import argparse
import statistics

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="NCCL / filesystem I/O interference benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fs-path", required=True,
                   help="Filesystem path to probe (e.g. /p/lustre5/yourdir or /dev/shm/scratch)")
    p.add_argument("--label", default="",
                   help="Short label for output (e.g. 'lustre' or 'vast')")
    p.add_argument("--tensor-mb", type=float, default=0.076,
                   help="All-reduce tensor size in MB (default: 0.076 = 76 KB, PECAN/EGNN 19K params)")
    p.add_argument("--iters", type=int, default=100,
                   help="Measurement iterations per phase")
    p.add_argument("--warmup", type=int, default=20,
                   help="Warmup iterations (not recorded)")

    # I/O worker configuration
    p.add_argument("--io-mode", choices=["paced", "tight"], default="paced",
                   help="'paced': workers read one batch per iteration (DataLoader emulation); "
                        "'tight': workers spin in a continuous loop (stress test)")
    p.add_argument("--io-workers", type=int, default=8,
                   help="Background reader threads per rank")
    p.add_argument("--batch-size", type=int, default=64,
                   help="[paced] Samples per GPU per iteration")
    p.add_argument("--sample-kb", type=float, default=286.0,
                   help="[paced] Bytes per sample in KB (default: 286 = PECAN graph-cache sample)")
    p.add_argument("--read-kb", type=int, default=256,
                   help="[tight] Per-read chunk size in KB")
    p.add_argument("--file-mb", type=int, default=512,
                   help="Per-rank temp file size in MB")

    p.add_argument("--no-cleanup", action="store_true",
                   help="Keep temp files after the run")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run baseline (no I/O) phase only")
    p.add_argument("--io-only", action="store_true",
                   help="Run I/O phase only (skip baseline)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def init_dist():
    """Return (rank, world_size, local_rank, device)."""
    try:
        from mpi4py import MPI
        from datetime import timedelta
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world = comm.Get_size()
        local_rank = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        if world > 1:
            hostname = MPI.Get_processor_name()
            hosts = MPI.COMM_WORLD.allgather(hostname)
            os.environ["MASTER_ADDR"] = hosts[0]
        device_id = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        dist.init_process_group(
            backend="nccl",
            world_size=world,
            rank=rank,
            timeout=timedelta(seconds=600),
            device_id=device_id,
        )
    except (ImportError, Exception):
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    return rank, world, local_rank, device


# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------

def create_test_file(path: str, size_bytes: int, rank: int) -> str:
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, f"_nccl_io_bench_rank{rank:04d}.bin")
    if os.path.exists(fpath) and os.path.getsize(fpath) == size_bytes:
        return fpath
    chunk = 1 << 20
    rng = random.Random(rank)
    with open(fpath, "wb") as f:
        written = 0
        while written < size_bytes:
            n = min(chunk, size_bytes - written)
            f.write(bytes(rng.getrandbits(8) for _ in range(n)))
            written += n
    return fpath


def remove_test_file(fpath: str):
    try:
        os.remove(fpath)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Background I/O reader
# ---------------------------------------------------------------------------

class IOReaderProcess:
    """
    Background reader running in a genuinely separate OS process (via
    multiprocessing, fork start method), with two modes:

    tight: spins in a continuous read loop until stop() is called.
           Stresses storage bandwidth but causes CPU/cache contention
           unrelated to filesystem network path (use only as a stress test).

    paced: reads exactly bytes_per_iter bytes per iteration, triggered by
           the main training loop via trigger_iter() / wait_done().
           Emulates a real PyTorch DataLoader worker (num_workers > 0):
           the worker reads one batch while the GPU runs backward +
           all-reduce, then blocks until the next iteration is signalled.

    This intentionally uses a separate process rather than a Python thread:
    a real DataLoader worker is its own process with its own GIL, so it
    cannot contend for the GIL with the main thread's dist.all_reduce()
    call the way a threading.Thread reader would. Using fork (not spawn)
    matches perf_analysis.md's own finding that spawn's per-worker startup
    cost is prohibitive on this system (Experiment 2).
    """

    _ctx = mp.get_context("fork")

    def __init__(self, fpath: str, chunk_bytes: int, mode: str = "paced",
                 bytes_per_iter: int = 0, log_affinity: bool = False):
        self.fpath = fpath
        self.chunk_bytes = chunk_bytes
        self.mode = mode
        self.bytes_per_iter = bytes_per_iter
        self.log_affinity = log_affinity
        self._stop = self._ctx.Event()
        self._trigger = self._ctx.Event()   # main sets  → start reading this iter
        self._done = self._ctx.Event()      # worker sets → batch read complete
        self._bytes_read = self._ctx.Value("q", 0)
        self._read_count = self._ctx.Value("q", 0)
        self._proc = None

    @property
    def bytes_read(self):
        return self._bytes_read.value

    @property
    def read_count(self):
        return self._read_count.value

    def start(self):
        self._proc = self._ctx.Process(target=self._run, daemon=True)
        self._proc.start()

    def stop(self):
        self._stop.set()
        self._trigger.set()  # unblock any waiting process

    def join(self, timeout: float = 5.0):
        if self._proc is not None:
            self._proc.join(timeout=timeout)

    def trigger_iter(self):
        """Signal worker to start reading next batch (paced mode)."""
        self._done.clear()
        self._trigger.set()

    def wait_done(self, timeout: float = 60.0) -> bool:
        """Block until this iteration's reads are complete. Returns True if finished."""
        return self._done.wait(timeout=timeout)

    def _run(self):
        file_size = os.path.getsize(self.fpath)
        rng = random.Random()
        bytes_read = 0
        read_count = 0

        if self.log_affinity:
            print(f"[nccl_io_bench] reader pid={os.getpid()} "
                  f"affinity={sorted(os.sched_getaffinity(0))}", flush=True)

        with open(self.fpath, "rb") as f:
            if self.mode == "tight":
                buf = bytearray(self.chunk_bytes)
                max_off = max(0, file_size - self.chunk_bytes)
                while not self._stop.is_set():
                    offset = rng.randint(0, max_off)
                    f.seek(offset)
                    n = f.readinto(buf)
                    bytes_read += n
                    read_count += 1
            else:
                # paced: wait for trigger, read bytes_per_iter, signal done, repeat
                while not self._stop.is_set():
                    triggered = self._trigger.wait(timeout=0.05)
                    if not triggered:
                        continue
                    self._trigger.clear()
                    if self._stop.is_set():
                        break

                    remaining = self.bytes_per_iter
                    while remaining > 0 and not self._stop.is_set():
                        n_req = min(self.chunk_bytes, remaining)
                        max_off = max(0, file_size - n_req)
                        offset = rng.randint(0, max_off)
                        f.seek(offset)
                        buf = bytearray(n_req)
                        n = f.readinto(buf)
                        bytes_read += n
                        read_count += 1
                        remaining -= n

                    self._bytes_read.value = bytes_read
                    self._read_count.value = read_count
                    self._done.set()

        self._bytes_read.value = bytes_read
        self._read_count.value = read_count


# ---------------------------------------------------------------------------
# Benchmark phase
# ---------------------------------------------------------------------------

def run_phase(
    label: str,
    iters: int,
    warmup: int,
    tensor: torch.Tensor,
    io_workers: int,
    io_fpath: str,
    args,
    rank: int,
) -> dict:
    """
    Run warmup + iters all-reduce iterations with optional background I/O.

    Paced mode: workers are triggered at the start of each all-reduce and
    read one batch worth of data (bytes_per_iter per worker).  After the
    all-reduce, the main thread waits for workers to finish and records
    dl_wait_ms — the residual stall time after compute (DataLoader stall).

    Tight mode: workers run in a continuous loop for the entire phase
    (backward-compatible stress test; not a realistic DataLoader model).
    """
    chunk_bytes_tight = args.read_kb * 1024
    chunk_bytes_paced = int(args.sample_kb * 1024)
    samples_per_worker = math.ceil(args.batch_size / io_workers) if io_workers > 0 else 0
    bytes_per_iter = samples_per_worker * int(args.sample_kb * 1024)

    readers = []
    if io_workers > 0:
        for widx in range(io_workers):
            log_affinity = (rank == 0 and widx == 0)
            if args.io_mode == "tight":
                t = IOReaderProcess(io_fpath, chunk_bytes_tight, mode="tight",
                                    log_affinity=log_affinity)
            else:
                t = IOReaderProcess(io_fpath, chunk_bytes_paced, mode="paced",
                                    bytes_per_iter=bytes_per_iter, log_affinity=log_affinity)
            t.start()
            readers.append(t)

    latencies = []
    dl_waits = []
    e_start = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)

    dist.barrier()

    for i in range(warmup + iters):
        # Trigger I/O workers at start of each iteration (overlaps with all-reduce)
        if args.io_mode == "paced":
            for t in readers:
                t.trigger_iter()

        tensor.fill_(float(rank))
        e_start.record()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        e_end.record()
        torch.cuda.synchronize()
        nccl_ms = e_start.elapsed_time(e_end)

        # Wait for prefetch to complete; measure residual stall (paced mode only)
        dl_wait_ms = 0.0
        if args.io_mode == "paced" and readers:
            t0 = time.perf_counter()
            for t in readers:
                t.wait_done(timeout=120.0)
            dl_wait_ms = (time.perf_counter() - t0) * 1000.0

        if i >= warmup:
            latencies.append(nccl_ms)
            dl_waits.append(dl_wait_ms)

    for r in readers:
        r.stop()
    for r in readers:
        r.join(timeout=5.0)

    total_bytes = sum(r.bytes_read for r in readers)

    result = {
        "label": label,
        "latencies": latencies,
        "dl_waits": dl_waits,
        "io_bytes": total_bytes,
    }
    if rank == 0:
        sorted_lats = sorted(latencies)
        result.update({
            "median": statistics.median(latencies),
            "p95":    sorted_lats[int(0.95 * len(latencies))],
            "p99":    sorted_lats[int(0.99 * len(latencies))],
            "max":    max(latencies),
            "min":    min(latencies),
            "n":      len(latencies),
        })
        if dl_waits:
            sorted_dl = sorted(dl_waits)
            result.update({
                "dl_median": statistics.median(dl_waits),
                "dl_p95":    sorted_dl[int(0.95 * len(dl_waits))],
                "dl_max":    max(dl_waits),
            })
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(rank: int, world: int, args, results: list, n_nodes: int = 1):
    if rank != 0:
        return

    label = f" [{args.label}]" if args.label else ""
    print()
    print(f"{'='*72}")
    print(f"  NCCL / Filesystem I/O Interference Benchmark{label}")
    print(f"{'='*72}")
    print(f"  Ranks      : {world}  (tensor {args.tensor_mb:.3f} MB = "
          f"{int(args.tensor_mb*1024**2/4):,} float32 elements)")
    print(f"  Iterations : {args.iters} (+ {args.warmup} warmup)")
    print(f"  FS path    : {args.fs_path}")
    print(f"  I/O mode   : {args.io_mode}")
    if args.io_mode == "paced" and args.io_workers > 0:
        samples_per_worker = math.ceil(args.batch_size / args.io_workers)
        bytes_per_worker_mb = samples_per_worker * args.sample_kb / 1024
        print(f"  I/O workers: {args.io_workers}/rank  "
              f"(batch={args.batch_size}, sample={args.sample_kb:.0f} KB  ->  "
              f"{samples_per_worker} reads x {args.sample_kb:.0f} KB = "
              f"{bytes_per_worker_mb:.1f} MB/worker/iter)")
    else:
        print(f"  I/O workers: {args.io_workers}/rank  "
              f"({args.read_kb} KB/read, {args.file_mb} MB file/rank, continuous loop)")

    base_median = results[0]["median"] if results else 1.0
    stall_ms = 10.0 * base_median
    print(f"  Stall threshold: {stall_ms:.2f} ms  (10x baseline median)")
    print()

    has_dl = args.io_mode == "paced"
    hdr = (f"  {'Phase':<14}  {'nccl med':>9}  {'p95':>9}  {'p99':>9}  "
           f"{'max':>10}  {'stalls':>10}  {'total':>8}")
    if has_dl:
        hdr += f"  {'dl_wait med':>12}  {'dl_wait max':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in results:
        stalls = sum(1 for x in r["latencies"] if x > stall_ms)
        total_s = r["n"] * r["median"] / 1000
        stall_str = f"{stalls}/{r['n']}"
        line = (f"  {r['label']:<14}  "
                f"{r['median']:>8.2f}ms  "
                f"{r['p95']:>8.2f}ms  "
                f"{r['p99']:>8.2f}ms  "
                f"{r['max']:>9.2f}ms  "
                f"{stall_str:>10}  "
                f"{total_s:>7.1f}s")
        if has_dl:
            if "dl_median" in r:
                line += f"  {r['dl_median']:>11.2f}ms  {r['dl_max']:>11.2f}ms"
            else:
                line += f"  {'n/a':>12}  {'n/a':>12}"
        print(line)

    print()

    if n_nodes == 1:
        print("  Note: single-node run — NCCL uses intra-node links (XGMI/NVLink/PCIe),")
        print("        not the fabric. Re-run with >= 2 nodes for fabric TC interference.")
        print()

    if len(results) == 2:
        base, io = results[0], results[1]
        factor_median = io["median"] / max(base["median"], 0.001)
        factor_max    = io["max"]    / max(base["max"],    0.001)
        io_stalls = sum(1 for x in io["latencies"] if x > stall_ms)

        print(f"  NCCL interference: {factor_median:.1f}x median,  {factor_max:.1f}x max")
        if has_dl and "dl_median" in io and io["dl_median"] > 1.0:
            print(f"  DL stall:     dl_wait median={io['dl_median']:.1f}ms, "
                  f"max={io['dl_max']:.1f}ms  (I/O did not complete before nccl)")

        if n_nodes == 1:
            verdict = ("HIGH PCIe/CPU contention (single-node; re-run multi-node for fabric story)."
                       if factor_median >= 3.0
                       else "LOW contention on single node. Re-run multi-node for fabric test.")
            suggest = ""
        elif factor_median >= 3.0 or io_stalls > 0:
            verdict = "HIGH — filesystem likely shares NCCL fabric traffic class."
            suggest = "Consider DYAD or node-local NVMe to eliminate contention."
        elif factor_median >= 1.5:
            verdict = "MODERATE — some fabric sharing or congestion observed."
            suggest = "Monitor at larger scale; may worsen with more nodes."
        else:
            verdict = "LOW — filesystem and NCCL use separate fabric resources."
            suggest = "I/O prefetch should not impede communication at this scale."

        print(f"  Verdict:      {verdict}")
        if suggest:
            print(f"  Suggestion:   {suggest}")

    print(f"{'='*72}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world, local_rank, device = init_dist()

    if rank == 0:
        print(f"[nccl_io_bench] {world} ranks initialised, device={device}", flush=True)

    affinity = sorted(os.sched_getaffinity(0))
    print(f"[nccl_io_bench] rank {rank}: {len(affinity)} CPU cores available "
          f"(os.cpu_count={os.cpu_count()}), io_workers={args.io_workers}, "
          f"affinity={affinity}", flush=True)

    n_elems = int(args.tensor_mb * 1024**2 / 4)
    tensor = torch.zeros(n_elems, dtype=torch.float32, device=device)

    file_size = args.file_mb * 1024**2
    # ensure temp file is large enough for paced reads
    if args.io_mode == "paced" and args.io_workers > 0:
        samples_per_worker = math.ceil(args.batch_size / args.io_workers)
        min_size = int(samples_per_worker * args.sample_kb * 1024) * 4
        file_size = max(file_size, min_size)

    if rank == 0:
        print(f"[nccl_io_bench] Creating temp files ({file_size//1024//1024} MB each) "
              f"on {args.fs_path} ...", flush=True)
    dist.barrier()

    fpath = create_test_file(args.fs_path, file_size, rank)
    dist.barrier()

    if rank == 0:
        print(f"[nccl_io_bench] Files ready. Starting benchmark ...", flush=True)

    results = []

    if not args.io_only:
        if rank == 0:
            print(f"[nccl_io_bench] Phase: baseline (no I/O) ...", flush=True)
        r = run_phase("baseline", args.iters, args.warmup, tensor,
                      0, fpath, args, rank)
        results.append(r)
        dist.barrier()

    if not args.baseline_only:
        if rank == 0:
            print(f"[nccl_io_bench] Phase: with-io "
                  f"({args.io_workers} workers/rank, mode={args.io_mode}) ...", flush=True)
        r = run_phase("with-io", args.iters, args.warmup, tensor,
                      args.io_workers, fpath, args, rank)
        results.append(r)
        dist.barrier()

    gpus_per_node = torch.cuda.device_count()
    n_nodes = max(1, world // gpus_per_node)

    print_report(rank, world, args, results, n_nodes=n_nodes)

    if not args.no_cleanup:
        remove_test_file(fpath)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
