# nccl_io_bench

A microbenchmark for measuring NCCL/RCCL collective-communication latency under
concurrent, DataLoader-style parallel filesystem I/O.

On systems where storage and the GPU interconnect share the same physical
network fabric (e.g. HPE Slingshot, where a parallel filesystem's RDMA path
and NCCL/RCCL's RDMA path can land in the same fabric traffic class),
concurrent I/O can silently inflate all-reduce latency — an effect that's
easy to misattribute to raw storage bandwidth rather than fabric-level
contention. This tool isolates the effect from a full training stack (no
model, no real DataLoader, no framework overhead) so the two can't be
conflated: it runs an NCCL/RCCL all-reduce in a tight loop while background
reader processes concurrently read data from a filesystem under test, and
reports whether the all-reduce itself slows down.

## What it measures

Each iteration reports two numbers:

- **`nccl_ms`** — CUDA/HIP-timed all-reduce latency. An increase vs. the
  no-I/O baseline indicates the collective is contending with I/O for shared
  fabric resources (e.g. Slingshot traffic-class injection credits).
- **`dl_wait`** — time the main thread blocks waiting for the background
  reader(s) to finish *after* the all-reduce completes. A nonzero value means
  I/O is too slow to fully hide behind compute (a DataLoader-style stall,
  independent of whether the fabric itself is contended).

The two are orthogonal: a filesystem can show high `nccl_ms` inflation with
near-zero `dl_wait` (fabric contention, I/O itself is fast) or the reverse
(slow I/O, but on a network path that doesn't interfere with the collective).

Works with either GPU vendor's collective library — `torch.distributed`'s
`nccl` backend transparently resolves to RCCL on ROCm builds, and this tool
makes no NVIDIA- or AMD-specific assumptions anywhere.

## Files

| File | Purpose |
|---|---|
| `nccl_io_bench.py` | The benchmark itself. Portable — only depends on PyTorch (`torch.distributed`) and, optionally, `mpi4py` for rank/world-size discovery. Works under `torchrun`, `mpirun`, or a scheduler-native launcher (e.g. Flux's `flux run`). |
| `dyad_stage.py` | Optional extension that adds a `dyad`-staged phase: each rank self-serves its test file through a real [DYAD](https://github.com/flux-framework/dyad) produce/consume round trip onto node-local storage before the same measurement runs against the staged copy. Only relevant if you're evaluating DYAD specifically; the core benchmark has no DYAD dependency. |
| `run_nccl_io_bench.sh` | Example driver that sweeps several storage tiers (tmpfs, node-local NVMe, a parallel filesystem, a network-attached flash tier, and DYAD-staged) back-to-back in one job. Written for LLNL Tuolumne (Flux scheduler, Slingshot-11, Cray DataWarp/Rabbit NVMe) — **treat it as a worked example to adapt, not a portable script**: the storage paths, `flux run` flags, and `dyad start` staging step are all site-specific. |
| `submit_nccl_io_bench.sh` | Thin Flux job-submission wrapper around `run_nccl_io_bench.sh` (sets up DYAD's `LD_LIBRARY_PATH`/`PATH` and submits the job). Also Tuolumne-specific. |

## Quick start

Single filesystem, via `torchrun`:

```shell
torchrun --nproc_per_node=4 nccl_io_bench.py --fs-path /path/to/filesystem
```

Via an MPI-aware launcher (auto-detects rank/world size/hostnames through
`mpi4py` if available):

```shell
flux run -N 4 -n 16 python nccl_io_bench.py --fs-path /p/lustre5/yourdir
mpirun -N 4 -n 16 python nccl_io_bench.py --fs-path /p/lustre5/yourdir
```

Compare two filesystems by running the same benchmark against each and
diffing the reports:

```shell
flux run -N 4 -n 16 python nccl_io_bench.py --fs-path /p/lustre5/x --label lustre
flux run -N 4 -n 16 python nccl_io_bench.py --fs-path /p/vast1/x   --label vast
```

Or sweep several tiers (including a DYAD-staged phase) in one job on
Tuolumne — adapt the storage paths at the top of the script for your own
system first:

```shell
bash submit_nccl_io_bench.sh [N_NODES]
```

## Key options

| Flag | Default | Meaning |
|---|---|---|
| `--fs-path` | *(required)* | Filesystem path to probe, e.g. `/p/lustre5/...`, `/p/vast1/...`, `/dev/shm/...`, or a node-local NVMe mount. |
| `--io-mode` | `paced` | `paced`: each background reader reads exactly one batch's worth of data per all-reduce iteration, synchronized to the collective — emulates a real multi-process PyTorch `DataLoader` (`num_workers > 0`, each worker its own OS process, no shared GIL with the training loop). `tight`: readers spin in a continuous loop regardless of iteration rate — a raw bandwidth stress test, not a realistic access pattern, and prone to CPU/cache contention unrelated to the fabric. |
| `--io-workers` | 8 | Background reader processes per rank. |
| `--batch-size`, `--sample-kb` | 64, 286 | `[paced]` Determines bytes read per worker per iteration: `ceil(batch_size / io_workers)` samples of `sample_kb` KB each. Defaults match a PECAN/EGNN-style GNN training workload; adjust to match your own DataLoader's real per-iteration read volume. |
| `--tensor-mb` | 0.076 | All-reduce tensor size in MB. Default (76 KB) matches a small GNN gradient; scale up for larger models. |
| `--iters` / `--warmup` | 100 / 20 | Measured / warmup iterations. |
| `--label` | *(none)* | Short tag for the report, e.g. `lustre`, `vast`. |

Run `python nccl_io_bench.py --help` for the full list.

## Interpreting the report

The report prints per-phase latency percentiles, a stall count (iterations
exceeding 10x the no-I/O baseline median), and, when both a baseline and
with-I/O phase are run, an **interference factor** (`with-I/O median /
baseline median`) plus a rule-of-thumb verdict:

- **`< 1.5x`** — filesystem and the collective are using separate fabric
  resources; I/O shouldn't meaningfully impede communication.
- **`1.5x`–`3x`** — moderate, possibly worth watching at larger scale.
- **`>= 3x`, or any stalls** — the filesystem's I/O path likely shares the
  same fabric traffic class as the collective; consider node-local storage
  or a staging layer (e.g. DYAD) to remove it from the shared path.

Single-node runs are flagged separately: NCCL/RCCL uses intra-node links
(NVLink/XGMI/PCIe) there, not the network fabric, so single-node contention
reflects PCIe/CPU sharing, not the fabric-level effect this tool is built to
isolate — re-run at 2+ nodes for the fabric-contention story.
