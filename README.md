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
| `pecan_train_bench.py` | Companion benchmark: the real model and real training-loop phases (see below), instead of a bare all-reduce. |
| `submit_pecan_train_bench.sh` | Flux job-submission wrapper for `pecan_train_bench.py`. Tuolumne-specific (paths/venv); adapt for your own site. |

## pecan_train_bench.py: a realistic compute+comm companion

`nccl_io_bench.py` isolates fabric contention as cleanly as possible: a bare
all-reduce, no model, no real DataLoader. That's the right tool for proving
the *mechanism* exists. But it doesn't reproduce a real training iteration's
actual cost — in particular, a real `bwd+nccl`-style measurement is
dominated by backward-pass compute and the optimizer step, not by the
all-reduce itself (a 76 KB gradient all-reduces in under 1ms uncontended;
real backward+optimizer.step() commonly costs 100ms+). `pecan_train_bench.py`
fills that gap: single-file, self-contained reproduction of a real GNN
training iteration, for when you want a compute+comm workload that's
faithful to actual training rather than a synthetic stand-in.

It contains:

- **The actual model** — an `EGNN`/`E_GCL` equivariant graph network, copied
  verbatim (not reimplemented) from a real production training codebase, run
  through the real forward/backward/optimizer-step/DDP-all-reduce sequence
  every iteration. Reports its own parameter count and gradient size at
  startup so you can sanity-check it against whatever model you're
  comparing against.
- **Synthetic protein-ligand-complex-shaped graphs** — generated once into an
  in-memory pool at startup (no external dataset, no HDF5 files), matching
  real measured shape statistics (atoms/sample, edges/atom at the model's
  distance cutoff), so there's zero setup cost and no data-license concerns.
- **Optional real storage I/O** (`--fs-path`) — when set, each DataLoader
  worker additionally reads real bytes from a test file on that path as
  part of fetching every sample, so `dl`/`dl_max` in the output reflect
  genuine storage-tier latency. Point it at Lustre, VAST, node-local NVMe,
  etc., the same way `nccl_io_bench.py`'s `--fs-path` is used, but with a
  real model+training loop generating the concurrent compute+comm load
  instead of a bare tensor fill.

Output uses the same per-iteration line format a real training run would
log (`startup=... dl=... dl_max=... h2d=... fwd=... bwd+nccl=... iter=...`),
so it's a drop-in stand-in anywhere you'd otherwise need real training logs.

### Quick start

Pure compute+comm, no I/O at all:

```shell
flux run -N 4 -n 16 -o mpibind=off --exclusive \
    python pecan_train_bench.py --batch-size 64 --num-workers 8 --iters 100
```

With real storage I/O against a filesystem under test:

```shell
flux run -N 4 -n 16 -o mpibind=off --exclusive \
    python pecan_train_bench.py --batch-size 64 --num-workers 8 --iters 100 \
    --fs-path /p/lustre5/yourdir --sample-kb 286
```

Or via the Flux submission wrapper:

```shell
bash submit_pecan_train_bench.sh [N_NODES] [ITERS]
```

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--batch-size`, `--num-workers` | 64, 8 | Match your real DataLoader's config. |
| `--iters` / `--epochs` | 100 / 1 | Iterations per epoch (per rank). |
| `--fs-path` | *(none)* | Storage tier to read real per-sample bytes from (omit for pure-synthetic, zero-I/O mode). |
| `--sample-kb` | 286 | Real bytes read per sample when `--fs-path` is set (default matches a measured real HDF5+graph-cache average). |
| `--n-layers`, `--distance-cutoff`, `--out-dim`, `--in-channels`, `--in-edge-nf` | 6, 5.0, 6, 19, 1 | EGNN architecture params — match these to your own model config if it differs. |
| `--min-atoms`, `--max-atoms`, `--avg-degree` | 400, 1000, 20.0 | Synthetic graph shape (atoms/sample, edges/atom at `--distance-cutoff`) — calibrate to your own workload's measured stats. |
| `--pool-size` | 256 | Number of distinct synthetic graph templates. |

Run `python pecan_train_bench.py --help` for the full list.

### Known fidelity gaps (validated against real training logs)

Cross-checked against real 4-node/16-GPU PECAN/EGNN training logs (same
batch size, worker count, and node/GPU topology). Model parameter count
matched exactly (19,473 params / 76.1 KB gradient) both times. Per-field
comparison, real vs. this benchmark:

- **Compute+comm only, no `--fs-path`**: overall iteration time landed
  within ~10% of real training's steady state (`fwd` nearly identical);
  `bwd+nccl` ran moderately lower (~96ms vs. ~124ms), plausibly because the
  synthetic graphs are calibrated so nearly all edges already pass the
  model's distance-cutoff filter, while real graphs likely have more
  edges filtered out, changing the effective backward FLOPs.
- **With `--fs-path` against VAST**: reasonably close (`fwd` matched almost
  exactly; overall `iter` time ran ~2.4x lower than real).
- **With `--fs-path` against Lustre**: real training showed severe stalls
  (`dl_max` mean 706ms, max 15.8s, 8% of iterations >1s) that this
  benchmark's I/O did not reproduce (0 stalls observed). Real DataLoader
  workers do metadata-heavy HDF5 access — opening and attribute-reading
  across two separate files per sample — while this benchmark currently
  does one bulk read from a flat test file per sample, which is much
  friendlier to Lustre's metadata servers. If you need Lustre-realistic
  I/O severity specifically, this is the place to improve fidelity next
  (e.g. splitting each sample's read into several smaller, separately-
  opened reads to mimic real HDF5 access instead of one bulk read).

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
