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

## Files

| File | Purpose |
|---|---|
| `nccl_io_bench.py` | The benchmark itself. Portable — only depends on PyTorch (`torch.distributed`) and, optionally, `mpi4py` for rank/world-size discovery. Works under `torchrun`, `mpirun`, or a scheduler-native launcher (e.g. Flux's `flux run`). |
| `dyad_stage.py` | Optional extension that adds a `dyad`-staged phase: each rank self-serves its test file through a real [DYAD](https://github.com/flux-framework/dyad) produce/consume round trip onto node-local storage before the same measurement runs against the staged copy. Only relevant if you're evaluating DYAD specifically; the core benchmark has no DYAD dependency. |
| `run_nccl_io_bench.sh` | Example driver that sweeps several storage tiers (tmpfs, node-local NVMe, a parallel filesystem, a network-attached flash tier, and DYAD-staged) back-to-back in one job. Written for LLNL Tuolumne (Flux scheduler, Slingshot-11, Cray DataWarp/Rabbit NVMe) — **treat it as a worked example to adapt, not a portable script**: the storage paths, `flux run` flags, and `dyad start` staging step are all site-specific. |
| `submit_nccl_io_bench.sh` | Thin Flux job-submission wrapper around `run_nccl_io_bench.sh` (sets up DYAD's `LD_LIBRARY_PATH`/`PATH` and submits the job). Also Tuolumne-specific. |
| `pecan_train_bench.py` | Companion benchmark: the real model and real training-loop phases (see below), instead of a bare all-reduce. |
| `submit_pecan_train_bench.sh` | Flux job-submission wrapper for `pecan_train_bench.py`. Tuolumne-specific (paths/venv); adapt for your own site. |
| `generate_synthetic_hdf5.py` | Generates a dependency-free HDF5 dataset with the real dataset's exact on-disk structure, for use with `pecan_train_bench.py --io-style real` (see below). |

## nccl_io_bench.py: measuring I/O-vs-NCCL fabric interference

### What it measures

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

### Quick start

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

### Key options

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

### Interpreting the report

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
- **Optional real storage I/O**, in one of two styles selected by
  `--io-style`:
  - `flat` (default) — each DataLoader worker reads `--sample-kb` KB from a
    random offset in a synthetic per-rank test file on `--fs-path`, so
    `dl`/`dl_max` reflect genuine storage-tier latency for a single bulk
    transfer per sample. Cheap and dependency-free, but — see "Known
    fidelity gaps" below — only reproduces mild I/O-vs-NCCL interference; a
    single bulk read per sample is much friendlier to a parallel
    filesystem's metadata server than real HDF5 access is.
  - `real` — bypasses the synthetic dataset entirely and uses the actual
    `pecan.dataset.Dataset_PDB` class against real (or
    `generate_synthetic_hdf5.py`-generated, see below) HDF5 files via
    `--real-csv`/`--real-graph-cache-dir`: real per-sample group/attribute
    lookups, real chunked dataset reads, no guessing about access shape.
    This is the one that actually reproduces severe Lustre interference.

Output uses the same per-iteration line format a real training run would
log (`startup=... dl=... dl_max=... h2d=... fwd=... bwd+nccl=... iter=...`),
so it's a drop-in stand-in anywhere you'd otherwise need real training logs.

### Quick start

Pure compute+comm, no I/O at all:

```shell
flux run -N 4 -n 16 -o mpibind=off --exclusive \
    python pecan_train_bench.py --batch-size 64 --num-workers 8 --iters 100
```

With real storage I/O against a filesystem under test (bulk-read approximation):

```shell
flux run -N 4 -n 16 -o mpibind=off --exclusive \
    python pecan_train_bench.py --batch-size 64 --num-workers 8 --iters 100 \
    --fs-path /p/lustre5/yourdir --sample-kb 286
```

With genuine HDF5 access against a real (or generated) dataset:

```shell
flux run -N 4 -n 16 -o mpibind=off --exclusive \
    python pecan_train_bench.py --batch-size 64 --num-workers 8 --iters 100 \
    --io-style real \
    --real-csv /p/lustre5/yourdir/synth_all.csv \
    --real-graph-cache-dir /p/lustre5/yourdir/graph_cache
```

Or via the Flux submission wrapper:

```shell
bash submit_pecan_train_bench.sh [N_NODES] [ITERS]
```

Each run prints a baseline-vs-with-I/O comparison table (median/p95/max for
`bwd+nccl` and `dl_max`, plus interference factors) at the end — see
"Interpreting the report" above (same report format as `nccl_io_bench.py`).

### generate_synthetic_hdf5.py: a dependency-free real-shaped dataset

`--io-style real` needs real HDF5 files on disk. If you don't have access to
the actual (private) dataset, `generate_synthetic_hdf5.py` builds one with
the exact same on-disk structure — nested `h5[pdbid]["dcomplex"][poseid]`
groups holding `coord`/`feat` datasets and 5 scalar attributes, companion
`*_graph.h5` files holding precomputed `edge_index`/`edge_attr`, and a
matching CSV — so `pecan.dataset.Dataset_PDB` (and therefore
`pecan_train_bench.py`) can't tell the difference structurally, even though
the content is random.

```shell
python generate_synthetic_hdf5.py --out-dir /p/lustre5/yourdir/synthdata \
    --n-files 105 --groups-per-file 2000 --poses-per-group 5 --workers 16

python pecan_train_bench.py --io-style real \
    --real-csv /p/lustre5/yourdir/synthdata/synth_all.csv \
    --real-graph-cache-dir /p/lustre5/yourdir/synthdata/graph_cache
```

`--groups-per-file` matters more than `--n-files` for reproducing
interference severity: HDF5 switches a group's internal link storage from a
flat "compact" layout to a B-tree-indexed layout once entry counts cross a
small threshold (~8-16 by default), and the real dataset's severity appears
tied to how deep/large that indexed structure gets (real files hold ~10,000
groups each) rather than just file count. See "Known fidelity gaps" below
for measured results at different scales.

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--batch-size`, `--num-workers` | 64, 8 | Match your real DataLoader's config. |
| `--iters` / `--epochs` | 100 / 1 | Iterations per epoch (per rank). |
| `--io-style` | `flat` | `flat`: one bulk read per sample from a synthetic test file. `real`: genuine `Dataset_PDB` access against real HDF5 files. |
| `--fs-path` | *(none)* | `[io-style=flat]` Storage tier to read real per-sample bytes from (omit for pure-synthetic, zero-I/O mode). |
| `--sample-kb` | 286 | `[io-style=flat]` Real bytes read per sample when `--fs-path` is set (default matches a measured real HDF5+graph-cache average). |
| `--real-csv`, `--real-graph-cache-dir` | *(none)* | `[io-style=real]` Paths to the training CSV and precomputed graph-cache dir (real or `generate_synthetic_hdf5.py` output). |
| `--baseline-only`, `--io-only` | off | Run just one phase instead of the baseline-vs-with-I/O A/B comparison. |
| `--n-layers`, `--distance-cutoff`, `--out-dim`, `--in-channels`, `--in-edge-nf` | 6, 5.0, 6, 19, 1 | EGNN architecture params — match these to your own model config if it differs. |
| `--min-atoms`, `--max-atoms`, `--avg-degree` | 400, 1000, 20.0 | Synthetic graph shape (atoms/sample, edges/atom at `--distance-cutoff`) — calibrate to your own workload's measured stats. |
| `--pool-size` | 256 | Number of distinct synthetic graph templates (`--io-style flat` only). |

Run `python pecan_train_bench.py --help` for the full list.

### Known fidelity gaps (validated against real training logs)

Cross-checked against real 4-node/16-GPU PECAN/EGNN training logs (same
batch size, worker count, and node/GPU topology). Model parameter count
matched exactly (19,473 params / 76.1 KB gradient) both times. Per-field
comparison, real vs. this benchmark:

- **Compute+comm only, no I/O**: overall iteration time landed within ~10%
  of real training's steady state (`fwd` nearly identical); `bwd+nccl` ran
  moderately lower (~96ms vs. ~124ms), plausibly because the synthetic
  graphs are calibrated so nearly all edges already pass the model's
  distance-cutoff filter, while real graphs likely have more edges
  filtered out, changing the effective backward FLOPs.
- **`--io-style flat` against VAST**: reasonably close (`fwd` matched
  almost exactly; overall `iter` time ran ~2.4x lower than real).
- **`--io-style flat` against Lustre**: does *not* reproduce real
  training's severe stalls. A same-job baseline-vs-with-io A/B comparison
  showed only a ~1.0x median effect and no meaningful tail inflation —
  real training's Lustre run showed `dl_max` mean 706ms, max 15.8s, 8% of
  iterations >1s. One bulk read per sample (even split into several
  smaller separately-opened reads against a large randomized file pool —
  tried and rejected, see git history) is fundamentally too metadata-server
  -friendly to reproduce this; only genuine HDF5 access does.
- **`--io-style real` against Lustre, using the actual private dataset**:
  reproduces the severity dramatically — `bwd+nccl` p95 23x / max 47x over
  baseline, `dl_max` p95 51x / max 46x, with individual iterations showing
  genuine 20+ second stalls. This is the closest match to real training
  found so far.
- **`--io-style real` against Lustre, using `generate_synthetic_hdf5.py`
  output**: reproduces the effect only partially so far, and not in the
  same *shape* as the real dataset. At `--groups-per-file 500` (105 files,
  262K samples, ~33GB): `bwd+nccl` p95 3.6x over baseline (vs. ~1.0x for
  `--io-style flat`) but median and max stayed close to baseline — a mild,
  tail-only effect, qualitatively like the real dataset's signature (real
  training's `dl_max` *median* actually drops slightly under I/O, with only
  p95/max exploding — a bursty effect, not a uniform slowdown), just much
  smaller in magnitude. At `--groups-per-file 2000` (105 files, 1.05M
  samples, ~148GB) the picture changed shape rather than simply scaling up:
  `dl_max` *median* jumped 12.7x (not just its tail), while `bwd+nccl`
  showed no interference at all (its tail actually tightened vs. baseline).
  This looks like a cache-size effect rather than more severity — 148GB
  likely exceeds Lustre's client-side cache where 33GB didn't, so most
  reads became genuinely cold rather than the access pattern becoming more
  contended, and once the DataLoader is blocking for that long every
  iteration, it stops overlapping with the *next* iteration's compute, which
  may be why `bwd+nccl` stopped showing any effect. Whether the real
  dataset's bursty, tail-only signature (rather than this uniform-slowdown
  one) shows up at scales closer to the real ~10,000 groups/file, or
  requires genuine multi-rank contention on the *same* files (each rank
  here still gets its own private synthetic files), is still open —
  a `--groups-per-file 10000` run (matching the real dataset almost
  exactly, ~5.25M samples, ~700GB) is in progress to test this directly.