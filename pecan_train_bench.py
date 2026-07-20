#!/usr/bin/env python3
"""
pecan_train_bench.py — standalone, single-file reproduction of PECAN's real
EGNN training iteration: the actual model (model/egnn.py's EGNN, copied
verbatim below) and the actual per-iteration phases (DataLoader fetch, H2D
copy, forward, backward pass fused with DDP's gradient all-reduce, optimizer
step), exactly as pecan/trainer.py's train_one_epoch() runs them.

Why this exists: nccl_io_bench.py's all-reduce-in-a-tight-loop is a clean,
isolated way to measure fabric interference, but it does not reproduce real
training's compute -- in particular, bwd+nccl's steady-state cost is
dominated by backward-pass compute and optimizer.step(), not by the
all-reduce itself (see the paper's Section 1 breakdown). This script fills
that gap: a compute+comm workload that is model-for-model identical to real
training, but with zero external dependencies -- no HDF5 files, no CSVs, no
private ~157GB SAIR dataset. Samples are synthetic protein-ligand-complex-
shaped graphs, generated once at startup (see SyntheticPDBGraphDataset) and
served cheaply from an in-memory pool thereafter, so DataLoader-side cost
reflects real per-sample IPC/collation overhead rather than disk I/O or
graph construction -- matching how real training reads pre-computed graph
edges from graph_cache_dir rather than building them on the fly.

Per-iteration timing uses the exact same fields/format as pecan/trainer.py,
so output lines are drop-in compatible with the paper's existing log
parsers:
    [e/E]-[i/N] loss: L, lr: LR, sec/iter: S  [startup=Ams dl=Bms dl_max=Cms
    h2d=Dms fwd=Ems bwd+nccl=Fms iter=Gms]

Quick start (mpi4py / Flux):
    flux run -N 4 -n 16 -o mpibind=off --exclusive \\
        python pecan_train_bench.py --batch-size 64 --num-workers 8 --iters 100

Quick start (torchrun, single node):
    torchrun --nproc_per_node=4 pecan_train_bench.py --iters 100

Reproduce a real full-epoch run's shape (5,121 iterations at 4N/16GPU/b64):
    flux run -N 4 -n 16 -o mpibind=off --exclusive \\
        python pecan_train_bench.py --batch-size 64 --num-workers 8 --iters 5121
"""

import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import global_add_pool


# =============================================================================
# Model: EGNN -- verbatim copy of model/egnn.py (PECAN's real nn_type=3 model,
# the one used for every measurement in the paper). Not reimplemented or
# simplified: this is the actual architecture, actual forward math, actual
# parameter count.
# =============================================================================

class E_GCL(nn.Module):
    """E(n) Equivariant Convolutional Layer."""

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(),
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class EGNN_MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EGNN_MLP, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / 1.5)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 1.5), int(in_channels / 2)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 2), out_channels),
        )

    def forward(self, data, return_hidden_feature=False):
        if return_hidden_feature:
            return self.output(data), self.output[:-2](data), self.output[:-4](data)
        else:
            return self.output(data)


class EGNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, in_edge_nf=1, distance_cutoff=1.5,
                 act_fn=nn.SiLU(), n_layers=4, residual=True, attention=True, normalize=False,
                 tanh=False, get_feature_only=False):
        super(EGNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = 20
        self.hidden_nf = 20
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(self.in_channels, self.hidden_channels)
        self.embedding_out = nn.Linear(self.hidden_nf, self.hidden_channels)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.pool = global_add_pool
        self.relu = nn.ReLU()
        self.get_feature_only = get_feature_only
        self.mlp = EGNN_MLP(self.hidden_channels, self.out_channels)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_channels, self.hidden_nf, self.hidden_nf,
                                                edges_in_d=0, act_fn=act_fn, residual=residual,
                                                attention=attention, normalize=normalize, tanh=tanh))
            self.add_module("tanh_%d" % i, nn.Tanh())
            self.add_module("tanh_c_%d" % i, nn.Tanh())
            self.add_module("bn_%d" % i, nn.BatchNorm1d(self.hidden_nf))
            self.add_module("bn_c_%d" % i, nn.BatchNorm1d(3))

        self.pair_distance = nn.PairwiseDistance(p=2)
        self.distance_cutoff = distance_cutoff
        self.n_layers = n_layers

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        out = self.embedding_in(x)
        coords = data.pos
        distances = self.pair_distance(coords[edge_index[0]], coords[edge_index[1]]) + 1e-4
        edge_index = edge_index[:, distances < self.distance_cutoff]
        edge_attr = data.edge_attr[distances < self.distance_cutoff].view(-1, 1)
        for i in range(0, self.n_layers):
            out, coords, _ = self._modules["gcl_%d" % i](out, edge_index, coords, edge_attr=edge_attr)
            out = self._modules["tanh_%d" % i](out)
            out = self._modules["bn_%d" % i](out)
            coords = self._modules["tanh_c_%d" % i](coords)
            coords = self._modules["bn_c_%d" % i](coords)

        out = self.embedding_out(out)
        out = self.pool(out, data.batch)
        if self.get_feature_only:
            return out
        else:
            return self.mlp(self.relu(out), True)


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


# =============================================================================
# Optional real-storage backing: mirrors pecan/dataset.py's lazy
# per-process file handle cache (self._h5_handles), so that when --fs-path
# is set, each DataLoader worker process opens its own independent file
# descriptor on first access rather than sharing one inherited across fork.
# =============================================================================

def create_test_file(path, size_bytes, rank):
    """Create (or reuse) a per-rank test file to read real bytes from."""
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, f"_pecan_train_bench_rank{rank:04d}.bin")
    if os.path.exists(fpath) and os.path.getsize(fpath) == size_bytes:
        return fpath
    chunk = 1 << 20
    with open(fpath, "wb") as f:
        written = 0
        while written < size_bytes:
            n = min(chunk, size_bytes - written)
            f.write(os.urandom(n))
            written += n
    return fpath


# =============================================================================
# Synthetic dataset: protein-ligand-complex-shaped graphs
# =============================================================================

class SyntheticPDBGraphDataset(Dataset):
    """
    Stand-in for pecan/dataset.py's Dataset_PDB (graph-cache path). Real atom
    counts (measured from the actual SAIR HDF5 files) run ~400-1000 per
    sample with ~20 edges/atom at a 5A distance cutoff; this generates
    random graphs with the same shape statistics using the exact same edge
    construction as pecan/precompute_graphs.py's compute_edges()
    (pairwise distance + threshold), so downstream cost -- including EGNN's
    own redundant distance_cutoff filter in forward() -- matches the real
    graph-cache code path.

    A fixed-size pool of `pool_size` templates is built once in __init__
    (before DataLoader workers fork), and __getitem__ just indexes into it.
    This mirrors real training's cost profile: graph edges are read from a
    pre-computed cache, not built on the fly, so per-sample DataLoader cost
    here reflects realistic IPC/collation overhead rather than O(n^2)
    distance-matrix construction on every access.

    If `fs_path` is set, __getitem__ additionally performs a real disk read
    of `sample_kb` KB (default 286 KB, matching perf_analysis.md's measured
    real per-sample volume: ~74 KB raw HDF5 + ~212 KB graph-cache) from a
    per-rank test file at that path -- so `dl`/`dl_max` in this script's
    output reflects genuine storage-tier latency, exactly like the real
    DataLoader workers reading real HDF5 files. Point --fs-path at Lustre,
    VAST, node-local NVMe, etc. to compare, the same way nccl_io_bench.py's
    --fs-path is used, but with the real model+training loop generating the
    concurrent compute+comm load instead of a bare tensor fill.
    """

    def __init__(self, length, pool_size=256, min_atoms=400, max_atoms=1000,
                 distance_cutoff=5.0, avg_degree=20.0, in_channels=19, out_dim=6, seed=0,
                 read_fpath=None, sample_kb=286.0):
        self.length = length
        self.out_dim = out_dim
        self.read_fpath = read_fpath  # pre-created file to read real bytes from, or None
        self.sample_bytes = int(sample_kb * 1024)
        self._fh = None      # lazily opened per-worker-process (see __getitem__)
        self._file_size = None
        self._rng_io = None
        rng = np.random.default_rng(seed)
        self.pool = []
        for _ in range(pool_size):
            n = int(rng.integers(min_atoms, max_atoms + 1))
            # Box size calibrated so a uniform random packing of n atoms
            # gives ~avg_degree neighbors within distance_cutoff on average
            # (sphere-volume heuristic): L^3 = n * (4/3 pi r^3) / avg_degree.
            sphere_vol = (4.0 / 3.0) * math.pi * (distance_cutoff ** 3)
            box = (n * sphere_vol / avg_degree) ** (1.0 / 3.0)
            coord = rng.uniform(0, box, size=(n, 3)).astype(np.float32)
            feat = rng.standard_normal(size=(n, in_channels)).astype(np.float32)

            diff = coord[:, None, :] - coord[None, :, :]
            d = np.sqrt((diff ** 2).sum(-1))
            rows, cols = np.where((d > 0) & (d < distance_cutoff))
            edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
            edge_attr = d[rows, cols].astype(np.float32)

            labels = rng.uniform(4.0, 10.0, size=1).astype(np.float32)          # affinity (pKd-like)
            bond_counts = rng.integers(0, 10, size=5).astype(np.float32)        # hbond/hpbond/habond/sbond/pbond

            self.pool.append((coord, feat, edge_index, edge_attr, labels, bond_counts))

    def __len__(self):
        return self.length

    def _read_real_bytes(self):
        """Read self.sample_bytes from this worker process's own handle on
        self.read_fpath -- opened lazily so each DataLoader worker (forked
        independently) gets its own fd, never one inherited/shared across
        workers from the parent process (mirrors pecan/dataset.py's
        self._h5_handles lazy-open-per-process pattern)."""
        if self._fh is None:
            import random as _random
            self._fh = open(self.read_fpath, "rb")
            self._file_size = os.fstat(self._fh.fileno()).st_size
            self._rng_io = _random.Random(os.getpid())
        max_off = max(0, self._file_size - self.sample_bytes)
        offset = self._rng_io.randint(0, max_off)
        self._fh.seek(offset)
        self._fh.read(self.sample_bytes)

    def __getitem__(self, ind):
        if self.read_fpath is not None:
            self._read_real_bytes()

        coord, feat, edge_index, edge_attr, labels, bond_counts = self.pool[ind % len(self.pool)]

        data = Data()
        data.pos = torch.from_numpy(coord)
        data.x = torch.from_numpy(feat)
        data.edge_index = torch.from_numpy(edge_index)
        data.edge_attr = torch.from_numpy(edge_attr).view(-1, 1)

        return {
            "data": data,
            "affinity": torch.from_numpy(labels),
            "hbond": torch.tensor([bond_counts[0]]),
            "hpbond": torch.tensor([bond_counts[1]]),
            "habond": torch.tensor([bond_counts[2]]),
            "sbond": torch.tensor([bond_counts[3]]),
            "pbond": torch.tensor([bond_counts[4]]),
        }


# =============================================================================
# Distributed init (mpi4py/Flux, falling back to torchrun env vars)
# =============================================================================

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
        if world > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl", world_size=world, rank=rank,
                                     timeout=timedelta(seconds=600))
    except (ImportError, Exception):
        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if world > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return rank, world, local_rank, device


# =============================================================================
# Training loop -- exact structure/timers/print format of
# pecan/trainer.py's train_one_epoch()
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone reproduction of PECAN's real EGNN training iteration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--iters", type=int, default=100, help="Iterations per epoch (per rank)")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--loss-weight", type=float, default=1.0e-2)
    p.add_argument("--out-dim", type=int, default=6)
    p.add_argument("--in-channels", type=int, default=19)
    p.add_argument("--in-edge-nf", type=int, default=1)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--distance-cutoff", type=float, default=5.0)
    p.add_argument("--attention", action="store_true", default=True)
    p.add_argument("--residual", action="store_true", default=True)
    p.add_argument("--min-atoms", type=int, default=400)
    p.add_argument("--max-atoms", type=int, default=1000)
    p.add_argument("--avg-degree", type=float, default=20.0,
                   help="Target avg edges/atom at --distance-cutoff (measured real value: ~20)")
    p.add_argument("--pool-size", type=int, default=256,
                   help="Number of distinct synthetic graph templates")
    p.add_argument("--use-amp", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fs-path", default=None,
                   help="Storage tier to read real per-sample bytes from each "
                        "iteration (e.g. a Lustre or VAST directory). Omit for "
                        "pure-synthetic, zero-I/O mode (default).")
    p.add_argument("--sample-kb", type=float, default=286.0,
                   help="Real bytes read per sample when --fs-path is set "
                        "(default: 286 = measured real HDF5+graph-cache average)")
    p.add_argument("--file-mb", type=int, default=512,
                   help="Per-rank test file size on --fs-path")
    p.add_argument("--no-cleanup", action="store_true",
                   help="Keep the --fs-path test file after the run")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world, local_rank, device = init_dist()

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if rank == 0:
        print(f"[pecan_train_bench] {world} ranks initialised, device={device}", flush=True)

    model = EGNN(
        in_channels=args.in_channels, out_channels=args.out_dim,
        in_edge_nf=args.in_edge_nf, distance_cutoff=args.distance_cutoff,
        n_layers=args.n_layers, residual=args.residual, attention=args.attention,
        normalize=False, tanh=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[pecan_train_bench] EGNN parameters: {n_params:,} "
              f"({4 * n_params / 1024:.1f} KB fp32 gradient)", flush=True)

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    loss_mse = nn.MSELoss().float()

    read_fpath = None
    if args.fs_path is not None:
        file_bytes = max(args.file_mb << 20, int(args.sample_kb * 1024) * 8)
        if rank == 0:
            print(f"[pecan_train_bench] Creating per-rank test files "
                  f"({file_bytes // 1024 // 1024} MB each) on {args.fs_path} ...", flush=True)
        if dist.is_initialized():
            dist.barrier()
        read_fpath = create_test_file(args.fs_path, file_bytes, rank)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            print(f"[pecan_train_bench] Test files ready; each DataLoader worker "
                  f"reads {args.sample_kb:.0f} KB/sample from {args.fs_path}", flush=True)

    dataset_len = args.iters * args.batch_size * world
    dataset = SyntheticPDBGraphDataset(
        length=dataset_len, pool_size=args.pool_size,
        min_atoms=args.min_atoms, max_atoms=args.max_atoms,
        distance_cutoff=args.distance_cutoff, avg_degree=args.avg_degree,
        in_channels=args.in_channels, out_dim=args.out_dim, seed=args.seed,
        read_fpath=read_fpath, sample_kb=args.sample_kb)

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    dataloader = DataListLoader(dataset=dataset, shuffle=(sampler is None), batch_size=args.batch_size,
                                 num_workers=args.num_workers, sampler=sampler, pin_memory=True, drop_last=True)

    if rank == 0:
        print(f"[pecan_train_bench] len(dataloader): {len(dataloader)}", flush=True)

    import time as time_mod

    for epoch in range(args.epochs):
        model.train()
        iter_time = time_mod.perf_counter()
        t_prev_iter_end = None
        losses = []
        start_time = time_mod.perf_counter()

        for ind, batch in enumerate(dataloader):
            t_iter_start = time_mod.perf_counter()
            startup_ms = (t_iter_start - iter_time) * 1000 if ind == 0 else 0.0
            dl_wait_ms = (t_iter_start - t_prev_iter_end) * 1000 if t_prev_iter_end is not None else 0.0

            if world > 1:
                dl_wait_max_ms = torch.tensor(dl_wait_ms, device=device)
                dist.all_reduce(dl_wait_max_ms, op=dist.ReduceOp.MAX)
                dl_wait_max_ms = dl_wait_max_ms.item()
            else:
                dl_wait_max_ms = dl_wait_ms

            use_cuda_events = device.type == "cuda"
            if use_cuda_events:
                e_fwd_s = torch.cuda.Event(enable_timing=True)
                e_fwd_e = torch.cuda.Event(enable_timing=True)
                e_bwd_s = torch.cuda.Event(enable_timing=True)
                e_bwd_e = torch.cuda.Event(enable_timing=True)

            t_h2d = time_mod.perf_counter()
            input_batch = Batch.from_data_list([x["data"] for x in batch]).to(device)
            affinity = torch.stack([x["affinity"] for x in batch]).float().view(-1).to(device)
            hbond = torch.stack([x["hbond"] for x in batch]).float().view(-1).to(device)
            hpbond = torch.stack([x["hpbond"] for x in batch]).float().view(-1).to(device)
            habond = torch.stack([x["habond"] for x in batch]).float().view(-1).to(device)
            sbond = torch.stack([x["sbond"] for x in batch]).float().view(-1).to(device)
            pbond = torch.stack([x["pbond"] for x in batch]).float().view(-1).to(device)
            h2d_ms = (time_mod.perf_counter() - t_h2d) * 1000

            if use_cuda_events:
                e_fwd_s.record()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.use_amp):
                pred, _, _ = model(input_batch)
                loss = loss_mse(pred[:, 0].float(), affinity)
                if args.out_dim > 1:
                    loss_hbond = loss_mse(pred[:, 1].float(), hbond)
                    loss_hpbond = loss_mse(pred[:, 2].float(), hpbond)
                    loss_habond = loss_mse(pred[:, 3].float(), habond)
                    loss_sbond = loss_mse(pred[:, 4].float(), sbond)
                    loss_pbond = loss_mse(pred[:, 5].float(), pbond)
                    loss = loss + (loss_hbond + loss_hpbond + loss_habond + loss_sbond
                                   + loss_pbond) * args.loss_weight
            if use_cuda_events:
                e_fwd_e.record()

            if use_cuda_events:
                e_bwd_s.record()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if use_cuda_events:
                e_bwd_e.record()

            loss_val = loss.detach().cpu().item()
            t_prev_iter_end = time_mod.perf_counter()
            losses.append(loss_val)

            if rank == 0:
                if use_cuda_events:
                    torch.cuda.synchronize()
                    fwd_ms = e_fwd_s.elapsed_time(e_fwd_e)
                    bwd_ms = e_bwd_s.elapsed_time(e_bwd_e)
                else:
                    fwd_ms = bwd_ms = 0.0
                iter_ms = (t_prev_iter_end - t_iter_start) * 1000
                sec_per_iter = (dl_wait_ms + iter_ms) / 1000.0
                print("[%d/%d]-[%d/%d] loss: %f, lr: %f, sec/iter: %.2f  "
                      "[startup=%.0fms dl=%.0fms dl_max=%.0fms h2d=%.0fms fwd=%.0fms "
                      "bwd+nccl=%.0fms iter=%.0fms]"
                      % (epoch + 1, args.epochs, ind, len(dataloader), loss_val, args.lr,
                         sec_per_iter, startup_ms, dl_wait_ms, dl_wait_max_ms, h2d_ms,
                         fwd_ms, bwd_ms, iter_ms), flush=True)

            if ind + 1 >= args.iters:
                break

        comp_time = time_mod.perf_counter() - start_time
        if rank == 0:
            print("[%d/%d] Elapsed time: %f, Average loss: %f"
                  % (epoch + 1, args.epochs, comp_time, sum(losses) / len(losses)), flush=True)

    if read_fpath is not None and not args.no_cleanup:
        try:
            os.remove(read_fpath)
        except OSError:
            pass

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
