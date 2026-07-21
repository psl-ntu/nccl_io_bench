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
edges from graph_cache_dir rather than building them on the fly. An
optional --io-style real mode reads actual HDF5 files instead (see
RealHDF5PDBGraphDataset below); this still has no dependency on the real
training codebase itself -- only h5py and the CSV/HDF5 file format it reads
are shared with it -- so the whole script, including this mode, remains a
single self-contained file.

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
import csv
import math
import argparse

import h5py
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
    per-rank test file at that path -- a single bulk transfer at a random
    offset -- so `dl`/`dl_max` in this script's output reflects genuine
    storage-tier latency. Point --fs-path at Lustre, VAST, node-local NVMe,
    etc. to compare, the same way nccl_io_bench.py's --fs-path is used, but
    with the real model+training loop generating the concurrent compute+comm
    load instead of a bare tensor fill.

    This bulk-read approximation only reproduces mild I/O-vs-NCCL
    interference (see README) -- reproducing the real dataset's severe tail
    behavior requires genuine HDF5 access patterns, which is what
    --io-style real (build_real_dataloader, below) uses instead.
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

    def _ensure_open(self):
        """Lazily open this worker process's own handle on self.read_fpath,
        so each DataLoader worker (forked independently) gets its own fd,
        never one inherited/shared across workers from the parent process
        (mirrors pecan/dataset.py's self._h5_handles lazy-open-per-process
        pattern)."""
        if self._fh is None:
            import random as _random
            self._fh = open(self.read_fpath, "rb")
            self._file_size = os.fstat(self._fh.fileno()).st_size
            self._rng_io = _random.Random(os.getpid())

    def _read_at_random_offset(self, nbytes):
        max_off = max(0, self._file_size - nbytes)
        offset = self._rng_io.randint(0, max_off)
        self._fh.seek(offset)
        self._fh.read(nbytes)

    def _read_real_bytes(self):
        self._ensure_open()
        self._read_at_random_offset(self.sample_bytes)

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
# io-style=real: genuine HDF5 access against real (or generate_synthetic_
# hdf5.py-generated) data, inlined directly here rather than importing the
# real training codebase's pecan.dataset.Dataset_PDB -- this file has no
# dependency on that private repo anywhere, by design (see module
# docstring), and --io-style real is no exception: it reads the exact same
# on-disk group/dataset/attribute layout Dataset_PDB does, using only h5py
# and the stdlib csv module.
# =============================================================================

H5_DCOMPLEX = "dcomplex"
H5_COORD = "coord"
H5_FEAT = "feat"
H5_NUM_HBONDS = "num_hbonds"
H5_NUM_HYDROPHOBIC = "num_hydrophobic_contacts"
H5_NUM_HALOGEN = "num_halogenbonds"
H5_NUM_SALT = "num_salt_bridges"
H5_NUM_PI = "num_pi_stacking"


class RealHDF5PDBGraphDataset(Dataset):
    """
    Reads real (or generate_synthetic_hdf5.py-generated) PECAN/SAIR-format
    HDF5 files directly: nested h5[pdbid]["dcomplex"][poseid] groups holding
    coord/feat datasets and 5 scalar attributes, plus a companion
    <fn>_graph.h5 file per raw file holding precomputed edge_index/edge_attr
    at h5[pdbid][poseid]. This is the same on-disk format and the same
    per-sample access sequence (CSV row -> raw-file group lookup -> graph-
    cache-file group lookup) the real training codebase's Dataset_PDB uses --
    intentionally reimplemented here (only the ~40 lines this benchmark
    actually needs: docking poses only, affinity labels, graph
    representation) rather than imported, so this file has zero dependency
    on the private training repo.

    Only "docking pose" rows are used (poseid 1..max_poses; the crystal-pose
    poseid==0 case, and the on-the-fly graph-construction fallback for when
    no graph_cache_dir is given, are both real Dataset_PDB features this
    benchmark doesn't need and doesn't reimplement).
    """

    def __init__(self, csv_path, graph_cache_dir, max_poses=5):
        self.graph_cache_dir = graph_cache_dir
        self.data_list = []  # (fpath, pdbid, poseid, affinity)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                poseid = int(row["poseid"])
                if poseid == 0 or poseid > max_poses:
                    continue
                rmsd = float(row["rmsd"])
                if rmsd < -1:  # -1000 marks an error row in the real dataset
                    continue
                fpath = os.path.join(row["fdir"], row["fn"])
                self.data_list.append((fpath, row["pdbid"], poseid, float(row["affinity"])))

        self._h5_handles = {}      # per-process cache: fpath -> h5py.File
        self._graph_handles = {}   # per-process cache: graph_fpath -> h5py.File

    def __len__(self):
        return len(self.data_list)

    def _h5(self, fpath):
        if fpath not in self._h5_handles:
            self._h5_handles[fpath] = h5py.File(fpath, "r")
        return self._h5_handles[fpath]

    def _graph_h5(self, fn):
        graph_fpath = os.path.join(self.graph_cache_dir, fn.replace(".h5", "_graph.h5"))
        if graph_fpath not in self._graph_handles:
            self._graph_handles[graph_fpath] = h5py.File(graph_fpath, "r")
        return self._graph_handles[graph_fpath]

    def __getitem__(self, ind):
        fpath, pdbid, poseid, affinity = self.data_list[ind]

        h5_data = self._h5(fpath)[pdbid][H5_DCOMPLEX][str(poseid)]
        coord = h5_data[H5_COORD][:]
        feat = h5_data[H5_FEAT][:]
        bond_counts = [
            h5_data.attrs[H5_NUM_HBONDS], h5_data.attrs[H5_NUM_HYDROPHOBIC],
            h5_data.attrs[H5_NUM_HALOGEN], h5_data.attrs[H5_NUM_SALT],
            h5_data.attrs[H5_NUM_PI],
        ]

        gh5 = self._graph_h5(os.path.basename(fpath))
        g_group = gh5[pdbid][str(poseid)]
        edge_index = torch.from_numpy(g_group["edge_index"][:]).long()
        edge_attr = torch.from_numpy(g_group["edge_attr"][:]).float()

        data = Data()
        data.pos = torch.from_numpy(coord)
        data.x = torch.from_numpy(feat).float()
        data.edge_index = edge_index
        data.edge_attr = edge_attr.view(-1, 1)

        return {
            "data": data,
            "affinity": torch.tensor([affinity], dtype=torch.float32),
            "hbond": torch.tensor([bond_counts[0]], dtype=torch.float32),
            "hpbond": torch.tensor([bond_counts[1]], dtype=torch.float32),
            "habond": torch.tensor([bond_counts[2]], dtype=torch.float32),
            "sbond": torch.tensor([bond_counts[3]], dtype=torch.float32),
            "pbond": torch.tensor([bond_counts[4]], dtype=torch.float32),
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
    p.add_argument("--iters", type=int, default=100, help="Measured iterations per phase (per rank)")
    p.add_argument("--warmup", type=int, default=10,
                   help="Warmup iterations per phase, excluded from reported stats "
                        "(first iterations pay one-time compile/cache costs that "
                        "would otherwise confound a baseline-vs-with-io comparison)")
    p.add_argument("--baseline-only", action="store_true",
                   help="Run only the no-I/O baseline phase (skip --fs-path phase)")
    p.add_argument("--io-only", action="store_true",
                   help="Run only the --fs-path phase (skip the no-I/O baseline)")
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
    p.add_argument("--io-style", choices=["flat", "real"], default="flat",
                   help="'flat': one bulk read of --sample-kb per sample from a "
                        "synthetic per-rank test file at a random offset -- cheap, "
                        "dependency-free, but only reproduces mild I/O-vs-NCCL "
                        "interference (see README). "
                        "'real': skip the synthetic approximation entirely and read "
                        "real HDF5 files directly (see --real-csv/"
                        "--real-graph-cache-dir, and RealHDF5PDBGraphDataset above) "
                        "-- real coord/feat/edge_index/edge_attr, real group/"
                        "attribute lookups, no guessing about access shape, and no "
                        "dependency on the real training codebase. Use "
                        "generate_synthetic_hdf5.py to build a dependency-free "
                        "dataset with the same on-disk structure if you don't have "
                        "access to the real one.")
    p.add_argument("--real-csv", default=None,
                   help="[io-style=real] Path to the training CSV (real or "
                        "generate_synthetic_hdf5.py output, e.g. .../synth_all.csv)")
    p.add_argument("--real-graph-cache-dir", default=None,
                   help="[io-style=real] Path to the precomputed graph-cache "
                        "directory (real or generate_synthetic_hdf5.py output)")
    p.add_argument("--file-mb", type=int, default=512,
                   help="Per-rank test file size on --fs-path")
    p.add_argument("--no-cleanup", action="store_true",
                   help="Keep the --fs-path test file after the run")
    return p.parse_args()


def build_dataloader(args, world, rank, read_fpath):
    dataset_len = (args.warmup + args.iters) * args.batch_size * world
    dataset = SyntheticPDBGraphDataset(
        length=dataset_len, pool_size=args.pool_size,
        min_atoms=args.min_atoms, max_atoms=args.max_atoms,
        distance_cutoff=args.distance_cutoff, avg_degree=args.avg_degree,
        in_channels=args.in_channels, out_dim=args.out_dim, seed=args.seed,
        read_fpath=read_fpath, sample_kb=args.sample_kb)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    return DataListLoader(dataset=dataset, shuffle=(sampler is None), batch_size=args.batch_size,
                           num_workers=args.num_workers, sampler=sampler, pin_memory=True, drop_last=True)


def build_real_dataloader(args, world, rank):
    """io-style=real: genuine HDF5 access against real (or generate_synthetic_
    hdf5.py-generated) data via RealHDF5PDBGraphDataset, instead of any
    synthetic approximation -- and with no dependency on the real training
    codebase (see that class's docstring). Returns a dict shaped identically
    to SyntheticPDBGraphDataset's (data/affinity/hbond/hpbond/habond/sbond/
    pbond), so run_phase()'s training loop needs no changes at all -- only
    the data source differs."""
    if args.real_csv is None:
        raise ValueError("--io-style real requires --real-csv")

    if rank == 0:
        print(f"[pecan_train_bench] Loading real dataset metadata from "
              f"{args.real_csv} (graph_cache_dir={args.real_graph_cache_dir}) ...",
              flush=True)
    dataset = RealHDF5PDBGraphDataset(
        csv_path=args.real_csv, graph_cache_dir=args.real_graph_cache_dir, max_poses=5)
    if rank == 0:
        print(f"[pecan_train_bench] Real dataset ready: {len(dataset):,} samples", flush=True)

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    return DataListLoader(dataset=dataset, shuffle=(sampler is None), batch_size=args.batch_size,
                           num_workers=args.num_workers, sampler=sampler, pin_memory=True, drop_last=True)


def run_phase(label, model, optimizer, loss_mse, dataloader, args, rank, world, device):
    """Run warmup+iters iterations, tagging every printed line with `label` so
    the two phases (baseline vs with-io) are distinguishable in the raw log,
    and return per-iteration metric arrays (rank 0 only, warmup excluded) for
    the interference-factor comparison in print_comparison()."""
    import time as time_mod

    model.train()
    iter_time = time_mod.perf_counter()
    t_prev_iter_end = None
    losses = []
    metrics = {"dl_max": [], "h2d": [], "fwd": [], "bwd_nccl": [], "iter": []}
    total = args.warmup + args.iters
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
            tag = "warmup" if ind < args.warmup else "meas"
            print("[%s/%s]-[%d/%d] loss: %f, lr: %f, sec/iter: %.2f  "
                  "[startup=%.0fms dl=%.0fms dl_max=%.0fms h2d=%.0fms fwd=%.0fms "
                  "bwd+nccl=%.0fms iter=%.0fms]"
                  % (label, tag, ind, total, loss_val, args.lr,
                     sec_per_iter, startup_ms, dl_wait_ms, dl_wait_max_ms, h2d_ms,
                     fwd_ms, bwd_ms, iter_ms), flush=True)
            if ind >= args.warmup:
                metrics["dl_max"].append(dl_wait_max_ms)
                metrics["h2d"].append(h2d_ms)
                metrics["fwd"].append(fwd_ms)
                metrics["bwd_nccl"].append(bwd_ms)
                metrics["iter"].append(iter_ms)

        if ind + 1 >= total:
            break

    comp_time = time_mod.perf_counter() - start_time
    if rank == 0:
        print("[%s] Elapsed time: %f, Average loss: %f"
              % (label, comp_time, sum(losses) / len(losses)), flush=True)
    return {k: np.array(v) for k, v in metrics.items()}


def print_comparison(rank, phases):
    """phases: ordered dict-like list of (label, metrics) pairs. Prints
    percentiles per phase and, if exactly two phases ran, an interference
    factor (phase2 median / phase1 median) for bwd_nccl and dl_max --
    the same style as nccl_io_bench.py's own baseline-vs-with-io report."""
    if rank != 0:
        return
    print()
    print("=" * 72)
    print("  pecan_train_bench: baseline vs. with-I/O comparison")
    print("=" * 72)
    hdr = f"  {'Phase':<12}  {'bwd+nccl med':>13}  {'p95':>9}  {'max':>9}  {'dl_max med':>11}  {'p95':>9}  {'max':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, m in phases:
        bwd = m["bwd_nccl"]
        dl = m["dl_max"]
        print(f"  {label:<12}  {np.median(bwd):>11.1f}ms  {np.percentile(bwd,95):>7.1f}ms  "
              f"{bwd.max():>7.1f}ms  {np.median(dl):>9.1f}ms  {np.percentile(dl,95):>7.1f}ms  "
              f"{dl.max():>7.1f}ms")
    print()
    if len(phases) == 2:
        (label1, m1), (label2, m2) = phases
        bwd_factor = np.median(m2["bwd_nccl"]) / max(np.median(m1["bwd_nccl"]), 0.001)
        dl_factor = np.median(m2["dl_max"]) / max(np.median(m1["dl_max"]), 0.001)
        corr = np.corrcoef(m2["dl_max"], m2["bwd_nccl"])[0, 1] if len(m2["dl_max"]) > 1 else float("nan")
        print(f"  bwd+nccl median, {label2} vs {label1}: {bwd_factor:.2f}x")
        print(f"  dl_max median,   {label2} vs {label1}: {dl_factor:.2f}x")
        print(f"  same-iteration corr(dl_max, bwd_nccl) in {label2} phase: {corr:.3f}")
    print("=" * 72)
    print()


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

    phases = []

    if not args.io_only:
        if rank == 0:
            print(f"[pecan_train_bench] Phase: baseline (no I/O) ...", flush=True)
        dataloader = build_dataloader(args, world, rank, read_fpath=None)
        m = run_phase("baseline", model, optimizer, loss_mse, dataloader, args, rank, world, device)
        phases.append(("baseline", m))
        if dist.is_initialized():
            dist.barrier()

    read_fpath = None
    real_io = args.io_style == "real"
    run_with_io = not args.baseline_only and (real_io or args.fs_path is not None)
    if run_with_io:
        if real_io:
            if rank == 0:
                print(f"[pecan_train_bench] Phase: with-io ({args.num_workers} workers/rank, "
                      f"io-style=real, {args.real_csv}) ...", flush=True)
            dataloader = build_real_dataloader(args, world, rank)
        else:
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
            if rank == 0:
                print(f"[pecan_train_bench] Phase: with-io ({args.num_workers} workers/rank, "
                      f"{args.sample_kb:.0f} KB/sample from {args.fs_path}, "
                      f"io-style={args.io_style}) ...", flush=True)
            dataloader = build_dataloader(args, world, rank, read_fpath=read_fpath)
        m = run_phase("with-io", model, optimizer, loss_mse, dataloader, args, rank, world, device)
        phases.append(("with-io", m))
        if dist.is_initialized():
            dist.barrier()

    print_comparison(rank, phases)

    if not args.no_cleanup and read_fpath is not None:
        try:
            os.remove(read_fpath)
        except OSError:
            pass

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
