#!/usr/bin/env python3
"""
generate_synthetic_hdf5.py -- build a fully self-contained, structurally-real
HDF5 dataset for pecan_train_bench.py's `--io-style real` mode, so the
severe-interference reproduction from real HDF5 access (see README) doesn't
require the private SAIR dataset.

Mirrors the real dataset's exact on-disk layout that pecan/dataset.py's
Dataset_PDB and pecan/precompute_graphs.py actually read/wrote:

  raw file:   h5[pdbid]["dcomplex"][str(poseid)]/coord   (N,3) float32
                                                  /feat    (N,19) float32
                                     .attrs: num_hbonds, num_hydrophobic_contacts,
                                             num_halogenbonds, num_salt_bridges,
                                             num_pi_stacking

  graph file: h5[pdbid][str(poseid)]/edge_index  (2,E) int32
                                     /edge_attr   (E,) float32

...across --n-files separate raw files (default 105, matching the real
dataset) and one companion "*_graph.h5" per raw file, each holding
--groups-per-file distinct pdbid groups with --poses-per-group pose
subgroups apiece -- comfortably above the ~8-16 entry threshold at which
HDF5 switches a group's link storage from compact to a B-tree-indexed
symbol table, which is the real per-sample cost pecan/dataset.py's
`h5[pdbid][...]` lookups actually pay and none of pecan_train_bench.py's
earlier synthetic I/O approximations (flat reads, scattered reads, repeated
small-file open/close) reproduced.

A combined CSV (fdir, fn, pdbid, poseid, affinity, rmsd, score) is written
alongside, in the exact format pecan.dataset.Dataset_PDB expects -- so
--real-csv/--real-graph-cache-dir can point straight at this output.

Usage:
    python generate_synthetic_hdf5.py --out-dir /p/lustre5/yourdir/synthdata \
        --n-files 105 --groups-per-file 500 --poses-per-group 5

Then:
    python pecan_train_bench.py --io-style real \
        --real-csv /p/lustre5/yourdir/synthdata/synth_all.csv \
        --real-graph-cache-dir /p/lustre5/yourdir/synthdata/graph_cache
"""
import os
import math
import argparse
import multiprocessing as mp

import numpy as np
import h5py


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a synthetic, structurally-real HDF5 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-files", type=int, default=105,
                   help="Number of raw files (+ one companion graph file each) -- "
                        "matches the real dataset's file count")
    p.add_argument("--groups-per-file", type=int, default=500,
                   help="Distinct pdbid groups per file -- comfortably above the "
                        "~8-16 entry threshold where HDF5 switches from compact "
                        "to B-tree-indexed group storage (real files: ~10,000)")
    p.add_argument("--poses-per-group", type=int, default=5)
    p.add_argument("--min-atoms", type=int, default=400)
    p.add_argument("--max-atoms", type=int, default=1000)
    p.add_argument("--in-channels", type=int, default=19)
    p.add_argument("--distance-cutoff", type=float, default=5.0)
    p.add_argument("--avg-degree", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel file-generation processes (files are independent)")
    return p.parse_args()


def _make_one_group(rng, n_atoms, in_channels, distance_cutoff, avg_degree):
    coord = rng.uniform(0, 30, size=(n_atoms, 3)).astype(np.float32)
    feat = rng.standard_normal(size=(n_atoms, in_channels)).astype(np.float32)

    sphere_vol = (4.0 / 3.0) * math.pi * (distance_cutoff ** 3)
    box = (n_atoms * sphere_vol / avg_degree) ** (1.0 / 3.0)
    coord_for_edges = rng.uniform(0, box, size=(n_atoms, 3)).astype(np.float32)
    diff = coord_for_edges[:, None, :] - coord_for_edges[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    rows, cols = np.where((d > 0) & (d < distance_cutoff))
    edge_index = np.stack([rows, cols], axis=0).astype(np.int32)
    edge_attr = d[rows, cols].astype(np.float32)

    return coord, feat, edge_index, edge_attr


def generate_file(args_tuple):
    (file_idx, out_dir, n_groups, n_poses, min_atoms, max_atoms,
     in_channels, distance_cutoff, avg_degree, seed) = args_tuple

    rng = np.random.default_rng(seed + file_idx)
    raw_fn = f"synth_{file_idx:04d}.h5"
    graph_fn = f"synth_{file_idx:04d}_graph.h5"
    raw_path = os.path.join(out_dir, raw_fn)
    graph_path = os.path.join(out_dir, "graph_cache", graph_fn)

    rows = []  # (fdir, fn, pdbid, poseid, affinity, rmsd, score)
    with h5py.File(raw_path, "w") as h5_raw, h5py.File(graph_path, "w") as h5_graph:
        for g in range(n_groups):
            # "s" prefix keeps this non-numeric so pandas reads the CSV's
            # pdbid column as strings -- a bare digit string gets parsed as
            # int64 and silently loses its leading zeros on read-back,
            # breaking the h5[pdbid] lookup key.
            pdbid = f"s{file_idx:04d}{g:05d}"
            raw_grp = h5_raw.require_group(pdbid)
            dcom = raw_grp.create_group("dcomplex")
            graph_grp = h5_graph.require_group(pdbid)

            affinity = float(rng.uniform(4.0, 10.0))
            for poseid in range(1, n_poses + 1):
                n_atoms = int(rng.integers(min_atoms, max_atoms + 1))
                coord, feat, edge_index, edge_attr = _make_one_group(
                    rng, n_atoms, in_channels, distance_cutoff, avg_degree)

                pose_grp = dcom.create_group(str(poseid))
                pose_grp.create_dataset("coord", data=coord, compression="lzf")
                pose_grp.create_dataset("feat", data=feat, compression="lzf")
                pose_grp.attrs["num_hbonds"] = int(rng.integers(0, 10))
                pose_grp.attrs["num_hydrophobic_contacts"] = int(rng.integers(0, 10))
                pose_grp.attrs["num_halogenbonds"] = int(rng.integers(0, 10))
                pose_grp.attrs["num_salt_bridges"] = int(rng.integers(0, 10))
                pose_grp.attrs["num_pi_stacking"] = int(rng.integers(0, 10))

                gpose_grp = graph_grp.create_group(str(poseid))
                gpose_grp.create_dataset("edge_index", data=edge_index, compression="lzf")
                gpose_grp.create_dataset("edge_attr", data=edge_attr, compression="lzf")

                rmsd = float(rng.uniform(0, 8))
                score = float(rng.uniform(0, 5))
                rows.append((out_dir, raw_fn, pdbid, poseid, affinity, rmsd, score))

    print(f"[generate_synthetic_hdf5] file {file_idx+1} done: {raw_fn} "
          f"({n_groups} groups x up to {n_poses} poses)", flush=True)
    return rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "graph_cache"), exist_ok=True)

    tasks = [
        (i, args.out_dir, args.groups_per_file, args.poses_per_group,
         args.min_atoms, args.max_atoms, args.in_channels, args.distance_cutoff,
         args.avg_degree, args.seed)
        for i in range(args.n_files)
    ]

    print(f"[generate_synthetic_hdf5] generating {args.n_files} files x "
          f"{args.groups_per_file} groups x up to {args.poses_per_group} poses "
          f"(up to {args.n_files * args.groups_per_file * args.poses_per_group:,} "
          f"samples) with {args.workers} workers ...", flush=True)

    all_rows = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for rows in pool.imap_unordered(generate_file, tasks):
            all_rows.extend(rows)

    csv_path = os.path.join(args.out_dir, "synth_all.csv")
    with open(csv_path, "w") as f:
        f.write("fdir,fn,pdbid,poseid,affinity,rmsd,score\n")
        for fdir, fn, pdbid, poseid, affinity, rmsd, score in all_rows:
            f.write(f"{fdir},{fn},{pdbid},{poseid},{affinity},{rmsd},{score}\n")

    print(f"[generate_synthetic_hdf5] wrote {len(all_rows):,} samples across "
          f"{args.n_files} files to {args.out_dir}", flush=True)
    print(f"[generate_synthetic_hdf5] CSV: {csv_path}", flush=True)
    print(f"[generate_synthetic_hdf5] graph_cache: {os.path.join(args.out_dir, 'graph_cache')}",
          flush=True)


if __name__ == "__main__":
    main()
