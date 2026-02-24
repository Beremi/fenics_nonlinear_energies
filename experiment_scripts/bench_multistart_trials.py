#!/usr/bin/env python3
"""
Benchmark the multistart coloring pipeline with varying trials_per_rank.

Reports timing breakdown (A², Bcast, coloring) and best color count
for trials_per_rank = 1, 5, 10.

Usage:
    mpirun -n 16 python experiment_scripts/bench_multistart_trials.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from graph_coloring.mesh_loader import PROBLEMS, load_adjacency
from graph_coloring.multistart_coloring import multistart_color

BENCHMARKS = [
    ("pLaplace2D", 9),
    ("GinzburgLandau2D", 9),
    ("HyperElasticity3D", 4),
]

TRIALS_LIST = [1, 5, 10]

for prob_name, lvl in BENCHMARKS:
    h5 = PROBLEMS[prob_name]["path"](lvl)

    # Load adjacency on rank 0
    if rank == 0:
        A = load_adjacency(h5)
        n_ref = A.shape[0]
    else:
        A = None
        n_ref = 0
    n_ref = comm.bcast(n_ref, root=0)

    if rank == 0:
        print(f"\n{'='*90}")
        print(f"  {prob_name} level {lvl}  (N={n_ref:,}, np={size})")
        print(f"{'='*90}")
        print(f"  {'trials/rank':<12}  {'total trials':>12}  {'A²(s)':>7}  "
              f"{'Bcast':>7}  {'Color':>7}  {'Total':>7}  {'best #col':>9}")
        print(f"  {'-'*12}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}  "
              f"{'-'*7}  {'-'*9}")
        sys.stdout.flush()
    comm.Barrier()

    for trials in TRIALS_LIST:
        # Warmup run
        nc, colors, info = multistart_color(A, comm, trials_per_rank=trials)
        # Timed run
        nc, colors, info = multistart_color(A, comm, trials_per_rank=trials)

        if rank == 0:
            total_trials = trials * size
            print(f"  {trials:<12}  {total_trials:>12}  "
                  f"{info['a2_time']:>7.4f}  {info['bcast_time']:>7.4f}  "
                  f"{info['color_time']:>7.4f}  {info['total_time']:>7.4f}  "
                  f"{nc:>9}")
            sys.stdout.flush()
        comm.Barrier()

    # Also print per-rank detail for trials=10
    trials = 10
    nc, colors, info = multistart_color(A, comm, trials_per_rank=trials)
    all_trials = comm.gather(info['colors_per_trial'], root=0)

    if rank == 0:
        print(f"\n  Per-rank detail (trials_per_rank={trials}, "
              f"total={trials*size}):")
        for r, t_list in enumerate(all_trials):
            best = min(t_list)
            print(f"    rank {r:2d}: colors={t_list}  best={best}")
        print(f"  Global best: {nc}")
        sys.stdout.flush()
    comm.Barrier()

if rank == 0:
    print()
