"""
Comprehensive graph coloring benchmark: custom, igraph, PETSc (all types), NetworkX.

Usage (inside Docker):
  Serial:
    python3 experiment_scripts/bench_graph_coloring_all.py
  Parallel (PETSc + custom):
    mpirun -n 16 python3 experiment_scripts/bench_graph_coloring_all.py

Measures coloring time and number of colors for all three problems.
Stops progressing through levels once a coloring exceeds TIME_LIMIT.
Results written to tmp_work/ as JSON.
"""

from graph_coloring.coloring_petsc import color_petsc
from graph_coloring.mesh_loader import PROBLEMS, load_adjacency
import json
import os
import sys
import time

import numpy as np
from mpi4py import MPI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


TMP_DIR = os.path.join(REPO_ROOT, "tmp_work")
os.makedirs(TMP_DIR, exist_ok=True)
TIME_LIMIT = 10.0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# ---------------------------------------------------------------------------
# Discover available backends
# ---------------------------------------------------------------------------
HAS_IGRAPH = False
HAS_NETWORKX = False
HAS_CUSTOM = False

# Custom coloring (C backend) – works in serial and parallel
try:
    from graph_coloring.coloring_custom import color_custom, color_custom_mpi
    HAS_CUSTOM = True
except Exception:
    pass

if nprocs == 1:
    try:
        from graph_coloring.coloring_igraph import color_igraph
        HAS_IGRAPH = True
    except ImportError:
        pass
    try:
        from graph_coloring.coloring_networkx import color_networkx, HAS_NETWORKX as _nx
        HAS_NETWORKX = _nx
    except ImportError:
        pass

# PETSc coloring types to benchmark — test all types in serial and parallel
PETSC_TYPES = ["greedy", "jp", "sl", "lf", "id"]

# NetworkX strategies (serial only, and only on smaller meshes due to speed)
NX_STRATEGIES = ["DSATUR", "largest_first", "smallest_last"]
NX_SIZE_LIMIT = 50000  # skip NetworkX for N > this


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def bench_petsc(adjacency, ctype, comm):
    comm.Barrier()
    t0 = time.perf_counter()
    n_colors, _ = color_petsc(adjacency, coloring_type=ctype, comm=comm)
    comm.Barrier()
    return int(n_colors), time.perf_counter() - t0


def bench_igraph(adjacency):
    t0 = time.perf_counter()
    n_colors, _ = color_igraph(adjacency)
    return int(n_colors), time.perf_counter() - t0


def bench_networkx(adjacency, strategy):
    t0 = time.perf_counter()
    n_colors, _ = color_networkx(adjacency, strategy=strategy)
    return int(n_colors), time.perf_counter() - t0


def bench_custom(adjacency):
    t0 = time.perf_counter()
    n_colors, _ = color_custom(adjacency)
    return int(n_colors), time.perf_counter() - t0


def bench_custom_mpi(adjacency, comm):
    comm.Barrier()
    t0 = time.perf_counter()
    n_colors, _ = color_custom_mpi(adjacency, comm=comm)
    comm.Barrier()
    return int(n_colors), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_results = {}

    for pname, pinfo in PROBLEMS.items():
        if rank == 0:
            print(f"\n{'=' * 90}")
            print(f" {pname}  (nprocs={nprocs})")
            print(f"{'=' * 90}")

        results = []
        exceeded_methods = set()  # track which methods exceeded time limit

        for level in pinfo["levels"]:
            h5_path = pinfo["path"](level)
            if not os.path.isfile(h5_path):
                if rank == 0:
                    print(f"  Level {level}: FILE NOT FOUND")
                continue

            adjacency = load_adjacency(h5_path)
            n = adjacency.shape[0]
            nnz = adjacency.nnz

            if rank == 0:
                print(f"\n  Level {level} | N={n:>10,} | nnz={nnz:>12,}")

            row_data = {"level": level, "n_vertices": int(n), "nnz_adjacency": int(nnz)}

            all_exceeded = True  # will be set to False if any method still runs

            # ---- PETSc ----
            for ctype in PETSC_TYPES:
                key = f"petsc_{ctype}"
                if key in exceeded_methods:
                    continue
                all_exceeded = False
                try:
                    nc, t = bench_petsc(adjacency, ctype, comm)
                    row_data[f"{key}_colors"] = nc
                    row_data[f"{key}_time"] = round(t, 6)
                    if rank == 0:
                        print(f"    petsc-{ctype:>7}: {nc:>4} colors  {t:>8.4f}s")
                    if t > TIME_LIMIT:
                        exceeded_methods.add(key)
                except Exception as e:
                    if rank == 0:
                        print(f"    petsc-{ctype:>7}: ERROR {e}")

            # ---- igraph (serial) ----
            if HAS_IGRAPH and "igraph" not in exceeded_methods:
                all_exceeded = False
                nc, t = bench_igraph(adjacency)
                row_data["igraph_colors"] = nc
                row_data["igraph_time"] = round(t, 6)
                if rank == 0:
                    print(f"    igraph:          {nc:>4} colors  {t:>8.4f}s")
                if t > TIME_LIMIT:
                    exceeded_methods.add("igraph")

            # ---- Custom (serial or parallel) ----
            if HAS_CUSTOM and "custom" not in exceeded_methods:
                all_exceeded = False
                try:
                    if nprocs == 1:
                        nc, t = bench_custom(adjacency)
                    else:
                        nc, t = bench_custom_mpi(adjacency, comm)
                    row_data["custom_colors"] = nc
                    row_data["custom_time"] = round(t, 6)
                    if rank == 0:
                        print(f"    custom:          {nc:>4} colors  {t:>8.4f}s")
                    if t > TIME_LIMIT:
                        exceeded_methods.add("custom")
                except Exception as e:
                    if rank == 0:
                        print(f"    custom: ERROR {e}")

            # ---- NetworkX (serial, small only) ----
            if HAS_NETWORKX and n <= NX_SIZE_LIMIT:
                for strat in NX_STRATEGIES:
                    key = f"nx_{strat}"
                    if key in exceeded_methods:
                        continue
                    all_exceeded = False
                    try:
                        nc, t = bench_networkx(adjacency, strat)
                        row_data[f"{key}_colors"] = nc
                        row_data[f"{key}_time"] = round(t, 6)
                        if rank == 0:
                            print(f"    nx-{strat:>15}: {nc:>4} colors  {t:>8.4f}s")
                        if t > TIME_LIMIT:
                            exceeded_methods.add(key)
                    except Exception as e:
                        if rank == 0:
                            print(f"    nx-{strat:>15}: ERROR {e}")

            results.append(row_data)

            if all_exceeded:
                if rank == 0:
                    print(f"  *** All methods exceeded {TIME_LIMIT}s – stopping {pname}")
                break

        all_results[pname] = results

    # Save JSON
    if rank == 0:
        suffix = f"np{nprocs}"
        out_path = os.path.join(TMP_DIR, f"graph_coloring_all_{suffix}.json")
        with open(out_path, "w") as f:
            json.dump({"nprocs": nprocs, "results": all_results}, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
