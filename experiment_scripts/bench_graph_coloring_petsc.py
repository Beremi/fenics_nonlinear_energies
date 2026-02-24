"""
Benchmark graph coloring: igraph vs PETSc (serial & parallel).

Usage (inside Docker):
  Serial:
    python3 experiment_scripts/bench_graph_coloring_petsc.py
  Parallel:
    mpirun -n 16 python3 experiment_scripts/bench_graph_coloring_petsc.py

Measures coloring time and number of colors for all three problems.
Stops progressing through levels once a coloring exceeds 10 s.
Results are written to tmp_work/ as JSON.
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

# Determine which coloring types to benchmark
# "greedy" works both serial and parallel; "jp" is Jones-Plassmann (parallel-friendly)
COLORING_TYPES = ["greedy"]
if nprocs > 1:
    COLORING_TYPES.append("jp")

# Also benchmark igraph if serial
if nprocs == 1:
    try:
        from graph_coloring.coloring_igraph import color_igraph
        HAS_IGRAPH = True
    except ImportError:
        HAS_IGRAPH = False
else:
    HAS_IGRAPH = False


def run_benchmark_petsc(adjacency, coloring_type, comm):
    """Run a single PETSc coloring benchmark, return (n_colors, elapsed)."""
    comm.Barrier()
    t0 = time.perf_counter()
    n_colors, coloring = color_petsc(adjacency, coloring_type=coloring_type, comm=comm)
    comm.Barrier()
    elapsed = time.perf_counter() - t0
    return n_colors, elapsed


def run_benchmark_igraph(adjacency):
    """Run a single igraph coloring benchmark, return (n_colors, elapsed)."""
    t0 = time.perf_counter()
    n_colors, coloring = color_igraph(adjacency)
    elapsed = time.perf_counter() - t0
    return n_colors, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_results = {}

    for pname, pinfo in PROBLEMS.items():
        if rank == 0:
            print(f"\n{'=' * 80}")
            print(f" {pname}  (nprocs={nprocs})")
            print(f"{'=' * 80}")

        results = []
        exceeded = False

        for level in pinfo["levels"]:
            h5_path = pinfo["path"](level)
            if not os.path.isfile(h5_path):
                if rank == 0:
                    print(f"  Level {level}: FILE NOT FOUND")
                continue

            adjacency = load_adjacency(h5_path)
            n = adjacency.shape[0]
            nnz = adjacency.nnz

            row_data = {"level": level, "n_vertices": int(n), "nnz_adjacency": int(nnz)}

            # PETSc coloring types
            for ctype in COLORING_TYPES:
                n_colors, elapsed = run_benchmark_petsc(adjacency, ctype, comm)
                row_data[f"petsc_{ctype}_colors"] = int(n_colors)
                row_data[f"petsc_{ctype}_time"] = round(elapsed, 6)

                if rank == 0:
                    print(f"  Level {level} | N={n:>10,} | nnz={nnz:>12,} | "
                          f"petsc-{ctype}: {n_colors:>3} colors, {elapsed:.4f}s")

                if elapsed > TIME_LIMIT:
                    exceeded = True

            # igraph (serial only)
            if HAS_IGRAPH:
                n_colors, elapsed = run_benchmark_igraph(adjacency)
                row_data["igraph_colors"] = int(n_colors)
                row_data["igraph_time"] = round(elapsed, 6)
                if rank == 0:
                    print(f"  Level {level} | N={n:>10,} | nnz={nnz:>12,} | "
                          f"igraph:       {n_colors:>3} colors, {elapsed:.4f}s")
                if elapsed > TIME_LIMIT:
                    exceeded = True

            results.append(row_data)

            if exceeded:
                if rank == 0:
                    print(f"  *** Exceeded {TIME_LIMIT}s – skipping remaining levels for {pname}")
                break

        all_results[pname] = results

    # Save JSON (rank 0 only)
    if rank == 0:
        suffix = f"np{nprocs}"
        out_path = os.path.join(TMP_DIR, f"graph_coloring_petsc_{suffix}.json")
        with open(out_path, "w") as f:
            json.dump({"nprocs": nprocs, "results": all_results}, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
