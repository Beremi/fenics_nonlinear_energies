# Graph Coloring — Implementation & Build

## Building

The C shared library is pre-built and committed to git (`graph_coloring/custom_coloring.so`,
Linux x86-64). On a matching system you do not need to rebuild.

To rebuild (e.g. after editing `custom_coloring.c` or on a different architecture):

```bash
make            # build graph_coloring/custom_coloring.so
make clean      # remove .so
make rebuild    # clean + build
```

Requirements: `gcc` (any recent version).  No external libraries needed.

## Overview

Distance-2 graph coloring for sparse-finite-difference (SFD) Hessian recovery.
Given an element–DOF adjacency matrix $A$, a distance-2 coloring is equivalent
to a proper coloring of $A^2 = A \cdot A$.

### Pipeline (`graph_coloring/multistart_coloring.py`)

```
mpirun -n 16 python -c "
    from mpi4py import MPI
    from graph_coloring.multistart_coloring import multistart_color
    # adjacency loaded on rank 0
    n_colors, colors, info = multistart_color(A, MPI.COMM_WORLD, trials_per_rank=5)
"
```

**Three phases with MPI barriers between each:**

1. **A² computation** (rank 0 only) — `scipy.sparse.csr_matrix(A @ A)`.
   Other ranks idle at barrier.
2. **MPI Broadcast** — CSR `indptr` and `indices` arrays broadcast from
   rank 0 to all ranks.
3. **Multi-trial coloring** (all ranks, embarrassingly parallel) — each rank
   runs `trials_per_rank` independent randomised greedy colorings with
   different PRNG seeds. The globally best result is selected via
   `Allgather` + min.

### C backend (`graph_coloring/custom_coloring.c`)

Two exported functions used by the pipeline:

- **`custom_greedy_color_random(n, indptr, indices, colors, seed)`** —
  Randomised "most coloured neighbours" greedy coloring on $A^2$ (CSR).
  Uses xoshiro128\*\* PRNG for random starting vertex and tie-breaking.
  Returns number of colors.

- **`build_A2_pattern(n, ai, aj, a2_indptr, a2_indices, nnz_out)`** —
  Direct 2-hop adjacency computation (alternative to scipy A@A).
  Two-pass: first counts nnz per row, then fills sorted indices
  using a marker array for O(1) duplicate detection.

### Python wrapper (`graph_coloring/multistart_coloring.py`)

- `multistart_color(adjacency, comm, trials_per_rank=1)` —
  Full pipeline returning `(n_colors, colors_array, info_dict)`.
  `info_dict` contains timing breakdown (`a2_time`, `bcast_time`,
  `color_time`, `total_time`) and per-trial color counts.

- `greedy_color_random(n, indptr, indices, seed)` —
  Single randomised coloring (C backend via ctypes).

- `compute_A2_scipy(adjacency)` — Returns CSR `(n, indptr, indices)`.

- `bcast_csr(n, indptr, indices, comm)` — MPI broadcast of CSR arrays.

### Standalone CLI

```bash
mpirun -n 16 python graph_coloring/multistart_coloring.py  mesh_data/pLaplace/pLaplace_level9.h5  5
```

Arguments: `mesh.h5` path, optional `trials_per_rank` (default 5).

