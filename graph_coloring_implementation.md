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

---

## Benchmark results (np = 16)

Three benchmark problems on the largest mesh levels:

- **pLaplace 2D level 9:** $N = 784{,}385$, $\text{nnz}(A) = 5.5\text{M}$, $\text{nnz}(A^2) = 14.9\text{M}$
- **Ginzburg–Landau 2D level 9:** $N = 1{,}046{,}529$, $\text{nnz}(A) = 7.3\text{M}$, $\text{nnz}(A^2) = 19.8\text{M}$
- **HyperElasticity 3D level 4:** $N = 554{,}013$, $\text{nnz}(A) = 23.4\text{M}$, $\text{nnz}(A^2) = 95.7\text{M}$

### Timing breakdown & best color count

| Problem          | trials/rank | total trials | A² (s) | Bcast (s) | Color (s) | Total (s) | best #col |
| ---------------- | ----------: | -----------: | -----: | --------: | --------: | --------: | --------: |
| **pLaplace 2D**  |           1 |           16 |   0.16 |      0.08 |      0.09 |      0.33 |         9 |
|                  |           5 |           80 |   0.16 |      0.08 |      0.42 |      0.66 |         8 |
|                  |          10 |          160 |   0.12 |      0.09 |      0.86 |      1.07 |         8 |
| **GL 2D**        |           1 |           16 |   0.25 |      0.12 |      0.13 |      0.49 |         9 |
|                  |           5 |           80 |   0.24 |      0.12 |      0.58 |      0.94 |         8 |
|                  |          10 |          160 |   0.25 |      0.11 |      1.14 |      1.50 |         8 |
| **HE 3D**        |           1 |           16 |   1.24 |      0.52 |      0.32 |      2.08 |        69 |
|                  |           5 |           80 |   1.24 |      0.49 |      1.53 |      3.25 |        69 |
|                  |          10 |          160 |   1.24 |      0.48 |      3.01 |      4.74 |        68 |

### Reference color counts (other methods)

| Problem     | igraph | PETSc SL | PETSc ID | Custom deterministic |
| ----------- | -----: | -------: | -------: | -------------------: |
| pLaplace 2D |      7 |       10 |        9 |                   11 |
| GL 2D       |      7 |        8 |        9 |                   11 |
| HE 3D       |     68 |       75 |       78 |                   70 |

### Observations

1. **A² + Bcast is a fixed cost** (~0.2–1.8 s depending on problem) that does
   not grow with `trials_per_rank`.  Coloring time scales linearly.

2. **5 trials per rank (80 total)** already achieves 8 colors for 2D problems
   (only 1 more than igraph's 7) and 69 for HE 3D (vs igraph's 68).
   With 10 trials (160 total), HE reaches **68 — matching igraph**.

3. **Bcast overhead** is proportional to `nnz(A²)`: 0.08 s for 15M entries,
   0.12 s for 20M, 0.5 s for 96M.  This is pure MPI_Bcast of int32 arrays.

4. **A² computation on rank 0** takes 0.12–1.24 s.  It is the scipy
   sparse matrix multiply `csr_matrix(A @ A)`.  With 16 processes resident
   in memory, this is ~50% slower than serial for HE 3D due to memory
   bandwidth contention; negligible for 2D.

5. **Total wall time** for the recommended `trials_per_rank=5`:
   0.66 s (pL), 0.94 s (GL), 3.25 s (HE).  This is faster than any
   other method achieving comparable color quality.
