# Graph Coloring Benchmark Results

## Overview

The JAX-based energy formulations require **sparse-finite-difference (SFD) Hessian
recovery**, which in turn relies on a **greedy graph coloring** of the
*connectivity* graph $C = A \cdot A$ (where $A$ is the element–DOF adjacency
matrix). The coloring is computed with
[`igraph.Graph.vertex_coloring_greedy`](https://python.igraph.org/).

This page reports, for every available mesh level of all three benchmark
problems:

| Metric      | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| **N**       | Number of vertices (DOFs) in the adjacency graph               |
| **nnz(A)**  | Non-zeros of the adjacency matrix                              |
| **nnz(A²)** | Non-zeros of the connectivity matrix                           |
| **Colors**  | Number of colors produced by greedy coloring                   |
| **Time**    | Wall-clock time (seconds) for building connectivity + coloring |

Benchmarks were stopped once a single coloring exceeded **10 s**.

*Script:* `experiment_scripts/bench_graph_coloring.py`

---

## p-Laplace 2D

| Level |       N |    nnz(A) |    nnz(A²) | Colors | Time (s) |
| ----: | ------: | --------: | ---------: | -----: | -------: |
|     1 |       5 |        13 |         19 |      3 |   0.0002 |
|     2 |      33 |       177 |        369 |      7 |   0.0001 |
|     3 |     161 |     1,009 |      2,481 |      8 |   0.0003 |
|     4 |     705 |     4,689 |     12,177 |      7 |   0.0013 |
|     5 |   2,945 |    20,113 |     53,457 |      8 |   0.0055 |
|     6 |  12,033 |    83,217 |    223,569 |      8 |   0.0228 |
|     7 |  48,641 |   338,449 |    914,001 |      8 |   0.0818 |
|     8 | 195,585 | 1,365,009 |  3,695,697 |      8 |   0.3290 |
|     9 | 784,385 | 5,482,513 | 14,862,417 |      7 |   1.4852 |

---

## Ginzburg–Landau 2D

| Level |         N |    nnz(A) |    nnz(A²) | Colors | Time (s) |
| ----: | --------: | --------: | ---------: | -----: | -------: |
|     2 |        49 |       289 |        671 |      7 |   0.0002 |
|     3 |       225 |     1,457 |      3,695 |      7 |   0.0003 |
|     4 |       961 |     6,481 |     17,039 |      8 |   0.0012 |
|     5 |     3,969 |    27,281 |     72,911 |      7 |   0.0051 |
|     6 |    16,129 |   111,889 |    301,391 |      7 |   0.0231 |
|     7 |    65,025 |   453,137 |  1,225,295 |      8 |   0.1011 |
|     8 |   261,121 | 1,823,761 |  4,940,879 |      7 |   0.4586 |
|     9 | 1,046,529 | 7,317,521 | 19,843,151 |      7 |   2.0018 |

---

## HyperElasticity 3D

| Level |       N |     nnz(A) |    nnz(A²) | Colors | Time (s) |
| ----: | ------: | ---------: | ---------: | -----: | -------: |
|     1 |   2,133 |     64,251 |    184,617 |     48 |   0.0130 |
|     2 |  11,925 |    426,411 |  1,488,897 |     58 |   0.1266 |
|     3 |  77,517 |  3,081,123 | 11,949,273 |     63 |   1.1740 |
|     4 | 554,013 | 23,369,715 | 95,726,889 |     68 |  10.1262 |

*(Level 4 exceeded the 10 s limit — remaining levels skipped.)*

---

## Key observations

1. **2D problems (p-Laplace, Ginzburg–Landau)** require only **7–8 colors**
   regardless of mesh size, which is consistent with the bounded vertex degree
   of structured triangular meshes in 2D.

2. **HyperElasticity 3D** needs **48–68 colors** — substantially more because
   the 3D tetrahedral connectivity graph has much higher vertex degree.

3. **Scaling.** Coloring time grows roughly linearly with the number of
   non-zeros in the connectivity matrix. For the 2D problems, meshes up to
   ~10⁶ DOFs are colored in about 2 s. The 3D problem hits the 10 s wall at
   ~550 k DOFs.

4. The number of colors directly determines the number of Hessian–vector
   products (tangent evaluations) needed per SFD Hessian assembly, so the
   3D problem pays roughly an **8–9× higher per-iteration cost** from
   coloring alone compared to 2D.
