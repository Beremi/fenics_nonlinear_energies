# Graph Coloring Benchmark Results

## Overview

The JAX-based energy formulations require **sparse-finite-difference (SFD) Hessian
recovery**, which in turn relies on a **distance-2 graph coloring** of the
element–DOF adjacency matrix $A$. A distance-2 coloring is equivalent to a
proper coloring of the connectivity graph $C = A \cdot A$.

Three libraries with multiple strategies are compared:

| Backend              | Library / Function                    | Algorithm                                       | Serial / Parallel |
| -------------------- | ------------------------------------- | ----------------------------------------------- | ----------------- |
| **igraph**           | `igraph.Graph.vertex_coloring_greedy` | Sequential greedy on explicitly formed $A^2$    | Serial only       |
| **PETSc greedy**     | `MatColoring` (`MATCOLORINGGREEDY`)   | Greedy distance-2 coloring                      | Both              |
| **PETSc JP**         | `MatColoring` (`MATCOLORINGJP`)       | Jones–Plassmann parallel coloring               | Both              |
| **PETSc SL**         | `MatColoring` (`MATCOLORINGSL`)       | Smallest-Last ordering + sequential coloring    | Both              |
| **PETSc LF**         | `MatColoring` (`MATCOLORINGLF`)       | Largest-First ordering + sequential coloring    | Both              |
| **PETSc ID**         | `MatColoring` (`MATCOLORINGID`)       | Incidence-Degree ordering + sequential coloring | Both              |
| **NX DSATUR**        | `networkx.coloring.greedy_color`      | DSATUR (saturation degree) heuristic            | Serial only       |
| **NX smallest_last** | `networkx.coloring.greedy_color`      | Smallest-last ordering                          | Serial only       |
| **NX largest_first** | `networkx.coloring.greedy_color`      | Largest-first ordering                          | Serial only       |

NetworkX is limited to $N \le 50{,}000$ due to its pure-Python overhead;
NX DSATUR is further limited by its $O(n^2)$ complexity.

*Scripts:*
- `experiment_scripts/bench_graph_coloring_all.py` (comprehensive, requires Docker / petsc4py)

*Reusable module:* `graph_coloring/` (igraph, PETSc, NetworkX backends)

---

## 1. Serial color quality (np = 1)

Fewer colors = fewer Hessian–vector products per SFD assembly step.

### p-Laplace 2D

| Level |       N |    nnz(A) | igraph | PETSc SL | PETSc ID | PETSc greedy | PETSc JP | PETSc LF | NX DSATUR | NX sm.last | NX lg.first |
| ----: | ------: | --------: | -----: | -------: | -------: | -----------: | -------: | -------: | --------: | ---------: | ----------: |
|     1 |       5 |        13 |      3 |        3 |        3 |            4 |        4 |        3 |         3 |          3 |           3 |
|     2 |      33 |       177 |      7 |        7 |        7 |            9 |        9 |        8 |         7 |          7 |           9 |
|     3 |     161 |     1,009 |      8 |        9 |        8 |           12 |       12 |       11 |         7 |          8 |          10 |
|     4 |     705 |     4,689 |      7 |        9 |        9 |           13 |       13 |       12 |         7 |          9 |          13 |
|     5 |   2,945 |    20,113 |      8 |       10 |        9 |           14 |       14 |       13 |         7 |         10 |          14 |
|     6 |  12,033 |    83,217 |      8 |        9 |        9 |           14 |       14 |       13 |         7 |          9 |          14 |
|     7 |  48,641 |   338,449 |      8 |       10 |        9 |           14 |       14 |       13 |         — |         10 |          14 |
|     8 | 195,585 | 1,365,009 |      8 |       10 |        9 |           15 |       15 |       13 |         — |          — |           — |
|     9 | 784,385 | 5,482,513 |      7 |       10 |        9 |           15 |       15 |       13 |         — |          — |           — |

### Ginzburg–Landau 2D

| Level |         N |    nnz(A) | igraph | PETSc SL | PETSc ID | PETSc greedy | PETSc JP | PETSc LF | NX DSATUR | NX sm.last | NX lg.first |
| ----: | --------: | --------: | -----: | -------: | -------: | -----------: | -------: | -------: | --------: | ---------: | ----------: |
|     2 |        49 |       289 |      7 |        8 |        8 |           12 |       12 |        8 |         7 |          8 |           9 |
|     3 |       225 |     1,457 |      7 |        9 |        9 |           12 |       12 |       12 |         7 |         10 |          13 |
|     4 |       961 |     6,481 |      8 |        8 |        9 |           14 |       14 |       13 |         7 |          8 |          14 |
|     5 |     3,969 |    27,281 |      7 |        8 |       10 |           14 |       14 |       13 |         7 |         10 |          14 |
|     6 |    16,129 |   111,889 |      7 |        8 |       10 |           15 |       15 |       13 |         7 |          8 |          14 |
|     7 |    65,025 |   453,137 |      8 |        8 |        9 |           14 |       14 |       13 |         — |          — |           — |
|     8 |   261,121 | 1,823,761 |      7 |        8 |        9 |           15 |       15 |       13 |         — |          — |           — |
|     9 | 1,046,529 | 7,317,521 |      7 |        8 |        9 |           15 |       15 |       13 |         — |          — |           — |

### HyperElasticity 3D

| Level |       N |     nnz(A) | igraph | PETSc SL | PETSc ID | PETSc greedy | PETSc JP | PETSc LF | NX DSATUR | NX sm.last | NX lg.first |
| ----: | ------: | ---------: | -----: | -------: | -------: | -----------: | -------: | -------: | --------: | ---------: | ----------: |
|     1 |   2,133 |     64,251 |     48 |       48 |       48 |           58 |       58 |       63 |        48 |         48 |          63 |
|     2 |  11,925 |    426,411 |     58 |       63 |       60 |           70 |       70 |       81 |        57 |         56 |          81 |
|     3 |  77,517 |  3,081,123 |     63 |       69 |       69 |           86 |       86 |       90 |         — |          — |           — |
|     4 | 554,013 | 23,369,715 |     68 |       75 |       78 |           91 |       91 |       90 |         — |          — |           — |

---

## 2. Serial timing (np = 1)

All times in seconds. NetworkX DSATUR omitted from this table because it is
100–10,000× slower than the other methods (e.g., 122 s for GL2D level 6
with $N = 16{,}129$).

### p-Laplace 2D

| Level |       N | igraph | PETSc SL | PETSc ID | PETSc greedy | NX sm.last |
| ----: | ------: | -----: | -------: | -------: | -----------: | ---------: |
|     5 |   2,945 |  0.004 |    0.004 |    0.004 |        0.004 |      0.038 |
|     6 |  12,033 |  0.018 |    0.016 |    0.016 |        0.017 |      0.241 |
|     7 |  48,641 |  0.079 |    0.065 |    0.068 |        0.069 |      1.155 |
|     8 | 195,585 |  0.308 |    0.262 |    0.260 |        0.294 |          — |
|     9 | 784,385 |  1.362 |    1.159 |    1.172 |        1.262 |          — |

### Ginzburg–Landau 2D

| Level |         N | igraph | PETSc SL | PETSc ID | PETSc greedy | NX sm.last |
| ----: | --------: | -----: | -------: | -------: | -----------: | ---------: |
|     5 |     3,969 |  0.005 |    0.006 |    0.006 |        0.006 |      0.056 |
|     6 |    16,129 |  0.023 |    0.022 |    0.021 |        0.024 |      0.343 |
|     7 |    65,025 |  0.100 |    0.087 |    0.087 |        0.096 |          — |
|     8 |   261,121 |  0.429 |    0.373 |    0.374 |        0.411 |          — |
|     9 | 1,046,529 |  1.936 |    1.741 |    1.718 |        1.881 |          — |

### HyperElasticity 3D

| Level |       N | igraph | PETSc SL | PETSc ID | PETSc greedy | NX sm.last |
| ----: | ------: | -----: | -------: | -------: | -----------: | ---------: |
|     1 |   2,133 |  0.013 |    0.006 |    0.006 |        0.010 |      0.126 |
|     2 |  11,925 |  0.127 |    0.040 |    0.040 |        0.072 |      1.374 |
|     3 |  77,517 |  1.114 |    0.291 |    0.292 |        0.515 |          — |
|     4 | 554,013 |  9.504 |    2.715 |    2.718 |        4.872 |          — |

---

## 3. Parallel results — PETSc, all types (np = 16)

### p-Laplace 2D

| Level |       N | greedy |       |   JP |       |   SL |       |   ID |       |   LF |       |
| ----: | ------: | -----: | ----: | ---: | ----: | ---: | ----: | ---: | ----: | ---: | ----: |
|       |         |   Cols |  Time | Cols |  Time | Cols |  Time | Cols |  Time | Cols |  Time |
|     5 |   2,945 |     16 | 0.001 |   13 | 0.002 |   10 | 0.002 |    9 | 0.001 |   13 | 0.001 |
|     6 |  12,033 |     17 | 0.003 |   14 | 0.005 |    9 | 0.004 |    9 | 0.004 |   13 | 0.003 |
|     7 |  48,641 |     17 | 0.011 |   14 | 0.017 |   10 | 0.019 |    9 | 0.019 |   13 | 0.016 |
|     8 | 195,585 |     19 | 0.048 |   14 | 0.073 |   10 | 0.124 |    9 | 0.124 |   13 | 0.098 |
|     9 | 784,385 |     19 | 0.255 |   15 | 0.422 |   10 | 0.674 |    9 | 0.684 |   13 | 0.591 |

### Ginzburg–Landau 2D

| Level |         N | greedy |       |   JP |       |   SL |       |   ID |       |   LF |       |
| ----: | --------: | -----: | ----: | ---: | ----: | ---: | ----: | ---: | ----: | ---: | ----: |
|       |           |   Cols |  Time | Cols |  Time | Cols |  Time | Cols |  Time | Cols |  Time |
|     5 |     3,969 |     17 | 0.002 |   14 | 0.002 |    8 | 0.002 |   10 | 0.002 |   13 | 0.001 |
|     6 |    16,129 |     18 | 0.005 |   14 | 0.006 |    8 | 0.006 |   10 | 0.006 |   13 | 0.005 |
|     7 |    65,025 |     18 | 0.022 |   15 | 0.029 |    8 | 0.035 |    9 | 0.036 |   13 | 0.029 |
|     8 |   261,121 |     19 | 0.129 |   15 | 0.155 |    8 | 0.234 |    9 | 0.250 |   13 | 0.217 |
|     9 | 1,046,529 |     18 | 0.653 |   15 | 0.921 |    8 | 1.233 |    9 | 1.240 |   13 | 1.127 |

### HyperElasticity 3D

| Level |       N | greedy |       |   JP |       |   SL |       |   ID |       |   LF |       |
| ----: | ------: | -----: | ----: | ---: | ----: | ---: | ----: | ---: | ----: | ---: | ----: |
|       |         |   Cols |  Time | Cols |  Time | Cols |  Time | Cols |  Time | Cols |  Time |
|     1 |   2,133 |     60 | 0.004 |   58 | 0.009 |   48 | 0.002 |   48 | 0.002 |   63 | 0.002 |
|     2 |  11,925 |     81 | 0.020 |   71 | 0.038 |   63 | 0.014 |   60 | 0.013 |   81 | 0.013 |
|     3 |  77,517 |     95 | 0.174 |   86 | 0.327 |   69 | 0.167 |   69 | 0.165 |   90 | 0.161 |
|     4 | 554,013 |     99 | 1.542 |   92 | 3.605 |   75 | 1.771 |   78 | 1.778 |   90 | 1.718 |

---

## 4. Speedup: serial (np = 1) vs parallel (np = 16)

Comparison of wall-clock times for selected methods on the largest mesh levels.

| Problem     | Level | Method       | np=1 (s) | np=16 (s) | Speedup |
| ----------- | ----: | ------------ | -------: | --------: | ------: |
| pLaplace 2D |     9 | PETSc greedy |    1.262 |     0.255 |    4.9× |
| pLaplace 2D |     9 | PETSc SL     |    1.159 |     0.674 |    1.7× |
| pLaplace 2D |     9 | PETSc ID     |    1.172 |     0.684 |    1.7× |
| GL 2D       |     9 | PETSc greedy |    1.881 |     0.653 |    2.9× |
| GL 2D       |     9 | PETSc SL     |    1.741 |     1.233 |    1.4× |
| GL 2D       |     9 | PETSc ID     |    1.718 |     1.240 |    1.4× |
| HE 3D       |     4 | PETSc greedy |    4.872 |     1.542 |    3.2× |
| HE 3D       |     4 | PETSc SL     |    2.715 |     1.771 |    1.5× |
| HE 3D       |     4 | PETSc ID     |    2.718 |     1.778 |    1.5× |

---

## Key observations

1. **PETSc SL and ID are the best PETSc options.** The smallest-last (`sl`)
   and incidence-degree (`id`) orderings produce dramatically fewer colors
   than the default `greedy` or `jp`:
   - **2D (pLaplace, GL):** 8–10 colors (SL/ID) vs 14–15 (greedy/jp)
   - **3D (HyperElasticity):** 48–78 (SL/ID) vs 58–99 (greedy/jp)
   - This reduces the number of Hessian–vector products by **~35–45 %** in 2D
     and **~15–20 %** in 3D.

2. **igraph gives the best color quality overall.** igraph produces 7–8 colors
   for 2D and 48–68 for 3D, consistently the fewest among practical methods.
   NX DSATUR occasionally matches or beats igraph by 1 color (e.g., 7 vs 8
   at pLaplace level 3) but is 100–10,000× slower.

3. **PETSc SL nearly matches igraph for structured 2D meshes.**
   On Ginzburg–Landau 2D, PETSc SL consistently produces exactly **8 colors**
   (igraph: 7–8), making it an excellent parallel-capable alternative.

4. **PETSc is the fastest backend in serial.** PETSc SL/ID are typically
   10–15 % faster than igraph in serial and **3.5× faster** on HE 3D level 4
   (2.7 s vs 9.5 s) because they avoid forming $A^2$ explicitly.

5. **Parallel scaling favors greedy.** PETSc `greedy` achieves the best
   speedup with 16 processes (3–5×), while SL/ID show more modest speedup
   (1.4–1.7×). However, for the largest problems (HE 3D level 4), parallel
   SL at 1.77 s is still **5.4× faster** than serial igraph at 9.5 s.

6. **NetworkX is impractical for production.** NX DSATUR gives excellent
   color quality but its $O(n^2)$ complexity makes it unusable for $N > 10{,}000$.
   NX `smallest_last` is better (~10–20× slower than PETSc) but still cannot
   compete with PETSc or igraph for large problems.

7. **PETSc LF is not recommended.** The largest-first ordering (`lf`)
   produces more colors than greedy in 3D (e.g., 90 vs 91 at HE level 4)
   and more than SL/ID everywhere. It offers no advantage.

8. **Practical recommendation.**
   - **Serial, Python-only:** igraph (optimal colors, fast).
   - **Serial, PETSc available:** PETSc ID or SL (near-optimal colors, fastest).
   - **Parallel (MPI):** PETSc SL for best color quality; PETSc greedy for
     fastest wall time when color count is less critical.
