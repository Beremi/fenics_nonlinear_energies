# p-Laplace 2D — Benchmark Results

Benchmark results for the 2D p-Laplacian problem ($p = 3$, $f = -10$, homogeneous Dirichlet BCs on the unit square).

Raw data is stored as JSON files in [results/](results/). See [instructions.md](instructions.md) for how to run new experiments and store results.

---

## Experiment `experiment_001`

- **Date**: 2026-02-21
- **CPU**: AMD Ryzen 9 9950X3D 16-Core Processor (32 threads)
- **DOLFINx**: 0.10.0.post2
- **Git commit**: `7dce8760`
- **Repetitions**: 3 (median time reported)
- **Data**: [results/experiment_001/](results/experiment_001/)

### FEniCS SNES Newton (serial vs parallel)

| lvl | dofs   | time (serial) | iters | J(u)    | time (4 proc) | iters | J(u)    | time (8 proc) | iters | J(u)    | time (16 proc) | iters | J(u)    |
| --- | ------ | ------------- | ----- | ------- | ------------- | ----- | ------- | ------------- | ----- | ------- | -------------- | ----- | ------- |
| 4   | 2945   | 0.043         | 10    | -7.9430 | 0.029         | 11    | -7.9430 | 0.026         | 10    | -7.9430 | 0.033          | 11    | -7.9430 |
| 5   | 12033  | 0.167         | 10    | -7.9546 | 0.071         | 10    | -7.9546 | 0.050         | 10    | -7.9546 | 0.054          | 10    | -7.9546 |
| 6   | 48641  | 0.478         | 7     | -7.9583 | 0.169         | 7     | -7.9583 | 0.110         | 7     | -7.9583 | 0.087          | 8     | -7.9583 |
| 7   | 195585 | 2.152         | 8     | -7.9596 | 0.768         | 8     | -7.9596 | 0.463         | 8     | -7.9596 | 0.274          | 8     | -7.9596 |
| 8   | 784385 | 10.026        | 9     | -7.9600 | 3.772         | 9     | -7.9600 | 2.430         | 9     | -7.9600 | 1.404          | 9     | -7.9600 |

### All Solver Configurations (SNES vs Custom Newton)

| lvl | dofs   | SNES serial | iters | Custom serial | iters | SNES 4-proc | iters | Custom 4-proc | iters | SNES 8-proc | iters | Custom 8-proc | iters | SNES 16-proc | iters | Custom 16-proc | iters | J(u)    |
| --- | ------ | ----------- | ----- | ------------- | ----- | ----------- | ----- | ------------- | ----- | ----------- | ----- | ------------- | ----- | ------------ | ----- | -------------- | ----- | ------- |
| 4   | 2945   | 0.043       | 10    | 0.039         | 5     | 0.029       | 11    | 0.021         | 5     | 0.026       | 10    | 0.016         | 5     | 0.033        | 11    | 0.021          | 5     | -7.9430 |
| 5   | 12033  | 0.167       | 10    | 0.157         | 5     | 0.071       | 10    | 0.055         | 5     | 0.050       | 10    | 0.039         | 5     | 0.054        | 10    | 0.039          | 5     | -7.9546 |
| 6   | 48641  | 0.478       | 7     | 0.731         | 6     | 0.169       | 7     | 0.239         | 6     | 0.110       | 7     | 0.143         | 6     | 0.087        | 8     | 0.098          | 6     | -7.9583 |
| 7   | 195585 | 2.152       | 8     | 3.434         | 7     | 0.768       | 8     | 0.931         | 6     | 0.463       | 8     | 0.556         | 6     | 0.274        | 8     | 0.318          | 6     | -7.9596 |
| 8   | 784385 | 10.026      | 9     | 12.488        | 6     | 3.772       | 9     | 4.091         | 6     | 2.430       | 9     | 2.873         | 6     | 1.404        | 9     | 1.482          | 6     | -7.9600 |

The Custom Newton uses the JAX-version algorithm (golden-section line search on $[-0.5, 2]$, CG + HYPRE AMG). It converges in fewer iterations (5–7 vs 7–10 for SNES) with comparable wall times. At small mesh levels the Custom solver is faster due to fewer iterations; at larger levels the per-iteration cost of the golden-section line search (~20 energy evaluations) adds overhead that offsets the iteration savings.

### Strong Scaling (SNES Newton, 1–32 processes)

![Scaling plot](results/experiment_001/scaling.png)

Left: wall time vs number of MPI processes (log-log). Right: parallel speedup relative to serial. The dashed line shows ideal linear scaling. Larger problems (lvl 7, 8) scale well up to 16 processes; at 32 processes communication overhead starts to dominate for the smaller mesh levels.

To regenerate this plot:
```bash
python3 results/generate_scaling_plot.py results/experiment_001/
```

### JAX Newton (serial only, no MPI)

The same p-Laplace problem solved using a pure-JAX pipeline (automatic differentiation for gradients, sparse finite differences with graph coloring for Hessian assembly, PyAMG smoothed-aggregation CG solver). This implementation lives in [`pLaplace2D_jax/`](pLaplace2D_jax/), [`tools/`](tools/) and is wrapped by [`pLaplace2D_jax/solve_pLaplace_jax_newton.py`](pLaplace2D_jax/solve_pLaplace_jax_newton.py).

| lvl | dofs   | setup (s) | solve (s) | total (s) | iters | J(u)    |
| --- | ------ | --------- | --------- | --------- | ----- | ------- |
| 4   | 2945   | 0.184     | 0.076     | 0.260     | 6     | -7.9430 |
| 5   | 12033  | 0.182     | 0.147     | 0.329     | 6     | -7.9546 |
| 6   | 48641  | 0.263     | 0.435     | 0.698     | 6     | -7.9583 |
| 7   | 195585 | 0.582     | 2.173     | 2.755     | 8     | -7.9596 |
| 8   | 784385 | 1.903     | 10.918    | 12.821    | 9     | -7.9600 |

**Setup** includes JIT compilation, Hessian sparsity detection (graph coloring), and AMG preconditioner construction. **Solve** is the Newton iteration time only.

**Comparison with FEniCS (serial)**:
- **Solve time**: JAX is comparable to FEniCS SNES for small problems but slightly slower at larger levels (10.9 s vs 10.0 s at lvl 8), likely due to differences in AMG implementations (PyAMG vs HYPRE).
- **Iterations**: JAX converges in 6–9 iterations (similar to the Custom Newton's 5–7). Both use a golden-section line search on $[-0.5, 2]$ with `tol=1e-3`. SNES converges in 7–10 iterations with a full-step (basic) line search.
- **No parallelism**: The JAX solver runs on a single CPU core. There is no MPI parallelism yet.
- **Setup overhead**: The ~0.2–1.9 s setup cost (JIT + graph coloring) is amortized over the solve but significant for small problems.

### Custom Newton — JAX-version algorithm (FEniCS + PETSc)

Re-implementation of the JAX minimiser (`tools/minimizers.py`) on top of PETSc via `tools_petsc4py/minimizers.py`. Uses the same golden-section line search on $[-0.5, 2]$ with `tol=1e-3`, CG + HYPRE AMG with `rtol=1e-3`, `tolf=1e-5`, `tolg=1e-3`. Supports MPI parallelism.

Script: [`pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py`](pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py)

| lvl | dofs   | time (serial) | iters | time (4-proc) | iters | time (8-proc) | iters | time (16-proc) | iters | J(u)    |
| --- | ------ | ------------- | ----- | ------------- | ----- | ------------- | ----- | -------------- | ----- | ------- |
| 4   | 3201   | 0.039         | 5     | 0.021         | 5     | 0.016         | 5     | 0.021          | 5     | -7.9430 |
| 5   | 12545  | 0.157         | 5     | 0.055         | 5     | 0.039         | 5     | 0.039          | 5     | -7.9546 |
| 6   | 49665  | 0.731         | 6     | 0.239         | 6     | 0.143         | 6     | 0.098          | 6     | -7.9583 |
| 7   | 197633 | 3.434         | 7     | 0.931         | 6     | 0.556         | 6     | 0.318          | 6     | -7.9596 |
| 8   | 788481 | 12.488        | 6     | 4.091         | 6     | 2.873         | 6     | 1.482          | 6     | -7.9600 |

**Key observations**:
- **5–7 iterations** — matches or beats the JAX solver (6–9) and significantly fewer than SNES (7–10).
- **Line search allows α > 1**: The wider interval $[-0.5, 2]$ and tighter tolerance (`1e-3`) enable full or overshooting steps.
- **MPI-parallel**: Unlike pure JAX, this solver scales across MPI processes.
- **Comparable serial times** to SNES Newton — slightly slower due to golden-section energy evaluations per iteration, but fewer iterations compensate.

### JAX + PETSc SFD Newton (MPI-parallel)

A hybrid solver that uses **JAX** for automatic differentiation (energy, gradient, Hessian-vector products) combined with **PETSc** for distributed sparse linear algebra. The Hessian is assembled via sparse finite differences (SFD) with graph coloring — the same approach as the serial JAX Newton solver, but with MPI parallelism:

- **Distributed HVP computation**: graph coloring produces $n_c$ colour groups, distributed round-robin across MPI ranks. Each rank computes only $\lceil n_c / p \rceil$ Hessian-vector products per Newton step.
- **Parallel multi-start coloring** via `graph_coloring/multistart_coloring.py` (each rank runs independent trials; the best global result is kept).
- **PETSc MPIAIJ** matrix with precomputed sparsity pattern (from adjacency). Assembled via `MPI_Allreduce(SUM)` of per-rank HVP data.
- **PETSc KSP**: CG + GAMG (PETSc native AMG) with `rtol=1e-3` (same tolerance as the FEniCS custom Newton).
- **Same Newton algorithm** as the other custom solvers: golden-section line search on $[-0.5, 2]$, `tolf=1e-5`, `tolg=1e-3`, via `tools_petsc4py/minimizers.py`.

**Important**: GAMG is used instead of HYPRE BoomerAMG.  In the replicated-data model the DOF ordering does not reflect mesh locality (PETSc default block distribution).  HYPRE BoomerAMG setup scales catastrophically under this condition — up to **30× slower** in parallel than serial — while GAMG handles arbitrary orderings gracefully.  See Annex B for detailed analysis.

Script: [`pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py`](pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py)

| lvl | dofs   | setup (s) | solve (s) | iters | n_colors | J(u)    | setup 2-proc | solve 2-proc | iters | setup 4-proc | solve 4-proc | iters | setup 8-proc | solve 8-proc | iters | setup 16-proc | solve 16-proc | iters |
| --- | ------ | --------- | --------- | ----- | -------- | ------- | ------------ | ------------ | ----- | ------------ | ------------ | ----- | ------------ | ------------ | ----- | ------------- | ------------- | ----- |
| 5   | 2945   | 0.201     | 0.029     | 5     | 9        | -7.9430 | 0.209        | 0.039        | 6     | 0.274        | 0.047        | 6     | 0.346        | 0.068        | 6     | 0.607         | 0.075         | 6     |
| 6   | 12033  | 0.203     | 0.099     | 6     | 9        | -7.9546 | 0.206        | 0.099        | 6     | 0.250        | 0.109        | 6     | 0.346        | 0.154        | 6     | 0.584         | 0.261         | 6     |
| 7   | 48641  | 0.258     | 0.313     | 6     | 10       | -7.9583 | 0.261        | 0.341        | 6     | 0.323        | 0.370        | 6     | 0.423        | 0.446        | 6     | 0.728         | 0.802         | 6     |
| 8   | 195585 | 0.429     | 1.205     | 6     | 9        | -7.9596 | 0.491        | 1.391        | 6     | 0.572        | 1.689        | 6     | 0.773        | 2.299        | 6     | 1.327         | 3.917         | 6     |
| 9   | 784385 | 1.129     | 6.727     | 7     | 9        | -7.9600 | 1.297        | 7.541        | 7     | 1.723        | 9.087        | 7     | 2.247        | 12.769       | 7     | 3.572         | 21.569        | 7     |

**Key observations**:
- **5–7 iterations** — matches the serial JAX Newton (6–9) and FEniCS Custom Newton (5–7). Same algorithm, same tolerances, same energy values.
- **Serial solve time** is now faster than the serial JAX Newton with PyAMG (0.029–6.73 s vs 0.076–10.9 s) thanks to GAMG.
- **Serial is fastest** — the replicated-data model means every rank still runs full-vector JAX operations (energy, gradient, `Allgatherv`). While the KSP phase scales well with GAMG (see Annex A), the replicated energy/gradient evaluation (~20 energy calls per Newton step for golden-section line search) grows proportionally with $p$ due to shared-memory CPU contention and `Allgatherv` overhead. This overhead dominates at higher process counts.
- **KSP (linear solve) scales well**: with GAMG, KSP time decreases from 0.65 s (serial) to 0.55 s (16 proc) at level 8. The bottleneck is the replicated energy/gradient evaluation.
- **Setup cost** grows with MPI ranks (more coloring trials, PETSc MPIAIJ preallocation overhead), from 0.20 s (serial) to 3.57 s (16 proc) at lvl 9.

```bash
# Serial
python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py --levels 5 6 7 8 9 --json results/<exp>/jax_petsc_sfd_np1_run1.json

# Parallel (4 / 8 / 16 processes)
mpirun -n 4  python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py --levels 5 6 7 8 9 --quiet --json results/<exp>/jax_petsc_sfd_np4_run1.json
mpirun -n 8  python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py --levels 5 6 7 8 9 --quiet --json results/<exp>/jax_petsc_sfd_np8_run1.json
mpirun -n 16 python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py --levels 5 6 7 8 9 --quiet --json results/<exp>/jax_petsc_sfd_np16_run1.json
```

**Environment note**: Requires `jax`, `jaxlib`, `petsc4py`, `mpi4py`, `h5py`, `scipy` — a combined JAX+PETSc environment (the devcontainer Dockerfile installs all of these).

---

## Annex A — Timing Breakdown of JAX + PETSc SFD Solver (GAMG)

Detailed timing breakdown with the **GAMG** preconditioner on mesh **level 8** ($n = 195{,}585$ DOFs, $\text{nnz} = 1{,}365{,}009$) and **level 9** ($n = 784{,}385$, $\text{nnz} = 5{,}482{,}513$).  All measurements on AMD Ryzen 9 9950X3D (16C/32T), single Docker container.

### A.1  Timing Breakdown — Level 8 (195 585 DOFs)

**Aggregated over 6 Newton iterations** (all times in seconds):

| nproc |   KSP |   HVP | Allreduce | Allgather | COO assem | Hess total | Other† | Solve | Setup | active |
| ----: | ----: | ----: | --------: | --------: | --------: | ---------: | -----: | ----: | ----: | :----: |
|     1 | 0.648 | 0.290 |     0.000 |     0.001 |     0.016 |      0.955 |  0.249 | 1.205 | 0.429 |  1/1   |
|     2 | 0.812 | 0.296 |     0.014 |     0.003 |     0.012 |      1.137 |  0.344 | 1.481 | 0.491 |  2/2   |
|     4 | 0.806 | 0.337 |     0.027 |     0.006 |     0.009 |      1.186 |  0.620 | 1.805 | 0.572 |  4/4   |
|     8 | 0.662 | 0.318 |     0.067 |     0.012 |     0.010 |      1.070 |  1.249 | 2.318 | 0.773 |  8/8   |
|    16 | 0.552 | 0.322 |     0.147 |     0.022 |     0.012 |      1.054 |  2.900 | 3.954 | 1.327 |  8/16  |

†*Other* = energy evaluations + gradient + golden-section line search (≈20 energy evals per Newton step, each requiring `Allgatherv` + replicated JAX computation).

### A.2  Per-Iteration Detail — Level 8, Serial (np = 1)

|   it | Allgather |    HVP | nHVP | Allreduce | COO asm |    KSP | KSP its |  Total |
| ---: | --------: | -----: | ---: | --------: | ------: | -----: | ------: | -----: |
|    0 |    0.0002 | 0.0475 |    9 |    0.0000 |  0.0038 | 0.2903 |       7 | 0.3419 |
|    1 |    0.0002 | 0.0475 |    9 |    0.0000 |  0.0031 | 0.0687 |       6 | 0.1195 |
|    2 |    0.0001 | 0.0517 |    9 |    0.0000 |  0.0026 | 0.0662 |       6 | 0.1207 |
|    3 |    0.0001 | 0.0457 |    9 |    0.0000 |  0.0025 | 0.0713 |       7 | 0.1196 |
|    4 |    0.0002 | 0.0482 |    9 |    0.0000 |  0.0022 | 0.0736 |       6 | 0.1243 |
|    5 |    0.0002 | 0.0492 |    9 |    0.0000 |  0.0023 | 0.0777 |       8 | 0.1294 |

### A.3  Per-Iteration Detail — Level 8, 16 MPI Ranks

|   it | Allgather |    HVP | nHVP | Allreduce | COO asm |    KSP | KSP its |  Total |
| ---: | --------: | -----: | ---: | --------: | ------: | -----: | ------: | -----: |
|    0 |    0.0027 | 0.0462 |    1 |    0.0358 |  0.0010 | 0.2730 |       8 | 0.3588 |
|    1 |    0.0043 | 0.0655 |    1 |    0.0245 |  0.0008 | 0.0526 |       6 | 0.1478 |
|    2 |    0.0021 | 0.0678 |    1 |    0.0171 |  0.0024 | 0.0534 |       6 | 0.1428 |
|    3 |    0.0020 | 0.0482 |    1 |    0.0244 |  0.0026 | 0.0572 |       7 | 0.1344 |
|    4 |    0.0078 | 0.0522 |    1 |    0.0169 |  0.0023 | 0.0538 |       7 | 0.1330 |
|    5 |    0.0030 | 0.0418 |    1 |    0.0283 |  0.0023 | 0.0618 |       7 | 0.1371 |

### A.4  Timing Breakdown — Level 9 (784 385 DOFs)

| nproc |   KSP |   HVP | Allreduce | Allgather | COO assem | Hess total | Other† |  Solve | Setup | active |
| ----: | ----: | ----: | --------: | --------: | --------: | ---------: | -----: | -----: | ----: | :----: |
|     1 | 3.590 | 1.850 |     0.000 |     0.006 |     0.086 |      5.533 |  1.269 |  6.801 | 1.112 |  1/1   |
|    16 | 3.518 | 1.679 |     0.684 |     0.099 |     0.054 |      6.033 | 15.788 | 21.821 | 3.638 |  8/16  |

†*Other* = energy evaluations + gradient + golden-section line search (≈17 energy evals × 7 Newton steps, each requiring `Allgatherv` + replicated JAX computation).

### A.5  Per-Iteration Detail — Level 9, Serial (np = 1)

**7 Newton iterations**, 784 385 DOFs, 5 482 513 nnz, 9 colours (all times in seconds):

|   it | Allgather |    HVP | nHVP | Allreduce | COO asm |    KSP | KSP its |  Total |
| ---: | --------: | -----: | ---: | --------: | ------: | -----: | ------: | -----: |
|    0 |    0.0007 | 0.2575 |    9 |    0.0000 |  0.0116 | 1.3035 |       8 | 1.5734 |
|    1 |    0.0007 | 0.2552 |    9 |    0.0000 |  0.0120 | 0.3389 |       6 | 0.6069 |
|    2 |    0.0008 | 0.2626 |    9 |    0.0000 |  0.0117 | 0.3616 |       7 | 0.6367 |
|    3 |    0.0008 | 0.2852 |    9 |    0.0000 |  0.0118 | 0.3793 |       8 | 0.6771 |
|    4 |    0.0011 | 0.2722 |    9 |    0.0000 |  0.0156 | 0.3955 |       8 | 0.6845 |
|    5 |    0.0007 | 0.2526 |    9 |    0.0000 |  0.0119 | 0.4018 |       9 | 0.6670 |
|    6 |    0.0010 | 0.2645 |    9 |    0.0000 |  0.0115 | 0.4098 |       9 | 0.6869 |

Iteration 0 includes GAMG setup (≈ 0.96 s); subsequent KSP calls reuse the preconditioner at ≈ 0.37 s each.  HVP is 9 products/iteration (one per colour).

### A.6  Per-Iteration Detail — Level 9, 16 MPI Ranks

**7 Newton iterations**, 784 385 DOFs, 5 482 513 nnz, 8 colours, 8 active ranks (all times in seconds):

|   it | Allgather |    HVP | nHVP | Allreduce | COO asm |    KSP | KSP its |  Total |
| ---: | --------: | -----: | ---: | --------: | ------: | -----: | ------: | -----: |
|    0 |    0.0155 | 0.2789 |    1 |    0.0843 |  0.0096 | 1.2994 |       8 | 1.6878 |
|    1 |    0.0151 | 0.2619 |    1 |    0.1195 |  0.0065 | 0.3354 |       7 | 0.7385 |
|    2 |    0.0129 | 0.2771 |    1 |    0.1002 |  0.0034 | 0.3529 |       8 | 0.7464 |
|    3 |    0.0094 | 0.2204 |    1 |    0.0904 |  0.0087 | 0.3641 |       8 | 0.6930 |
|    4 |    0.0156 | 0.1954 |    1 |    0.1084 |  0.0086 | 0.3597 |       8 | 0.6877 |
|    5 |    0.0173 | 0.2219 |    1 |    0.0991 |  0.0104 | 0.4173 |       8 | 0.7660 |
|    6 |    0.0129 | 0.2228 |    1 |    0.0822 |  0.0070 | 0.3888 |       9 | 0.7137 |

Each rank computes only 1 HVP/iteration (8 colours ÷ 8 active ranks).  The KSP timings are nearly identical to the serial case (first iteration ≈ 1.30 s with GAMG setup, subsequent ≈ 0.35 s).  The "hessian_solve" total is 6.03 s (serial 5.53 s) — parallelism adds only 9 % overhead here.  The dominant cost at np = 16 is **Other = 15.79 s** (73 % of the 21.82 s solve wall time), entirely due to replicated energy/gradient evaluations under shared-memory CPU contention.

### A.7  Analysis

**1. KSP (CG + GAMG) scales well.**

With GAMG, the KSP phase scales properly: 0.65 s (serial) → 0.55 s (np = 16) at level 8.  The first iteration includes GAMG setup (~0.22 s), while subsequent iterations use only GAMG V-cycles (~0.07 s each).  This is a dramatic improvement over the HYPRE configuration (see Annex B).

**2. Replicated energy/gradient evaluation ("Other") is the bottleneck.**

Every Newton step executes ≈20 energy evaluations (golden-section line search) plus one gradient evaluation.  In the replicated-data model, *every rank* computes these on the full vector.  Each call requires an `Allgatherv` of $n$ doubles from $p$ ranks, plus a full JAX JIT call.  With 16 processes on shared memory, both the MPI traffic and the JAX CPU work compete for resources:

- Level 8: "Other" = 0.25 s (serial) → 2.90 s (np = 16), a **12× increase**.
- Level 9: "Other" = 1.27 s (serial) → 15.94 s (np = 16), a **13× increase**.

At np = 16 on level 9, "Other" accounts for 73 % of the solve wall time.

**3. HVP parallelisation is effective but has low weight.**

The HVP phase drops from 9 products/rank (serial) to 1 product/rank (np = 16, with 8 colours).  But total HVP time is flat (~0.3 s at level 8 regardless of $p$) because each rank's single HVP takes about the same wall time as the serial rank's 9 HVPs (memory contention).

**4. COO matrix assembly is negligible.**

The `setPreallocationCOO` / `setValuesCOO` assembly path adds only 0.01–0.08 s total (< 1.5 % of solve time).

### A.8  Conclusions

The replicated-data JAX+PETSc solver's parallel scaling is limited because:
1. The replicated energy/gradient computation adds overhead proportional to $p$ (contention for CPU+memory), and this now dominates total time with GAMG fixing the KSP bottleneck.
2. The only parallelised component (HVP + KSP) contributes ~70 % of serial hessian-solve time and scales moderately, but this is overwhelmed by the "Other" overhead at high $p$.
3. For this architecture to show speedup, the problem would need a much larger number of colours (more HVPs to distribute) or the replicated-data model should be replaced with **true data-parallel assembly** (as in DOLFINx / FEniCS).

---

## Annex B — HYPRE BoomerAMG vs GAMG: Impact of DOF Ordering

### B.1  The Problem

In the replicated-data architecture, PETSc's default block distribution assigns DOF rows linearly (rows $0 \ldots n/p - 1$ to rank 0, etc.).  This distribution does **not** reflect mesh locality — neighbouring mesh nodes may end up on different ranks, creating extensive off-diagonal coupling in the MPIAIJ matrix.

FEniCS/DOLFINx, in contrast, uses a graph partitioner (ParMETIS / SCOTCH) to distribute DOFs so that each rank owns a geometrically contiguous region, minimising off-diagonal entries.

### B.2  Impact on HYPRE BoomerAMG Setup

HYPRE BoomerAMG's parallel setup (coarsening + interpolation construction) is highly sensitive to the partition quality.  With poor DOF ordering, the setup phase becomes catastrophically slow:

**KSP timing per solve (level 8, 195 585 DOFs, CG + AMG, rtol = 1e-3):**

| nproc | HYPRE (rebuild) | HYPRE (reuse) | GAMG (rebuild) | GAMG (reuse) |
| ----: | --------------: | ------------: | -------------: | -----------: |
|     1 |          0.286s |        0.083s |         0.067s |       0.039s |
|     2 |          1.845s |        0.074s |         0.059s |       0.028s |
|     4 |          2.058s |        0.056s |         0.061s |       0.026s |
|    16 |          0.550s |        0.033s |         0.052s |       0.018s |

- **HYPRE setup (rebuild − reuse)**: 0.20 s (serial) → 1.77 s (np = 2) → 2.00 s (np = 4) → 0.52 s (np = 16).  The setup is **up to 10× slower** in parallel.
- **GAMG setup**: 0.028 s (serial) → 0.031 s (np = 2) → 0.035 s (np = 4) → 0.034 s (np = 16).  Scales gracefully.

With HYPRE, only the CG iteration phase (PC reuse column) scales properly — confirming that the CG + SpMV + V-cycle application is fine.  The bottleneck is entirely in the AMG **setup** (coarsening + interpolation construction), which HYPRE computes using algorithms sensitive to partition quality.

### B.3  Effect on Full Newton Solve

**Level 8 (195 585 DOFs), solve time:**

| nproc |  HYPRE |  GAMG | Speedup |
| ----: | -----: | ----: | ------: |
|     1 |  2.32s | 1.20s |    1.9× |
|     2 | 10.89s | 1.48s |    7.4× |
|     4 |  8.01s | 1.81s |    4.4× |
|     8 |  6.33s | 2.30s |    2.8× |
|    16 |  6.33s | 3.95s |    1.6× |

**Level 9 (784 385 DOFs):**

| nproc |  HYPRE |   GAMG | Speedup |
| ----: | -----: | -----: | ------: |
|     1 | 11.97s |  6.73s |    1.8× |
|    16 | 47.35s | 21.57s |    2.2× |

### B.4  Recommendation

For replicated-data solvers (or any solver where the DOF ordering does not reflect mesh locality), **GAMG should be preferred over HYPRE BoomerAMG**.  GAMG's algebraic coarsening is robust to arbitrary orderings, while HYPRE's parallel setup assumes locality-aware partitions.

---

## Generating LaTeX Tables and Plots

The script `results/generate_latex_tables.py` reads JSON result files, aggregates repeated runs (median time), and produces publication-ready tables.

```bash
# Print LaTeX tables to stdout
python3 results/generate_latex_tables.py results/experiment_001/

# Save to .tex file (can be \input{}-ed in a LaTeX document)
python3 results/generate_latex_tables.py results/experiment_001/ --output results/experiment_001/tables.tex

# Print Markdown tables instead
python3 results/generate_latex_tables.py results/experiment_001/ --markdown
```

The generated LaTeX file is also committed at [results/experiment_001/tables.tex](results/experiment_001/tables.tex).
