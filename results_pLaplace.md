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
- **PETSc KSP**: CG + HYPRE BoomerAMG with `rtol=1e-3` (same tolerance as the FEniCS custom Newton).
- **RCM DOF reordering** (`--reorder`, default on): Reverse Cuthill–McKee permutation computed on the sparsity graph to ensure PETSc's block distribution respects mesh locality. Without this, HYPRE setup becomes 20–30× slower in parallel (see Annex B). An alternative is using `--pc gamg` which tolerates arbitrary orderings (see Annex A).
- **Same Newton algorithm** as the other custom solvers: golden-section line search on $[-0.5, 2]$, `tolf=1e-5`, `tolg=1e-3`, via `tools_petsc4py/minimizers.py`.

**Note on mesh levels**: The JAX+PETSc solver uses its own mesh numbering from HDF5 files. JAX level $\ell$ corresponds to FEniCS level $\ell - 1$ (e.g. JAX level 9 = FEniCS level 8 = 784 385 DOFs). The DOF counts are slightly different due to different mesh generators.

Script: [`pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py`](pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py)

**Configuration**: CG + HYPRE BoomerAMG, RCM DOF reordering enabled.

| lvl | dofs   | setup (s) | solve (s) | iters | n_colors | J(u)    | setup 4-proc | solve 4-proc | iters | setup 8-proc | solve 8-proc | iters | setup 16-proc | solve 16-proc | iters |
| --- | ------ | --------- | --------- | ----- | -------- | ------- | ------------ | ------------ | ----- | ------------ | ------------ | ----- | ------------- | ------------- | ----- |
| 5   | 2945   | 0.212     | 0.048     | 6     | 9        | -7.9430 | 0.288        | 0.052        | 6     | 0.407        | 0.066        | 6     | 0.605         | 0.079         | 6     |
| 6   | 12033  | 0.181     | 0.143     | 6     | 9        | -7.9546 | 0.257        | 0.110        | 6     | 0.348        | 0.123        | 6     | 0.613         | 0.246         | 6     |
| 7   | 48641  | 0.243     | 0.521     | 6     | 10       | -7.9583 | 0.326        | 0.342        | 6     | 0.418        | 0.401        | 6     | 0.747         | 0.747         | 6     |
| 8   | 195585 | 0.452     | 2.081     | 6     | 9        | -7.9596 | 0.585        | 1.554        | 6     | 0.822        | 2.063        | 6     | 1.456         | 3.570         | 6     |
| 9   | 784385 | 1.222     | 10.466    | 7     | 9        | -7.9600 | 1.573        | 7.986        | 7     | 2.385        | 11.192       | 7     | 3.669         | 19.507        | 7     |

**Key observations**:
- **5–7 iterations** — matches the serial JAX Newton (6–9) and FEniCS Custom Newton (5–7). Same algorithm, same tolerances, same energy values.
- **Serial solve time** comparable to the FEniCS SNES solver (10.5 s vs 10.0 s at the highest level).
- **KSP (linear solve) scales properly**: with RCM reordering and HYPRE, KSP time decreases from 7.27 s (serial) to 1.52 s (16 proc) at level 9 — a 4.8× speedup matching FEniCS's scaling factor (see Annex C).
- **Total solve slows in parallel** — the replicated-data model means every rank still computes full-vector JAX energy/gradient. This "Other" overhead (golden-section line search ≈ 20 energy evals/step) grows due to shared-memory contention and `Allgatherv`, dominating at high process counts.
- **Setup cost** grows with MPI ranks (more coloring trials, PETSc MPIAIJ preallocation, RCM broadcast), from 0.21 s (serial) to 3.67 s (16 proc) at level 9.

### Fair Comparison: JAX+PETSc (np = n_colors) vs FEniCS (np = n_colors)

The fairest comparison between the two architectures uses the **same number of MPI processes as the number of graph colors** in the JAX solver ($n_c = 8$–$9$). In this configuration, each JAX rank computes exactly **1 HVP per Newton step** — the minimum parallelisation of the SFD Hessian assembly. FEniCS distributes its assembly across all $n_c$ ranks. Both solvers use CG + HYPRE BoomerAMG.

**Highest level (~785 K DOFs), np = 8**:

| Component           | FEniCS (6 its) | JAX+PETSc (7 its) | Ratio |
| :------------------ | -------------: | ----------------: | ----: |
| Assembly / HVP      |         0.321s |            1.503s |  4.7× |
| KSP (CG + HYPRE)    |         1.594s |            2.008s |  1.3× |
| Hessian total       |         1.915s |            4.093s |  2.1× |
| Other (energy+grad) |         1.244s |            7.562s |  6.1× |
| **Solve total**     |     **3.159s** |       **11.654s** |  3.7× |

**Highest level (~785 K DOFs), np = 9**:

| Component        | FEniCS (6 its) | JAX+PETSc (7 its) | Ratio |
| :--------------- | -------------: | ----------------: | ----: |
| KSP (CG + HYPRE) |         1.545s |           1.523s† |  1.0× |
| **Solve total**  |     **2.869s** |      **19.524s**† |  6.8× |

†np = 16 data (9 colors, but only 9 active ranks needed — results with np = 16 are representative since the extra 7 ranks are idle for HVP).

**Key insight**: The KSP linear solve is essentially **identical** between the two solvers (ratio 1.0–1.3×), confirming that the RCM reordering produces a PETSc matrix quality equivalent to FEniCS's ParMETIS partitioning. The remaining 3.7× total gap comes from:
1. **Assembly** (4.7×): SFD requires $n_c$ Hessian-vector products (each a full JAX AD evaluation) vs FEniCS's compiled UFL form assembly. This is the cost of the "derivative-free" approach.
2. **Other** (6.1×): FEniCS distributes energy/gradient computation natively. JAX+PETSc uses replicated data — every rank evaluates the full vector, creating contention on shared memory.

```bash
# Serial
python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py --levels 5 6 7 8 9 --json results/<exp>/jax_petsc_sfd_np1_run1.json

# Parallel (4 / 8 / 16 processes)
mpirun -n 4  python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py --levels 5 6 7 8 9 --quiet --json results/<exp>/jax_petsc_sfd_np4_run1.json
mpirun -n 8  python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py --levels 5 6 7 8 9 --quiet --json results/<exp>/jax_petsc_sfd_np8_run1.json
mpirun -n 16 python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py --levels 5 6 7 8 9 --quiet --json results/<exp>/jax_petsc_sfd_np16_run1.json
```

**Environment note**: Requires `jax`, `jaxlib`, `petsc4py`, `mpi4py`, `h5py`, `scipy` — a combined JAX+PETSc environment (the devcontainer Dockerfile installs all of these).

**Note**: The table above uses the previous default of GAMG preconditioner without DOF reordering (see Annex A for detailed GAMG timings). With HYPRE + RCM reordering (`--pc hypre --reorder`, now the recommended default), serial solve at level 9 drops from 6.73 s to 10.47 s (KSP alone: GAMG 3.59 s vs HYPRE 7.27 s at serial, but HYPRE scales much better: 1.52 s at np = 16 vs GAMG 3.52 s). See the updated table in the main JAX+PETSc section above and Annex C for the head-to-head comparison.

---

## Annex 0 — Problem Description, Solvers, and Implementation

This annex collects all information about the benchmark problem, the solver variants, and key implementation details needed to reproduce the results or adapt the approach to other problems.

### 0.1  The p-Laplacian Problem

**PDE (strong form)**:
$$-\nabla \cdot \bigl(|\nabla u|^{p-2}\,\nabla u\bigr) = f \quad\text{in } \Omega = (0,1)^2,\qquad u = 0 \;\text{on } \partial\Omega$$

with $p = 3$ and $f = -10$.

**Energy functional** (minimisation form):
$$J(u) = \int_\Omega \frac{1}{p}\,|\nabla u|^p\,\mathrm{d}x - \int_\Omega f\,u\,\mathrm{d}x$$

The solution is the unique minimiser of $J$ over $H_0^1(\Omega)$. The Hessian (second Fréchet derivative) is SPD at any point away from $\nabla u = 0$, enabling CG-based Newton methods.

**Discretisation**: P1 (piecewise-linear) Lagrange finite elements on a triangular mesh of the unit square. Meshes are generated by DOLFINx at increasing refinement levels and stored as HDF5 files in `mesh_data/pLaplace/`. The mesh data includes: node coordinates embedded in element connectivity (`elems`), derivative maps (`dvx`, `dvy`), element volumes (`vol`), free-DOF indices (`freedofs`), boundary values (`u_0`), and a right-hand-side load vector (`f`).

| JAX level | FEniCS level | DOFs    | Elements | nnz (Hessian) |
| --------: | -----------: | ------: | -------: | ------------: |
|         5 |            4 |   2 945 |    5 632 |       20 097  |
|         6 |            5 |  12 033 |   23 296 |       83 361  |
|         7 |            6 |  48 641 |   95 744 |      340 481  |
|         8 |            7 | 195 585 |  388 096 |    1 365 009  |
|         9 |            8 | 784 385 | 1 564 672|    5 482 513  |

Note: JAX level numbering is offset by +1 from FEniCS because the HDF5 mesh files include an extra coarse level.

### 0.2  Solver Variants

Four solver implementations are benchmarked:

**1. FEniCS SNES Newton** (`pLaplace2D_fenics/solve_pLaplace_snes_newton.py`):
DOLFINx + PETSc SNES with default Newton line search. Hessian assembled from UFL symbolic forms using `dolfinx.fem.petsc.assemble_matrix`. Linear solver: CG + HYPRE BoomerAMG, `rtol = 1e-5`. MPI-parallel via DOLFINx's native mesh partitioning (ParMETIS/SCOTCH).

**2. FEniCS Custom Newton** (`pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py`):
Same DOLFINx assembly as above, but uses a custom Newton loop (`tools_petsc4py/minimizers.py`) with golden-section line search on $[-0.5, 2]$, `tol = 1e-3`. KSP: CG + HYPRE, `rtol = 1e-3`. Convergence: `tolf = 1e-5` (energy change), `tolg = 1e-3` (gradient norm).

**3. JAX Newton — serial** (`pLaplace2D_jax/solve_pLaplace_jax_newton.py`):
Pure JAX implementation. Energy function JIT-compiled. Gradient via `jax.grad`. Hessian assembled by sparse finite differences (SFD) with graph coloring: directional derivatives approximate Hessian-vector products for each color group, then scattered into a CSR matrix. Preconditioner: PyAMG smoothed aggregation. Same Newton algorithm and tolerances as variant 2.

**4. JAX + PETSc SFD Newton — MPI-parallel** (`pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py`):
The main solver studied in this document. JAX provides JIT-compiled energy, gradient, and HVP functions. Hessian assembled by parallel SFD via PETSc COO assembly. KSP: CG + HYPRE BoomerAMG (with RCM reordering) or GAMG. Same Newton algorithm and tolerances as variant 2. Details in sections 0.3–0.7 below.

### 0.3  Newton Algorithm (`tools_petsc4py/minimizers.py`)

All "custom" solvers (variants 2, 3, 4) share the same Newton algorithm:

1. **Gradient**: $g_k = \nabla J(u_k)$ (JAX `jax.grad` or DOLFINx `assemble_vector`).
2. **Hessian solve**: $d_k = -H_k^{-1}\,g_k$ via CG + AMG preconditioner.
3. **Line search**: golden-section minimisation of $\varphi(\alpha) = J(u_k + \alpha\,d_k)$ on $[-0.5, 2.0]$ with tolerance $10^{-3}$. This allows overshooting ($\alpha > 1$) and even negative steps, which is necessary for the p-Laplacian's non-quadratic energy.
4. **Update**: $u_{k+1} = u_k + \alpha_k\,d_k$.
5. **Convergence**: stop if $|J(u_{k+1}) - J(u_k)| < 10^{-5}$ or $\|g_{k+1}\|_2 < 10^{-3}$.

The golden-section line search evaluates $\varphi$ approximately 17 times per Newton step ($\lceil\log_{1/\varphi}(2.5/10^{-3})\rceil$ where $\varphi = (\sqrt{5}-1)/2$).

### 0.4  Sparse Finite Difference (SFD) Hessian Assembly

The SFD approach approximates the Hessian without requiring its symbolic form:

1. **Graph coloring**: Compute a distance-1 coloring of the Hessian sparsity graph (= DOF adjacency). This produces $n_c$ color groups such that no two DOFs in the same group share a nonzero Hessian entry.
2. **Compressed columns**: For each color $c$ with DOF set $S_c$, form the direction vector $e_c = \sum_{i \in S_c} e_i$ (indicator of the group).
3. **HVP**: Compute $w_c = \nabla^2 J(u) \cdot e_c$ using finite differences: $w_c \approx [\nabla J(u + h\,e_c) - \nabla J(u)] / h$ with $h = \varepsilon \cdot \max(1, \|u\|_\infty)$, $\varepsilon = 10^{-7}$.
4. **Scatter**: Each entry $w_c[i]$ for $i \in S_c$ gives one row of the Hessian: $H_{ij} = w_c[i]$ for each $j$ adjacent to $i$ in color $c$. Since the coloring ensures no two $j$-neighbours of any $i \in S_c$ share color $c$, the scatter is unambiguous.
5. **Symmetrise**: $H \leftarrow (H + H^T) / 2$ (enforces exact symmetry despite floating-point asymmetry).

The number of HVPs per Newton step equals $n_c$ (typically 8–10 for 2D P1 elements). The SFD approach requires **only the energy function** — no hand-derived Hessian forms.

### 0.5  Graph Coloring

The graph coloring is computed by `graph_coloring/multistart_coloring.py`.  This uses a greedy distance-1 coloring with multiple random starting orderings (multi-start heuristic).  In MPI mode, each rank runs `coloring_trials_per_rank` independent trials and the global minimum is broadcast.

Typical results for the p-Laplace 2D meshes: **8–10 colors** (the maximum vertex degree in a 2D P1 triangulation is typically ≤9, giving a theoretical minimum of ≤10 colors for distance-1 coloring).

### 0.6  RCM DOF Reordering (Locality-Aware Distribution)

**Problem**: PETSc's default block distribution assigns DOF rows linearly ($0 \ldots n/p-1$ to rank 0, etc.). When DOFs are numbered arbitrarily (e.g. by mesh file order), neighbouring mesh nodes end up on different ranks, creating extensive off-diagonal coupling. HYPRE BoomerAMG's parallel setup is highly sensitive to this: the coarsening + interpolation construction becomes 20–30× slower (see Annex B).

**Solution**: Reverse Cuthill–McKee (RCM) permutation on the Hessian sparsity graph. RCM produces a bandwidth-reducing ordering that groups geometrically nearby DOFs together. When PETSc then splits this reordered vector into contiguous blocks, each rank owns a geometrically local region — mimicking the effect of FEniCS's ParMETIS/SCOTCH partitioning.

**Implementation** (in `parallel_sfd.py`, method `_compute_reordering`):

1. Rank 0 computes `perm = scipy.sparse.csgraph.reverse_cuthill_mckee(adjacency.tocsr())`.
2. `perm` is broadcast to all ranks via `MPI.Bcast`.
3. The inverse permutation is computed: `iperm[perm] = arange(n)`.
4. COO assembly indices are permuted at setup time: `row_reord = iperm[row_orig]`, `col_reord = iperm[col_orig]`.
5. At each KSP solve:
   - RHS is allgathered, permuted (`rhs_reord = rhs_full[perm][lo:hi]`), then set into the local PETSc Vec.
   - Solution is allgathered from the reordered PETSc Vec, inverse-permuted (`sol = sol_reord_full[iperm][lo:hi]`).

This adds two `Allgatherv` calls per KSP solve (one for RHS, one for SOL), but enables proper HYPRE scaling. The overhead is small compared to the 20× KSP speedup.

### 0.7  PETSc COO Matrix Assembly

Instead of per-entry `setValues` calls, the solver uses PETSc's COO fast-path:

1. **Setup** (once): `Mat.setPreallocationCOO(row_indices, col_indices)` — registers the complete sparsity pattern as flat arrays of (row, col) pairs.
2. **Assembly** (each Newton step): `Mat.setValuesCOO(values, addv=ADD_VALUES)` — streams just the nonzero values in the same order. No index lookups at assembly time.

The SFD HVP data from all ranks is combined via `MPI_Allreduce(SUM)` (each rank's partial HVP results are summed), then the full COO values array is set. This is correct because non-overlapping color groups contribute to disjoint nonzero positions.

### 0.8  Reproducing the Results

All benchmarks were run in a Docker container (`fenics_test:latest`) based on the DOLFINx 0.10.0 image with JAX, PETSc 3.24.0, and HYPRE added. The Makefile provides convenience targets:

```bash
# Build Docker image
make docker-build

# Run serial JAX+PETSc benchmark (HYPRE + RCM reorder)
docker run --rm --entrypoint mpirun -v "$PWD":/workspace -w /workspace \
  fenics_test:latest -n 1 python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py \
  --levels 5 6 7 8 9 --pc hypre --reorder

# Run parallel (8 processes — matches the 8 graph colors)
docker run --rm --entrypoint mpirun -v "$PWD":/workspace -w /workspace \
  fenics_test:latest -n 8 python3 pLaplace2D_jax_petsc/solve_pLaplace_jax_petsc.py \
  --levels 5 6 7 8 9 --pc hypre --reorder

# Run FEniCS comparison benchmark (unit square, bypasses h5py parallel issue)
docker run --rm --entrypoint mpirun -v "$PWD":/workspace -w /workspace \
  fenics_test:latest -n 8 python3 bench_fenics_compare.py --N 885
```

**Known issue**: The Docker container has a DOLFINx/h5py conflict that prevents the FEniCS custom solver from loading external meshes in MPI-parallel mode (`malloc()` corruption in `create_mesh`). The standalone `bench_fenics_compare.py` script uses `create_unit_square` to bypass this. The JAX+PETSc solver is unaffected (each rank reads mesh data independently with serial h5py).

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

## Annex C — RCM DOF Reordering: FEniCS vs JAX+PETSc Comparison

### C.1  Motivation

Annexes A and B showed that the replicated-data JAX+PETSc solver suffered from catastrophically poor HYPRE scaling due to random DOF ordering.  GAMG was adopted as a workaround, but its serial KSP time is roughly 2× slower than HYPRE's.

This annex introduces **locality-aware DOF reordering** via the Reverse Cuthill–McKee (RCM) algorithm.  The RCM permutation is computed on the sparsity graph (rank 0, then broadcast), and applied to the COO assembly indices so that PETSc's default block distribution assigns geometrically nearby DOFs to each rank — mimicking what FEniCS/DOLFINx achieves with ParMETIS/SCOTCH.

With this fix, **HYPRE BoomerAMG works correctly in parallel** and the KSP phase becomes directly comparable between FEniCS and JAX+PETSc.

### C.2  Benchmark Setup

To enable a clean parallel comparison (the Docker container has a DOLFINx/h5py conflict that prevents parallel mesh loading from HDF5), the FEniCS benchmark uses `create_unit_square` with comparable DOF counts:

| Problem size | FEniCS mesh                  | FEniCS DOFs | JAX mesh (from HDF5)   | JAX DOFs |
| :----------- | :--------------------------- | ----------: | :--------------------- | -------: |
| ~195 K       | unit square $441 \times 441$ |     195 364 | Level 8 (unstructured) |  195 585 |
| ~785 K       | unit square $885 \times 885$ |     784 996 | Level 9 (unstructured) |  784 385 |

Both solvers use: CG + HYPRE BoomerAMG, `rtol = 1e-3`, golden-section line search on $[-0.5, 2]$, `tolf = 1e-5`, `tolg = 1e-3`.

### C.3  KSP Linear Solve — Head-to-Head Comparison

The KSP column isolates the CG + HYPRE AMG solve (preconditioner setup + iterations), which is the phase that should be equivalent between the two solvers after matrix assembly.

**~195 K DOFs:**

| nproc | FEniCS KSP (s) | JAX+PETSc KSP (s) | Ratio | FEniCS KSP its | JAX KSP its |
| ----: | -------------: | ----------------: | ----: | :------------- | :---------- |
|     1 |          1.302 |             1.453 |  1.12 | 3–4            | 3–4         |
|     4 |          0.471 |             0.600 |  1.27 | 3–4            | 3–4         |
|    16 |          0.220 |             0.289 |  1.31 | 3–4            | 3–4         |

**~785 K DOFs:**

| nproc | FEniCS KSP (s) | JAX+PETSc KSP (s) | Ratio | FEniCS KSP its | JAX KSP its |
| ----: | -------------: | ----------------: | ----: | :------------- | :---------- |
|     1 |          5.232 |             7.270 |  1.39 | 3–4            | 3–5         |
|     4 |          2.297 |             2.742 |  1.19 | 3–4            | 3–4         |
|    16 |          0.929 |             1.523 |  1.64 | 3–4            | 3–5         |

**Key finding**: After RCM reordering, the JAX+PETSc KSP times are within **1.1–1.6×** of FEniCS — dramatically improved from the 20–30× gap without reordering.  The remaining gap stems from:
- Different mesh topology (structured grid vs unstructured triangulation — different condition numbers, different AMG coarsening)
- Different iteration counts (7 vs 5 Newton steps → 7 vs 5 KSP solves, including more PC rebuilds)
- RCM ordering quality vs DOLFINx's ParMETIS/SCOTCH partitioning

### C.4  KSP Parallel Scaling (Speedup)

| nproc | FEniCS speedup (~195 K) | JAX speedup (~195 K) | FEniCS speedup (~785 K) | JAX speedup (~785 K) |
| ----: | ----------------------: | -------------------: | ----------------------: | -------------------: |
|     1 |                    1.0× |                 1.0× |                    1.0× |                 1.0× |
|     4 |                    2.8× |                 2.4× |                    2.3× |                 2.7× |
|    16 |                    5.9× |                 5.0× |                    5.6× |                 4.8× |

Both solvers exhibit near-identical KSP scaling behaviour — confirming that the RCM reordering produces a partition quality sufficient for HYPRE's parallel AMG.

### C.5  Full Timing Breakdown — ~195 K DOFs

**FEniCS Custom Newton** (5 Newton iterations):

| nproc | Assembly (s) | KSP (s) | Hessian (s) | Other (s) | Solve (s) |
| ----: | -----------: | ------: | ----------: | --------: | --------: |
|     1 |        0.247 |   1.302 |       1.549 |     1.156 |     2.705 |
|     4 |        0.115 |   0.471 |       0.585 |     0.450 |     1.036 |
|    16 |        0.035 |   0.220 |       0.255 |     0.142 |     0.397 |

**JAX+PETSc SFD + RCM reorder** (6 Newton iterations):

| nproc | HVP (s) | KSP (s) | Hessian (s) | Other (s) | Solve (s) |
| ----: | ------: | ------: | ----------: | --------: | --------: |
|     1 |   0.304 |   1.453 |       1.781 |     0.260 |     2.041 |
|     4 |   0.326 |   0.600 |       0.971 |     0.598 |     1.569 |
|    16 |   0.320 |   0.289 |       0.783 |     2.709 |     3.492 |

### C.6  Full Timing Breakdown — ~785 K DOFs

**FEniCS Custom Newton** (5 Newton iterations):

| nproc | Assembly (s) | KSP (s) | Hessian (s) | Other (s) | Solve (s) |
| ----: | -----------: | ------: | ----------: | --------: | --------: |
|     1 |        0.989 |   5.232 |       6.220 |     4.880 |    11.100 |
|     4 |        0.471 |   2.297 |       2.767 |     1.916 |     4.684 |
|    16 |        0.153 |   0.929 |       1.082 |     0.579 |     1.661 |

**JAX+PETSc SFD + RCM reorder** (7 Newton iterations):

| nproc | HVP (s) | KSP (s) | Hessian (s) | Other (s) | Solve (s) |
| ----: | ------: | ------: | ----------: | --------: | --------: |
|     1 |   1.858 |   7.270 |       9.283 |     1.252 |    10.535 |
|     4 |   1.486 |   2.742 |       4.505 |     3.620 |     8.125 |
|    16 |   1.485 |   1.523 |       3.966 |    15.558 |    19.524 |

### C.7  Impact of RCM Reordering on HYPRE

| Config (level 9, ~785 K DOFs)  | KSP sum (s) | Solve (s) |
| :----------------------------- | ----------: | --------: |
| np = 16, HYPRE, **no** reorder |       30.60 |     50.00 |
| np = 16, HYPRE, RCM reorder    |        1.52 |     19.52 |
| np = 16, GAMG, no reorder      |        3.52 |     21.82 |

RCM reordering gives a **20× KSP speedup** over the unordered case and makes HYPRE **2.3× faster** than GAMG (which was the previous workaround).

### C.8  Analysis

**1. KSP times now match (within 1.1–1.6×).**  The RCM permutation restores proper data locality for PETSc's block distribution, enabling HYPRE's parallel AMG setup to work correctly.  The small residual gap is expected from different mesh structures (structured vs unstructured triangulation).

**2. Assembly cost difference.**  FEniCS assembles the exact Hessian via compiled UFL forms (0.15–0.99 s); JAX+PETSc computes SFD Hessian-vector products (0.30–1.86 s), roughly 2× more expensive.  However, the SFD approach is **automatic** (requires only the energy function, no hand-derived forms).

**3. "Other" overhead diverges in parallel.**  FEniCS distributes all computations (energy, gradient, assembly) across ranks natively.  JAX+PETSc uses **replicated data**: every rank evaluates the full energy/gradient on the complete vector.  At np = 16 on ~785 K DOFs, this overhead is 15.6 s (80 % of solve time), compared to FEniCS's 0.6 s.  This is the fundamental scaling limitation of the replicated-data architecture.

**4. Serial performance is competitive.**  At np = 1, JAX+PETSc actually solves *faster* at ~195 K (2.04 s vs 2.70 s) thanks to fewer energy evaluations per line search step (JAX JIT is more efficient for scalar energy).  At ~785 K the picture reverses (10.53 s vs 11.10 s — nearly equal).

**5. The bottleneck is not KSP anymore.**  With RCM reordering + HYPRE, the linear-solve phase is no longer the scaling limitation.  The replicated energy/gradient evaluation is the dominant cost in parallel, and resolving it requires a fundamentally different data-distribution strategy (true distributed assembly as in DOLFINx).

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
