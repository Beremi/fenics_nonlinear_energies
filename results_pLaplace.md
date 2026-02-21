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
