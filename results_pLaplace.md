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

### All Solver Configurations

| lvl | dofs   | SNES serial | iters | Custom serial | iters | SNES 4-proc | iters | Custom 4-proc | iters | SNES 8-proc | iters | Custom 8-proc | iters | SNES 16-proc | iters | Custom 16-proc | iters | J(u)    |
| --- | ------ | ----------- | ----- | ------------- | ----- | ----------- | ----- | ------------- | ----- | ----------- | ----- | ------------- | ----- | ------------ | ----- | -------------- | ----- | ------- |
| 4   | 2945   | 0.043       | 10    | 0.043         | 8     | 0.029       | 11    | 0.025         | 8     | 0.026       | 10    | 0.026         | 9     | 0.033        | 11    | 0.031          | 9     | -7.9430 |
| 5   | 12033  | 0.167       | 10    | 0.193         | 9     | 0.071       | 10    | 0.075         | 9     | 0.050       | 10    | 0.055         | 9     | 0.054        | 10    | 0.058          | 9     | -7.9546 |
| 6   | 48641  | 0.478       | 7     | 0.754         | 9     | 0.169       | 7     | 0.259         | 9     | 0.110       | 7     | 0.164         | 9     | 0.087        | 8     | 0.119          | 9     | -7.9583 |
| 7   | 195585 | 2.152       | 8     | 3.373         | 10    | 0.768       | 8     | 1.014         | 9     | 0.463       | 8     | 0.681         | 10    | 0.274        | 8     | 0.386          | 10    | -7.9596 |
| 8   | 784385 | 10.026      | 9     | 15.162        | 11    | 3.772       | 9     | 5.337         | 11    | 2.430       | 9     | 3.508         | 11    | 1.404        | 9     | 1.921          | 11    | -7.9600 |

**Note on Custom Newton performance**: The Custom Newton solver is consistently slower due to two issues in its golden-section line search (see [analysis below](#custom-newton-line-search-analysis)).

### Custom Newton Line Search Analysis

The custom Newton solver uses a golden-section search on the interval $[0, 1]$ with tolerance `tol=0.1` to find the step size $\alpha$. Verbose output for mesh level 7 (serial) reveals the problem:

```
IT 1: |grad|_inf = 5.708e-02, alpha = 3.713e-01, ksp_its = 1
IT 2: |grad|_inf = 5.493e-01, alpha = 6.287e-01, ksp_its = 2
IT 3: |grad|_inf = 2.551e-01, alpha = 9.549e-01, ksp_its = 2
IT 4: |grad|_inf = 8.009e-02, alpha = 9.549e-01, ksp_its = 2
IT 5: |grad|_inf = 1.874e-02, alpha = 9.549e-01, ksp_its = 2
IT 6: |grad|_inf = 3.064e-03, alpha = 9.549e-01, ksp_its = 2
IT 7: |grad|_inf = 2.369e-04, alpha = 9.549e-01, ksp_its = 2
IT 8: |grad|_inf = 1.136e-05, alpha = 9.549e-01, ksp_its = 2
IT 9: |grad|_inf = 5.149e-07, alpha = 3.713e-01, ksp_its = 2
```

Two issues are evident:

1. **Alpha is capped at ~0.955**: The golden-section search with `tol=0.1` on `[0, 1]` can only resolve alpha to discrete golden-ratio values (0.371, 0.629, 0.955). Near convergence, the optimal step is $\alpha = 1$ (full Newton step), but the search always returns $\alpha \leq 0.955$, systematically undershooting and requiring more iterations (9–11 vs 7–9 for SNES).

2. **~5 energy evaluations per iteration**: Each golden-section step evaluates the energy (vector scatter + scalar assembly + allreduce). With `tol=0.1` this is about 5 evaluations per Newton step — pure overhead when a full step ($\alpha = 1$) would suffice (as SNES "basic" demonstrates).

**The SNES Newton with `linesearch_type=basic` (full step, no line search) is the correct and faster approach for this well-behaved problem.** The custom Newton's line search is solving a subproblem that doesn't need solving — it would only help for ill-conditioned problems where full Newton steps diverge.

### Strong Scaling (SNES Newton, 1–32 processes)

![Scaling plot](results/experiment_001/scaling.png)

Left: wall time vs number of MPI processes (log-log). Right: parallel speedup relative to serial. The dashed line shows ideal linear scaling. Larger problems (lvl 7, 8) scale well up to 16 processes; at 32 processes communication overhead starts to dominate for the smaller mesh levels.

To regenerate this plot:
```bash
python3 generate_scaling_plot.py results/experiment_001/
```

---

## Generating LaTeX Tables and Plots

The script `generate_latex_tables.py` reads JSON result files, aggregates repeated runs (median time), and produces publication-ready tables.

```bash
# Print LaTeX tables to stdout
python3 generate_latex_tables.py results/experiment_001/

# Save to .tex file (can be \input{}-ed in a LaTeX document)
python3 generate_latex_tables.py results/experiment_001/ --output results/experiment_001/tables.tex

# Print Markdown tables instead
python3 generate_latex_tables.py results/experiment_001/ --markdown
```

The generated LaTeX file is also committed at [results/experiment_001/tables.tex](results/experiment_001/tables.tex).
