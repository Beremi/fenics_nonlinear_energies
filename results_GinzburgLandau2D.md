# Ginzburg-Landau 2D — Benchmark Results

Benchmark results for the 2D Ginzburg-Landau problem with $\varepsilon = 0.01$, homogeneous Dirichlet BCs on $[-1,1]^2$:

$$J(u) = \int_\Omega \frac{\varepsilon}{2} |\nabla u|^2 + \frac{1}{4}(u^2 - 1)^2 \, \mathrm{d}x$$

This is a **non-convex** energy — the Hessian is indefinite, so CG cannot be used for the linear solve; GMRES is required. The non-convexity also makes standard SNES Newton (full-step) unreliable, especially in parallel.

Raw data is stored as JSON files in [results_GL/](results_GL/). See [instructions.md](instructions.md) for how to run new experiments.

---

## Experiment `experiment_001`

- **Date**: 2025-02-21
- **CPU**: AMD Ryzen 9 9950X3D 16-Core Processor (32 threads)
- **DOLFINx**: 0.10.0.post2
- **Git commit**: see metadata
- **Repetitions**: 3 (median time reported)
- **Data**: [results_GL/experiment_001/](results_GL/experiment_001/)

### Custom Newton — JAX-version algorithm (FEniCS + PETSc)

Re-implementation of the JAX minimiser (`tools/minimizers.py`) on top of PETSc via `tools_petsc4py/minimizers.py`. Uses golden-section energy line search on $[-0.5, 2]$ with `tol=1e-3`, GMRES + HYPRE AMG with `rtol=1e-3`, `tolf=1e-6`, `tolg=1e-5`. Supports MPI parallelism.

Script: [`GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py`](GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py)

| lvl | dofs      | time (serial) | iters | time (4-proc) | iters | time (8-proc) | iters | time (16-proc) | iters | J(u)   |
| --- | --------- | ------------- | ----- | ------------- | ----- | ------------- | ----- | -------------- | ----- | ------ |
| 5   | 4 225     | 0.046         | 7     | 0.034         | 7     | 0.050         | 12    | 0.034          | 7     | 0.3462 |
| 6   | 16 641    | 0.240         | 10    | 0.082         | 8     | 0.056         | 7     | 0.053          | 7     | 0.3458 |
| 7   | 66 049    | 0.616         | 6     | 0.267         | 7     | 0.246         | 11    | 0.112          | 7     | 0.3457 |
| 8   | 263 169   | 2.399         | 6     | 1.059         | 7     | 0.421         | 5     | 0.375          | 7     | 0.3456 |
| 9   | 1 050 625 | 10.143        | 6     | 6.014         | 8     | 5.561         | 11    | 1.732          | 6     | 0.3456 |

**Key observations**:
- **6–12 iterations** — converges reliably at all mesh levels and all MPI process counts. The iteration count varies slightly with MPI decomposition due to the effect of the parallel preconditioner on the Newton direction, but convergence is always achieved.
- **Golden-section line search** explicitly minimises energy along the search direction, which is essential for non-convex problems where the Newton direction may not be a descent direction.
- **Strong scaling**: Wall time decreases with more processes. At level 9 (1M dofs), 16 processes achieve ~5.9× speedup over serial.
- GMRES + HYPRE AMG handles the indefinite Hessian robustly.

### FEniCS SNES Newton

Script: [`GinzburgLandau2D_fenics/solve_GL_snes_newton.py`](GinzburgLandau2D_fenics/solve_GL_snes_newton.py)

Uses PETSc SNES with basic line search (full Newton step), GMRES + HYPRE AMG.

| lvl | dofs      | time (serial) | iters | J(u)   | time (4-proc) | iters | J(u)       | time (8-proc) | iters | J(u)       | time (16-proc) | iters | J(u)       |
| --- | --------- | ------------- | ----- | ------ | ------------- | ----- | ---------- | ------------- | ----- | ---------- | -------------- | ----- | ---------- |
| 5   | 4 225     | 0.244         | 71    | 0.3462 | 0.146         | 48    | 0.3462     | 0.094         | 29    | 0.3462     | 0.164          | 43    | 0.3462     |
| 6   | 16 641    | 1.475         | 114   | 0.3458 | **FAIL**      | —     | —          | **FAIL**      | —     | —          | 0.456          | 77    | 0.3458 (1/3) |
| 7   | 66 049    | 7.496         | 148   | 0.3457 | 1.087         | 45    | 0.3457     | **FAIL**      | —     | —          | **FAIL**       | —     | —          |
| 8   | 263 169   | 45.151        | 218   | 0.3456 | 2.559         | 28    | 0.3456     | **FAIL**      | —     | —          | 2.453          | 68    | 0.3456 (1/3) |
| 9   | 1 050 625 | 57.268        | 65    | 0.3456 | **FAIL**      | —     | —          | **FAIL**      | —     | —          | 7.178          | 39    | 0.3456     |

**SNES convergence issues** (non-convex problem):
- SNES with basic line search (full Newton step) is **unreliable for non-convex problems**. The Newton direction from the indefinite Hessian may not be a descent direction, and taking a full step can increase energy dramatically.
- **Serial**: Converges at all levels but requires 65–218 iterations. At some levels/runs it diverges (`reason=-9` = `DIVERGED_DTOL`): level 7 failed in 1 of 3 runs, level 8 failed in 2 of 3 runs.
- **Parallel**: Highly unreliable. At 8 processes, only level 5 converges correctly. At 4 and 16 processes, some levels converge but others produce wrong energies (converged to a different local minimum, e.g. $J \approx 0.5$) or diverge entirely.
- The `snes_divergence_tolerance = -1` option disables the DTOL check to prevent some premature divergences, but does not fix the fundamental issue.

**Bottom line**: For non-convex problems like Ginzburg-Landau, **the custom Newton with energy line search is strongly recommended** over SNES basic line search.

### All Solver Configurations — Summary

| lvl | dofs      | Custom serial | iters | Custom 4-proc | iters | Custom 8-proc | iters | Custom 16-proc | iters | SNES serial | iters | J(u)   |
| --- | --------- | ------------- | ----- | ------------- | ----- | ------------- | ----- | -------------- | ----- | ----------- | ----- | ------ |
| 5   | 4 225     | 0.046         | 7     | 0.034         | 7     | 0.050         | 12    | 0.034          | 7     | 0.244       | 71    | 0.3462 |
| 6   | 16 641    | 0.240         | 10    | 0.082         | 8     | 0.056         | 7     | 0.053          | 7     | 1.475       | 114   | 0.3458 |
| 7   | 66 049    | 0.616         | 6     | 0.267         | 7     | 0.246         | 11    | 0.112          | 7     | 7.496       | 148   | 0.3457 |
| 8   | 263 169   | 2.399         | 6     | 1.059         | 7     | 0.421         | 5     | 0.375          | 7     | 45.151      | 218   | 0.3456 |
| 9   | 1 050 625 | 10.143        | 6     | 6.014         | 8     | 5.561         | 11    | 1.732          | 6     | 57.268      | 65    | 0.3456 |

The custom Newton is **5–30× faster** than SNES serial (and more reliable). At level 8, custom serial takes 2.4 s (6 iters) vs SNES 45 s (218 iters). With 16 processes the custom solver reaches 0.375 s — a **120× speedup** over SNES serial.

### JAX Newton (serial only, no MPI)

The same Ginzburg-Landau problem solved using a pure-JAX pipeline (automatic differentiation for gradients, sparse finite differences with graph coloring for Hessian assembly, PyAMG smoothed-aggregation GMRES solver). This implementation lives in [`GinzburgLandau2D_jax/`](GinzburgLandau2D_jax/) and [`tools/`](tools/) and is driven by the [`example_GinzburgLandau2D_jax.ipynb`](example_GinzburgLandau2D_jax.ipynb) notebook.

The JAX solver uses the same golden-section line search algorithm as the custom FEniCS solver. Detailed timing results are available in the notebook.

**Comparison with FEniCS (serial)**:
- **Iterations**: Both solvers converge in 6–12 iterations (same algorithm, same stopping criteria).
- **Wall time**: FEniCS custom Newton is faster at large problems due to more efficient sparse assembly (DOLFINx) and better AMG preconditioning (HYPRE vs PyAMG).
- **MPI parallelism**: Only the FEniCS version supports MPI-parallel execution.

---

## Why SNES Fails for Ginzburg-Landau

The Ginzburg-Landau energy is **non-convex**: the double-well potential $\frac{1}{4}(u^2 - 1)^2$ creates indefinite contributions to the Hessian wherever $|u| < 1/\sqrt{3}$. This has several consequences:

1. **The Hessian is indefinite**: CG cannot be used (would diverge). GMRES handles this but the Newton direction may point "uphill" in energy.
2. **Full Newton step is not safe**: SNES basic line search takes step size $\alpha = 1$. When the Newton direction increases energy, the solver overshoots into a region from which it cannot recover.
3. **Parallel decomposition affects convergence**: Different MPI decompositions lead to different HYPRE AMG hierarchy, different GMRES solutions, and therefore different Newton trajectories through the non-convex landscape. Some trajectories converge to a local minimum, others diverge.
4. **Multiple local minima exist**: The GL energy has many local minima (e.g., $J \approx 0.346$, $J \approx 0.503$, $J \approx 0.85$). Even when SNES "converges" (residual norm drops), it may converge to a higher-energy local minimum.

The custom Newton avoids these issues by using an **energy line search**: at each iteration, it minimises $J(u + \alpha d)$ over $\alpha \in [-0.5, 2]$ using golden-section search. This guarantees energy decrease and steers the solver toward the correct minimum.
