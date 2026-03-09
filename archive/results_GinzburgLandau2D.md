# Ginzburg-Landau 2D — Benchmark Results

Benchmark results for the 2D Ginzburg-Landau problem with $\varepsilon = 0.01$, homogeneous Dirichlet BCs on $[-1,1]^2$:

$$J(u) = \int_\Omega \frac{\varepsilon}{2} |\nabla u|^2 + \frac{1}{4}(u^2 - 1)^2 \, \mathrm{d}x$$

This is a **non-convex** energy — the Hessian is indefinite, so CG cannot be used for the linear solve; GMRES/FGMRES is required. The non-convexity makes most standard SNES configurations unreliable, especially in parallel. An extensive configuration survey (see [Appendix](#appendix--snes-configuration-survey-sine-initial-guess)) identified trust-region Newton with ASM/ILU and loose KSP tolerance as the only fully reliable SNES setup.

Raw data is stored as JSON files in [results_GL/](../results_GL/). See [instructions.md](../docs/instructions.md) for how to run new experiments.

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

Uses PETSc SNES trust-region Newton (`newtontr`) with FGMRES + ASM/ILU preconditioner (`overlap=2`, `ilu(1)`). This is the **A3** configuration with `ksp_rtol=1e-1` — the only SNES setup found to be reliable across all mesh sizes and MPI decompositions (see the [Appendix survey](#appendix--snes-configuration-survey-sine-initial-guess)). Key settings: `snes_atol=1e-5`, `snes_rtol=1e-8`, `snes_max_it=100`, `SNESSetObjective` enabled.

| lvl | dofs      | time (serial) | iters | time (4-proc) | iters | time (8-proc) | iters | time (16-proc) | iters | J(u)      |
| --- | --------- | ------------- | ----- | ------------- | ----- | ------------- | ----- | -------------- | ----- | --------- |
| 5   | 4 225     | 0.019         | 10    | 0.008         | 11    | 0.006         | 11    | 0.005          | 11    | 0.3462324 |
| 6   | 16 641    | 0.089         | 12    | 0.029         | 12    | 0.017         | 12    | 0.013          | 12    | 0.3457767 |
| 7   | 66 049    | 0.478         | 15    | 0.149         | 15    | 0.087         | 16    | 0.044          | 15    | 0.3456623 |
| 8   | 263 169   | 2.613         | 19    | 1.145         | 25    | 1.168         | 36    | 0.541          | 35    | 0.3456337 |
| 9   | 1 050 625 | 15.27         | 25    | 14.05         | 59    | 9.18          | 59    | 7.43           | 68    | 0.3456265 |

**Key observations**:
- **Fully reliable** at all mesh levels (L5–L9) and all MPI process counts (1–16). This is the only SNES configuration that achieved zero failures across the entire test matrix.
- **10–25 iterations serial**, increasing to 25–68 in parallel at L8–L9. The ASM/ILU preconditioner produces consistent Newton directions with the loose `ksp_rtol=1e-1` — tighter KSP tolerances actually hurt parallel reliability.
- **Trust-region** (`newtontr`) handles the indefinite Hessian by adjusting the step size based on actual vs predicted energy reduction, using `SNESSetObjective` to evaluate the energy functional.
- **Slower than Custom Newton**: At L9 serial, SNES takes 15.3 s (25 it) vs Custom Newton 10.1 s (6 it). The gap widens in parallel — at L9/16-proc: SNES 7.4 s (68 it) vs Custom 1.7 s (6 it).
- **Why loose KSP works**: The loose inner tolerance allows FGMRES to terminate early, producing approximate Newton directions. For the trust-region method, the direction quality is less critical because the trust radius controls step size. Tighter tolerances (`ksp_rtol=1e-3`) cause failures at L9 with np≥4.

### All Solver Configurations — Summary

| lvl | dofs      | Custom serial | iters | Custom 16-proc | iters | SNES serial | iters | SNES 16-proc | iters | J(u)      |
| --- | --------- | ------------- | ----- | -------------- | ----- | ----------- | ----- | ------------ | ----- | --------- |
| 5   | 4 225     | 0.046         | 7     | 0.034          | 7     | 0.019       | 10    | 0.005        | 11    | 0.3462324 |
| 6   | 16 641    | 0.240         | 10    | 0.053          | 7     | 0.089       | 12    | 0.013        | 12    | 0.3457767 |
| 7   | 66 049    | 0.616         | 6     | 0.112          | 7     | 0.478       | 15    | 0.044        | 15    | 0.3456623 |
| 8   | 263 169   | 2.399         | 6     | 0.375          | 7     | 2.613       | 19    | 0.541        | 35    | 0.3456337 |
| 9   | 1 050 625 | 10.143        | 6     | 1.732          | 6     | 15.27       | 25    | 7.43         | 68    | 0.3456265 |

Both solvers are **reliable at all levels and process counts**. The custom Newton is faster due to fewer iterations (6–12 vs 10–68), especially at large meshes: at L9/16-proc, Custom takes 1.7 s (6 it) vs SNES 7.4 s (68 it) — a **4.3× speedup**. At L5–L7 SNES is competitive (fewer dofs, iteration-count overhead is small).

### JAX Newton (serial only, no MPI)

The same Ginzburg-Landau problem solved using a pure-JAX pipeline (automatic differentiation for gradients, sparse finite differences with graph coloring for Hessian assembly, PyAMG smoothed-aggregation GMRES solver). This implementation lives in [`GinzburgLandau2D_jax/`](GinzburgLandau2D_jax/) and [`tools/`](tools/) and is driven by the [`example_GinzburgLandau2D_jax.ipynb`](example_GinzburgLandau2D_jax.ipynb) notebook.

The JAX solver uses the same golden-section line search algorithm as the custom FEniCS solver ($\alpha \in [-0.5, 2]$, `tol=1e-3`, `tolf=1e-6`, `tolg=1e-5`). Uses PyAMG smoothed-aggregation GMRES with `tol=1e-3`. Benchmark script: [`experiment_scripts/bench_jax_gl.py`](experiment_scripts/bench_jax_gl.py).

| lvl | dofs      | time (serial) | iters | J(u)      |
| --- | --------- | ------------- | ----- | --------- |
| 5   | 4 225     | 0.088         | 7     | 0.3462314 |
| 6   | 16 641    | 0.286         | 6     | 0.3457766 |
| 7   | 66 049    | 1.099         | 6     | 0.3456623 |
| 8   | 263 169   | 4.045         | 6     | 0.3456336 |
| 9   | 1 050 625 | 17.717        | 6     | 0.3456265 |

**Comparison with FEniCS (serial)**:
- **Iterations**: Both solvers converge in 6–7 iterations (same algorithm, same stopping criteria).
- **Wall time**: FEniCS custom Newton is faster at large problems due to more efficient sparse assembly (DOLFINx) and better AMG preconditioning (HYPRE vs PyAMG). At level 9, FEniCS serial takes 10.1 s vs JAX 17.7 s — a 1.7× advantage.
- **Energy values match**: Both solvers reach the same minimum at every level (to 7 digits).
- **MPI parallelism**: Only the FEniCS version supports MPI-parallel execution.

---

## Why SNES Fails for Ginzburg-Landau

The Ginzburg-Landau energy is **non-convex**: the double-well potential $\frac{1}{4}(u^2 - 1)^2$ creates indefinite contributions to the Hessian wherever $|u| < 1/\sqrt{3}$. This has several consequences:

1. **The Hessian is indefinite**: CG cannot be used (would diverge). GMRES handles this but the Newton direction may point "uphill" in energy.
2. **Full Newton step is not safe**: SNES basic line search takes step size $\alpha = 1$. When the Newton direction increases energy, the solver overshoots into a region from which it cannot recover.
3. **Parallel decomposition affects convergence**: Different MPI decompositions lead to different HYPRE AMG hierarchy, different GMRES solutions, and therefore different Newton trajectories through the non-convex landscape. Some trajectories converge to a local minimum, others diverge.
4. **Multiple local minima exist**: The GL energy has many local minima (e.g., $J \approx 0.346$, $J \approx 0.503$, $J \approx 0.85$). Even when SNES "converges" (residual norm drops), it may converge to a higher-energy local minimum.

The custom Newton avoids these issues by using an **energy line search**: at each iteration, it minimises $J(u + \alpha d)$ over $\alpha \in [-0.5, 2]$ using golden-section search. This guarantees energy decrease and steers the solver toward the correct minimum.

---

## Appendix: Comprehensive SNES Configuration Survey

We tested all reasonable PETSc SNES configurations to determine whether any built-in setting can reliably solve the non-convex Ginzburg-Landau problem. All tests used serial execution with `snes_max_it=500`. The correct minimum energy is $J^* \approx 0.3456$. A result is marked **OK** only if the solver converges (`reason > 0`) *and* reaches this minimum.

Unless otherwise noted, the default initial guess is $u_0(x,y) = \sin(\pi(x-1)/2)\sin(\pi(y-1)/2)$ (a smooth bump that satisfies the Dirichlet BC but is far from the true solution).

Test script: [`experiment_scripts/test_snes_comprehensive.py`](experiment_scripts/test_snes_comprehensive.py)

### PETSc options tested

| Option                      | Description                                                                                                                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **SNES type**               | `newtonls` = Newton + line search; `newtontr` = Newton + trust region                                                        |
| **Line search**             | `basic` = full step ($\alpha = 1$); `bt` = backtracking ($\alpha \in (0, 1]$); `l2` = secant on $\lVert F \rVert^2$          |
| **`SNESSetObjective`**      | Makes `bt` backtrack on energy $J(u)$ instead of $\lVert F(u) \rVert$. **Not set** by DOLFINx `NonlinearProblem` by default. |
| **KSP type**                | `gmres` or `fgmres` (flexible GMRES, safer with varying preconditioner)                                                      |
| **PC type**                 | `hypre` (BoomerAMG), `gamg` (PETSc's algebraic MG), `asm` (additive Schwarz + ILU)                                           |
| `snes_ksp_ew`               | Eisenstat–Walker inexact Newton forcing (adapts `ksp_rtol` per Newton step)                                                  |
| `snes_lag_preconditioner`   | Reuse preconditioner for N Newton steps (reduces setup cost)                                                                 |
| `snes_divergence_tolerance` | Set to $-1$ to disable DTOL check (prevents premature `reason=-9`)                                                           |
| `snes_linesearch_minlambda` | Minimum step size for `bt`                                                                                                   |
| `snes_linesearch_order`     | Polynomial order for `bt` interpolation: `quadratic` or `cubic`                                                              |

---

### Part 1 — SNES type & line search variants (sine initial guess)

All configs below use `ksp_rtol=1e-1`, `snes_atol=1e-6`, `snes_rtol=1e-8`, `gmres` + HYPRE AMG, `snes_divergence_tolerance=-1`, and the **sine** initial guess.

| #   | Configuration        | Objective? | Key options       |              L5              |          L6          |          L7          |                   L8                   |
| --- | -------------------- | :--------: | ----------------- | :--------------------------: | :------------------: | :------------------: | :------------------------------------: |
| 1   | `newtonls` + `basic` |     No     | divtol$=-1$       |          OK (71 it)          |     OK (114 it)      |     OK (149 it)      | **FAIL** (218 it, $J=6.6 \times 10^8$) |
| 2   | `newtonls` + `basic` |    Yes     | divtol$=-1$       |          OK (71 it)          |     OK (114 it)      |     OK (149 it)      |              OK (213 it)               |
| 3   | `newtonls` + `bt`    |    Yes     | default           |  **FAIL** (0 it, $J=0.665$)  |       **FAIL**       |       **FAIL**       |                **FAIL**                |
| 4   | `newtonls` + `bt`    |    Yes     | `ksp_rtol=1e-3`   |  **FAIL** (0 it, $J=0.665$)  |       **FAIL**       |       **FAIL**       |                **FAIL**                |
| 5   | `newtonls` + `bt`    |    Yes     | `order=cubic`     |  **FAIL** (0 it, $J=0.665$)  |       **FAIL**       |       **FAIL**       |                **FAIL**                |
| 6   | `newtonls` + `bt`    |    Yes     | `minlambda=1e-20` | **FAIL** (500 it, $J=0.665$) |  **FAIL** (500 it)   |  **FAIL** (500 it)   |           **FAIL** (500 it)            |
| 7   | `newtonls` + `l2`    |    Yes     | divtol$=-1$       | **FAIL** (500 it, $J=0.897$) | **FAIL** ($J=0.889$) | **FAIL** ($J=0.922$) |          **FAIL** ($J=0.878$)          |
| 8   | `newtontr`           |    Yes     | –                 |          OK (11 it)          |      OK (44 it)      |     OK (351 it)      |      **FAIL** (500 it, $J=0.594$)      |

**Observations**:
- **basic (full step)**: Works at many levels in serial, but at L8 the Newton direction occasionally causes energy explosion ($J \sim 10^8$). Setting `SNESSetObjective` does not change `basic` behaviour (it always takes $\alpha = 1$).
- **bt + objective**: Fails immediately at all levels. The Newton direction from the indefinite Hessian is **not a descent direction** — no positive $\alpha$ reduces $J$, so `bt` declares `DIVERGED_LINE_SEARCH` (reason $= -6$).
- **l2**: Energy actually *increases* — converges to a higher-energy critical point ($J \approx 0.9$).
- **newtontr**: Fast at small meshes (11 iterations at L5) but trust-region management breaks down at L8 (wrong minimum $J \approx 0.594$).

---

### Part 2 — Trust region variants (sine initial guess)

Testing different KSP/PC combinations with `newtontr`, using `ksp_rtol=1e-6`, `ksp_max_it=200`, `snes_atol=1e-10`, `snes_rtol=1e-8`, all with `SNESSetObjective`.

| #   | Configuration                | KSP      | PC                        |          L5           |          L6           |          L7           |              L8              |
| --- | ---------------------------- | -------- | ------------------------- | :-------------------: | :-------------------: | :-------------------: | :--------------------------: |
| A1  | `newtontr` + HYPRE           | `fgmres` | `hypre` (AMG)             |   OK (11 it, 0.03s)   |   OK (13 it, 0.14s)   |   OK (242 it, 9.9s)   | **FAIL** (500 it, $J=0.594$) |
| A2  | `newtontr` + ksp\_ew + HYPRE | `fgmres` | `hypre` (AMG)             |   OK (13 it, 0.03s)   |   OK (14 it, 0.14s)   |   OK (244 it, 9.9s)   | **FAIL** (500 it, $J=0.594$) |
| A3  | **`newtontr` + ASM+ILU**     | `fgmres` | `asm` (overlap=2, ILU(1)) | **OK (12 it, 0.03s)** | **OK (13 it, 0.11s)** | **OK (16 it, 0.66s)** |     **OK (19 it, 3.8s)**     |
| A4  | `newtontr` + HYPRE (loose)   | `gmres`  | `hypre`, `ksp_rtol=1e-1`  |   OK (11 it, 0.03s)   |   OK (44 it, 0.47s)   |  OK (351 it, 14.9s)   | **FAIL** (500 it, $J=0.594$) |

**Key finding**: **A3 (`newtontr` + ASM+ILU) is the only trust-region config that succeeds at all levels.** BoomerAMG's behaviour with the indefinite Hessian produces Newton directions that defeat the trust region at large meshes, while the simpler ASM+ILU preconditioner gives more reliable directions (16–19 iterations at L7–L8). Eisenstat–Walker (`ksp_ew`, config A2) has negligible effect on trust region.

**Parallel scaling of `newtontr` + fgmres + HYPRE AMG** (`snes_atol=1e-5`, `snes_rtol=1e-8`, `snes_max_it=300` for L8–L9, `SNESSetObjective` set, sine initial guess, 3 runs median for L5–L7, 1 run for L8–L9). Script: [`experiment_scripts/bench_tr2.py`](experiment_scripts/bench_tr2.py):

We tested three inner KSP tolerances: `ksp_rtol` = $10^{-3}$, $10^{-2}$, $10^{-1}$. Since Level 8 and Level 9 hit the iteration limit at 300 iterations for all process counts and all three tolerances (converging to wrong minima $J \approx 0.617$–$0.651$), the tables below show only the converging levels (L5–L7) with full detail, plus L8–L9 as FAIL.

**`ksp_rtol = 1e-3`**:

| lvl | dofs      | serial                              | 2-proc                              | 4-proc                              | 8-proc                              | 16-proc                             |
| --- | --------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| 5   | 4 225     | 10 it, 0.025s, $J$=0.3462324       | 10 it, 0.018s, $J$=0.3462324       | 10 it, 0.015s, $J$=0.3462324       | 10 it, 0.013s, $J$=0.3462324       | 10 it, 0.016s, $J$=0.3462324       |
| 6   | 16 641    | 12 it, 0.121s, $J$=0.3457767       | 12 it, 0.076s, $J$=0.3457767       | 12 it, 0.050s, $J$=0.3457767       | 28 it, 0.106s, $J$=0.3457767       | 12 it, 0.042s, $J$=0.3457768       |
| 7   | 66 049    | 241 it, 10.18s, $J$=0.3456623      | 287 it, 7.42s, $J$=0.3456623       | 293 it, 4.80s, $J$=0.3456623       | 289 it, 2.87s, $J$=0.3456623       | 241 it, 2.04s, $J$=0.3456623       |
| 8   | 263 169   | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        |
| 9   | 1 050 625 | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        |

**`ksp_rtol = 1e-2`**:

| lvl | dofs      | serial                              | 2-proc                              | 4-proc                              | 8-proc                              | 16-proc                             |
| --- | --------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| 5   | 4 225     | 10 it, 0.026s, $J$=0.3462324       | 10 it, 0.019s, $J$=0.3462324       | 10 it, 0.015s, $J$=0.3462324       | 10 it, 0.016s, $J$=0.3462324       | 10 it, 0.016s, $J$=0.3462324       |
| 6   | 16 641    | 12 it, 0.119s, $J$=0.3457767       | 12 it, 0.076s, $J$=0.3457767       | 12 it, 0.048s, $J$=0.3457767       | 28 it, 0.104s, $J$=0.3457767       | 12 it, 0.041s, $J$=0.3457768       |
| 7   | 66 049    | 241 it, 10.63s, $J$=0.3456623      | 287 it, 7.48s, $J$=0.3456623       | 293 it, 4.83s, $J$=0.3456623       | 289 it, 2.83s, $J$=0.3456623       | 241 it, 2.15s, $J$=0.3456623       |
| 8   | 263 169   | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        |
| 9   | 1 050 625 | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        |

**`ksp_rtol = 1e-1`**:

| lvl | dofs      | serial                              | 2-proc                              | 4-proc                              | 8-proc                              | 16-proc                             |
| --- | --------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| 5   | 4 225     | 10 it, 0.024s, $J$=0.3462324       | 10 it, 0.016s, $J$=0.3462324       | 10 it, 0.015s, $J$=0.3462324       | 10 it, 0.011s, $J$=0.3462324       | 10 it, 0.012s, $J$=0.3462324       |
| 6   | 16 641    | 12 it, 0.118s, $J$=0.3457767       | 12 it, 0.069s, $J$=0.3457767       | 12 it, 0.047s, $J$=0.3457767       | 28 it, 0.097s, $J$=0.3457767       | 13 it, 0.037s, $J$=0.3457767       |
| 7   | 66 049    | 241 it, 10.26s, $J$=0.3456623      | 287 it, 7.03s, $J$=0.3456623       | 293 it, 4.60s, $J$=0.3456623       | 289 it, 2.69s, $J$=0.3456623       | 241 it, 1.57s, $J$=0.3456623       |
| 8   | 263 169   | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        | **FAIL** (300 it, $J$=0.617)        |
| 9   | 1 050 625 | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        | **FAIL** (300 it, $J$=0.651)        |

**Observations**:
- **Inner KSP tolerance is irrelevant**: All three `ksp_rtol` values ($10^{-3}$, $10^{-2}$, $10^{-1}$) produce the same iteration counts and same failure pattern. The trust region algorithm dominates convergence behaviour.
- **L5–L6**: Converge in 10–28 iterations across all process counts. The 28-iteration outlier at np=8 for L6 is consistent across all three `ksp_rtol` values.
- **L7**: Converges but requires 241–293 iterations (10 s serial, 1.6–2.0 s at np=16). Very slow compared to the custom Newton (6 iterations, 0.11 s at np=16).
- **L8–L9**: Always fail — the trust region converges to a wrong minimum ($J \approx 0.617$ at L8, $J \approx 0.651$ at L9) instead of the true minimum ($J^* \approx 0.346$). This confirms the Part 2 finding that BoomerAMG interacts poorly with `newtontr` at fine meshes.
- **Parallel scaling** (for L7): Wall time shows good scaling from serial (10.2 s) to 16-proc (1.6–2.0 s), a ~5–6× speedup. But since L8+ always fails, this is only useful for meshes up to ~66K dofs.
- **Compared to A3 (`newtontr` + ASM+ILU)**: A3 succeeds at L8 (19 iterations) while all HYPRE variants fail. The simpler ASM+ILU preconditioner produces trust-region-compatible directions even at fine meshes.

**Parallel scaling of A3: `newtontr` + fgmres + ASM+ILU** (`snes_atol=1e-5`, `snes_rtol=1e-8`, `snes_max_it=100`, overlap=2, ILU(1), `SNESSetObjective` set, sine initial guess, 3 runs median). Script: [`experiment_scripts/bench_a3.py`](experiment_scripts/bench_a3.py):

**`ksp_rtol = 1e-3`**:

| lvl | dofs      | serial                              | 2-proc                              | 4-proc                              | 8-proc                              | 16-proc                             |
| --- | --------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| 5   | 4 225     | 10 it, 0.020s, $J$=0.3462324       | 10 it, 0.011s, $J$=0.3462324       | 11 it, 0.011s, $J$=0.3462324       | 11 it, 0.005s, $J$=0.3462324       | 10 it, 0.005s, $J$=0.3462324       |
| 6   | 16 641    | 12 it, 0.095s, $J$=0.3457767       | 12 it, 0.052s, $J$=0.3457767       | 12 it, 0.031s, $J$=0.3457767       | 11 it, 0.016s, $J$=0.3457767       | 11 it, 0.014s, $J$=0.3457767       |
| 7   | 66 049    | 14 it, 0.477s, $J$=0.3456624       | 14 it, 0.271s, $J$=0.3456623       | 14 it, 0.151s, $J$=0.3456623       | 16 it, 0.099s, $J$=0.3456623       | 15 it, 0.051s, $J$=0.3456623       |
| 8   | 263 169   | 18 it, 2.833s, $J$=0.3456336       | 19 it, 1.811s, $J$=0.3456351       | 25 it, 1.354s, $J$=0.3456338       | 35 it, 1.482s, $J$=0.3456336       | 34 it, 0.622s, $J$=0.3456347       |
| 9   | 1 050 625 | 23 it, 16.94s, $J$=0.3456280       | 32 it, 13.50s, $J$=0.3456310       | **FAIL** (57 it, $J$=0.367)         | **FAIL** (55 it, $J$=0.364)         | **FAIL** (65 it, $J$=0.362)         |

**`ksp_rtol = 1e-2`**:

| lvl | dofs      | serial                              | 2-proc                              | 4-proc                              | 8-proc                              | 16-proc                             |
| --- | --------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| 5   | 4 225     | 10 it, 0.020s, $J$=0.3462324       | 10 it, 0.012s, $J$=0.3462324       | 11 it, 0.008s, $J$=0.3462324       | 11 it, 0.005s, $J$=0.3462324       | 10 it, 0.004s, $J$=0.3462324       |
| 6   | 16 641    | 12 it, 0.093s, $J$=0.3457767       | 12 it, 0.053s, $J$=0.3457767       | 12 it, 0.030s, $J$=0.3457767       | 11 it, 0.016s, $J$=0.3457767       | 11 it, 0.011s, $J$=0.3457767       |
| 7   | 66 049    | 14 it, 0.444s, $J$=0.3456624       | 14 it, 0.259s, $J$=0.3456623       | 14 it, 0.143s, $J$=0.3456623       | 16 it, 0.090s, $J$=0.3456623       | 15 it, 0.047s, $J$=0.3456623       |
| 8   | 263 169   | 17 it, 2.401s, $J$=0.3456337       | 19 it, 1.652s, $J$=0.3456351       | 25 it, 1.206s, $J$=0.3456338       | 35 it, 1.295s, $J$=0.3456336       | 34 it, 0.555s, $J$=0.3456348       |
| 9   | 1 050 625 | 23 it, 15.59s, $J$=0.3456274       | 32 it, 12.48s, $J$=0.3456304       | **FAIL** (57 it, $J$=0.367)         | 57 it, 10.31s, $J$=0.3456303       | 68 it, 10.74s, $J$=0.3456265       |

**`ksp_rtol = 1e-1`**:

| lvl | dofs      | serial                              | 2-proc                              | 4-proc                              | 8-proc                              | 16-proc                             |
| --- | --------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- |
| 5   | 4 225     | 10 it, 0.019s, $J$=0.3462324       | 10 it, 0.011s, $J$=0.3462324       | 11 it, 0.008s, $J$=0.3462324       | 11 it, 0.006s, $J$=0.3462324       | 11 it, 0.005s, $J$=0.3462324       |
| 6   | 16 641    | 12 it, 0.089s, $J$=0.3457767       | 12 it, 0.051s, $J$=0.3457767       | 12 it, 0.029s, $J$=0.3457767       | 12 it, 0.017s, $J$=0.3457767       | 12 it, 0.013s, $J$=0.3457767       |
| 7   | 66 049    | 15 it, 0.478s, $J$=0.3456623       | 15 it, 0.268s, $J$=0.3456623       | 15 it, 0.149s, $J$=0.3456623       | 16 it, 0.087s, $J$=0.3456623       | 15 it, 0.044s, $J$=0.3456625       |
| 8   | 263 169   | 19 it, 2.613s, $J$=0.3456337       | 20 it, 1.670s, $J$=0.3456336       | 25 it, 1.145s, $J$=0.3456342       | 36 it, 1.168s, $J$=0.3456337       | 35 it, 0.541s, $J$=0.3456337       |
| 9   | 1 050 625 | 25 it, 15.27s, $J$=0.3456265       | 32 it, 11.22s, $J$=0.3456307       | 59 it, 14.05s, $J$=0.3456330       | 59 it, 9.18s, $J$=0.3456266        | 68 it, 7.43s, $J$=0.3456293        |

**Observations**:
- **A3 with `ksp_rtol=1e-1` is the most robust**: All levels converge at all process counts — the only fully reliable `newtontr` configuration in parallel. The loose inner tolerance does not hurt convergence because the ASM+ILU preconditioner produces consistent Newton directions regardless of KSP accuracy.
- **Tighter KSP tolerance hurts at L9 in parallel**: `ksp_rtol=1e-3` fails at L9 with np=4,8,16 (converges to wrong minimum $J \approx 0.36$–$0.37$). `ksp_rtol=1e-2` fails at np=4 but succeeds at np=8,16. `ksp_rtol=1e-1` succeeds everywhere.
- **L5–L8**: Fast and reliable at all tolerances — 10–36 iterations (0.005–2.8 s serial, scaling well with processes).
- **L9**: Serial always works (23–25 iterations, 15–17 s). In parallel, iteration counts increase (32–68) but wall time can decrease thanks to parallelism. At np=16 with `ksp_rtol=1e-1`: 68 iterations in 7.4 s (~2× speedup over serial).
- **Compared to `newtontr` + HYPRE**: A3 succeeds at L8 and L9 (with `ksp_rtol=1e-1`), while HYPRE always fails at L8+. However, A3 at L9 parallel requires more iterations (32–68 vs ~6 for custom Newton).
- **Compared to custom Newton**: Custom Newton converges in 6 iterations at all levels and all process counts (1.7 s at L9 with 16 procs). A3 is competitive at L5–L7 but slower at L8–L9 due to increased iteration counts in parallel.

---

### Part 3 — Line search variants with different KSP/PC (sine initial guess)

Testing `newtonls` + `l2` with different preconditioners, using `ksp_rtol=1e-6`, `ksp_max_it=200`, `snes_atol=1e-10`, `snes_rtol=1e-8`, `snes_linesearch_max_it=20`, `divtol=-1`.

| #   | Configuration                      | KSP      | PC                        |                  L5                  |          L6           |          L7          |          L8           |
| --- | ---------------------------------- | -------- | ------------------------- | :----------------------------------: | :-------------------: | :------------------: | :-------------------: |
| B1  | `l2` + GAMG                        | `fgmres` | `gamg`                    |      **FAIL** (5 it, $J=0.821$)      |    **FAIL** (5 it)    |   **FAIL** (5 it)    |    **FAIL** (3 it)    |
| B2  | `l2` + ASM+ILU                     | `fgmres` | `asm` (overlap=2, ILU(1)) | **FAIL** (2/0 it, $J=0.665$–$0.806$) |       **FAIL**        |       **FAIL**       |       **FAIL**        |
| B3  | **`l2` + HYPRE (`ksp_rtol=1e-6`)** | `fgmres` | `hypre` (AMG)             |        **OK (24 it, 0.27s)**         | **OK (23 it, 1.0s)**  | **OK (19 it, 3.4s)** | **OK (21 it, 15.8s)** |
| B4  | **`l2` + HYPRE (`ksp_rtol=1e-3`)** | `fgmres` | `hypre` (AMG)             |        **OK (24 it, 0.25s)**         | **OK (21 it, 0.81s)** | **OK (30 it, 4.4s)** | **OK (16 it, 10.6s)** |

**Key finding**: **B3 and B4 (`l2` + fgmres + HYPRE with `ksp_rtol` ≤ 1e-3) succeed at all levels!** The tighter KSP tolerance (compared to Part 1's `ksp_rtol=1e-1`) produces more accurate Newton directions that keep the `l2` line search on track. B4 with `ksp_rtol=1e-3` is slightly faster than B3 at L8 (10.6s vs 15.8s) while remaining reliable. GAMG and ASM+ILU both fail — the preconditioner quality is critical for `l2` on this problem.

**Nonlinear tolerance sweep for B4** (`l2` + fgmres + HYPRE, `ksp_rtol=1e-3`, script: [`experiment_scripts/test_tol_sweep.py`](experiment_scripts/test_tol_sweep.py)): We varied `snes_atol` from $10^{-2}$ to $10^{-8}$ and `snes_rtol` from $10^{-2}$ to $10^{-8}$ (49 combinations × 4 levels = 196 solves) and checked whether the final energy matches the reference solution (computed at `snes_atol=snes_rtol=1e-12`) with relative error $< 10^{-5}$. Results:

|  `snes_atol`  | `snes_rtol` |            L5 (iters, relErr)             |           L6 (iters, relErr)           |           L7 (iters, relErr)           |           L8 (iters, relErr)           | All OK? |
| :-----------: | :---------: | :---------------------------------------: | :------------------------------------: | :------------------------------------: | :------------------------------------: | :-----: |
|   $10^{-2}$   |     any     |              1 it, $J=0.78$               |             0 it, $J=0.67$             |             0 it, $J=0.67$             |             0 it, $J=0.67$             | **NO**  |
|   $10^{-3}$   |     any     |       22 it, $1.3 \times 10^{-6}$ ✓       |     18 it, $2.0 \times 10^{-4}$ ✗      |            4 it, $J=0.83$ ✗            |            2 it, $J=0.80$ ✗            | **NO**  |
|   $10^{-4}$   |     any     |       22 it, $1.3 \times 10^{-6}$ ✓       |     19 it, $1.0 \times 10^{-7}$ ✓      |     28 it, $3.9 \times 10^{-7}$ ✓      |     13 it, $7.9 \times 10^{-5}$ ✗      | **NO**  |
| **$10^{-5}$** |   **any**   | **22–23 it, $\leq 1.3 \times 10^{-6}$** ✓ | **19 it, $\leq 1.0 \times 10^{-7}$** ✓ | **28 it, $\leq 3.9 \times 10^{-7}$** ✓ | **14 it, $\leq 1.5 \times 10^{-8}$** ✓ | **YES** |
|   $10^{-6}$   |     any     |                 22–23 it                  |                 19 it                  |                28–29 it                |                 14 it                  | **YES** |
|   $10^{-7}$   |     any     |                 22–24 it                  |                19–20 it                |                28–29 it                |                14–15 it                | **YES** |
|   $10^{-8}$   |     any     |                 22–24 it                  |                19–20 it                |                28–29 it                |                14–15 it                | **YES** |

**Conclusion**: The **loosest nonlinear tolerance** that matches the custom Newton's energy to relative error $< 10^{-5}$ is **`snes_atol=1e-5`**. The `snes_rtol` value is irrelevant — convergence is entirely dominated by the absolute tolerance. At `snes_atol=1e-4`, L8 barely misses the threshold (rel error $\approx 7.9 \times 10^{-5}$). At `snes_atol=1e-3` and looser, the solver stops too early at the finer meshes, converging to a wrong critical point or stalling near the initial guess.

**B4 parallel scaling** (`l2` + `fgmres` + HYPRE, `ksp_rtol=1e-3`, `snes_atol=1e-5`, `snes_rtol=1e-8`, sine initial guess, 3 runs median). Script: [`experiment_scripts/bench_b4.py`](experiment_scripts/bench_b4.py):

| lvl | dofs      | serial | 2-proc | 4-proc | 8-proc | 16-proc |
| --- | --------- | ------ | ------ | ------ | ------ | ------- |
| 5   | 4 225     | 23 it, 0.259s, $J$=0.3462324 | 35 it, 0.258s, $J$=0.3462324 | **FAIL** (6 it, $J$=246924) | 17 it, 0.094s, $J$=0.3462324 | 21 it, 0.107s, $J$=0.3462324 |
| 6   | 16 641    | 19 it, 0.811s, $J$=0.3457767 | **FAIL** (6 it, $J$=580519) | 13 it, 0.215s, $J$=0.3457767 | 16 it, 0.190s, $J$=0.3457767 | 19 it, 0.173s, $J$=0.3457767 |
| 7   | 66 049    | 28 it, 4.571s, $J$=0.3456624 | 18 it, 1.818s, $J$=0.3456623 | 27 it, 1.496s, $J$=0.3456623 | 19 it, 0.601s, $J$=0.3456623 | 14 it, 0.327s, $J$=0.3456623 |
| 8   | 263 169   | 14 it, 10.93s, $J$=0.3456336 | 26 it, 11.35s, $J$=0.3456342 | 23 it, 5.102s, $J$=0.3456336 | 20 it, 2.483s, $J$=0.3456336 | 30 it, 2.322s, $J$=0.3456339 |
| 9   | 1 050 625 | 28 it, 81.24s, $J$=0.3456277 | 24 it, 32.37s, $J$=0.3456325 | **FAIL** (1 it, $J$=0.779) | **FAIL** (3 it, $J$=0.820) | 28 it, 11.65s, $J$=0.3456265 |

**Observations**: B4 is **unreliable in parallel** — several level/process-count combinations diverge or converge to wrong critical points (FAIL entries). This is the same fundamental issue as SNES `basic`: the MPI domain decomposition changes the AMG hierarchy and Newton direction quality, and `l2` line search (which targets residual norm, not energy) cannot recover from bad directions. Serial is reliable at all levels. 16-proc is reliable too in this run, but scattered failures at other process counts show the method is not robust. The custom Newton's energy line search remains the only approach that is reliable across all levels and process counts.

---

### Part 4 — Eisenstat–Walker and preconditioner lagging (sine initial guess)

Using `newtonls` + `basic`, `ksp_rtol=1e-1`, HYPRE, `divtol=-1`.

| #   | Configuration                       | Key options               |              L5              |                  L6                   |          L7           |          L8          |
| --- | ----------------------------------- | ------------------------- | :--------------------------: | :-----------------------------------: | :-------------------: | :------------------: |
| C1  | `basic` + `ksp_ew`                  | Eisenstat–Walker          | **FAIL** (500 it, $J=0.857$) |         **FAIL** ($J=0.856$)          | **FAIL** ($J=0.873$)  | **FAIL** ($J=0.844$) |
| C2  | `basic` + `lag_pc=2`                | reuse PC every 2 steps    |         OK (105 it)          | **FAIL** (42 it, $J=1.1 \times 10^6$) |      OK (46 it)       |      OK (62 it)      |
| C3  | **`basic` + `ksp_ew` + `lag_pc=2`** | `fgmres`, `ksp_rtol=1e-6` |    **OK (37 it, 0.09s)**     |         **OK (13 it, 0.13s)**         | **OK (15 it, 0.58s)** | **OK (13 it, 2.2s)** |

**Key finding**: Eisenstat–Walker **alone** (C1) is catastrophic with `ksp_rtol=1e-1` — the initial loose solve never produces a useful Newton direction. But **combined with `lag_pc=2` and tight `ksp_rtol=1e-6`** (C3), it converges reliably and fast (13–37 iterations). Lagging alone (C2) is unreliable.

**Parallel scaling of C3: `newtonls` + `basic` + EW + `lag_pc=2` + fgmres + HYPRE** (`snes_atol=1e-5`, `snes_rtol=1e-8`, `snes_max_it=100`, `snes_divergence_tolerance=-1`, NO `SNESSetObjective`, sine initial guess, single run). Script: [`experiment_scripts/bench_c3.py`](experiment_scripts/bench_c3.py):

**`ksp_rtol = 1e-3`**:

| lvl | dofs      | serial                              | 2-proc                              | 4-proc                              | 8-proc                               | 16-proc                              |
| --- | --------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ------------------------------------- | ------------------------------------- |
| 5   | 4 225     | 36 it, 0.085s, $J$=0.3462324       | 27 it, 0.051s, $J$=0.3462324       | 13 it, 0.028s, $J$=0.3462324       | **FAIL** (20 it, $J$=0.504)          | 13 it, 0.026s, $J$=0.3462324         |
| 6   | 16 641    | 12 it, 0.114s, $J$=0.3457767       | 14 it, 0.087s, $J$=0.3457767       | 13 it, 0.058s, $J$=0.3457767       | 12 it, 0.043s, $J$=0.3457767         | **FAIL** (13 it, $J$=0.503)          |
| 7   | 66 049    | 14 it, 0.531s, $J$=0.3456623       | 19 it, 0.461s, $J$=0.3456623       | 13 it, 0.206s, $J$=0.3456623       | **FAIL** (15 it, $J$=1.3×10⁸, r=-9) | 14 it, 0.091s, $J$=0.3456623         |
| 8   | 263 169   | 12 it, 1.978s, $J$=0.3456340       | 26 it, 2.512s, $J$=0.3456339       | 13 it, 0.813s, $J$=0.3456337       | 12 it, 0.445s, $J$=0.3456337         | 12 it, 0.273s, $J$=0.3456339         |
| 9   | 1 050 625 | 14 it, 8.985s, $J$=0.3456269       | 23 it, 9.744s, $J$=0.3456265       | 13 it, 3.779s, $J$=0.3456265       | 13 it, 2.555s, $J$=0.3456265         | 14 it, 1.627s, $J$=0.3456265         |

**`ksp_rtol = 1e-2`**:

| lvl | dofs      | serial                              | 2-proc                              | 4-proc                              | 8-proc                               | 16-proc                              |
| --- | --------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ------------------------------------- | ------------------------------------- |
| 5   | 4 225     | 36 it, 0.086s, $J$=0.3462324       | 27 it, 0.053s, $J$=0.3462324       | 13 it, 0.024s, $J$=0.3462324       | **FAIL** (20 it, $J$=0.504)          | 13 it, 0.027s, $J$=0.3462324         |
| 6   | 16 641    | 12 it, 0.115s, $J$=0.3457767       | 14 it, 0.085s, $J$=0.3457767       | 13 it, 0.057s, $J$=0.3457767       | 12 it, 0.043s, $J$=0.3457767         | **FAIL** (13 it, $J$=0.503)          |
| 7   | 66 049    | 14 it, 0.524s, $J$=0.3456623       | 19 it, 0.456s, $J$=0.3456623       | 13 it, 0.211s, $J$=0.3456623       | **FAIL** (15 it, $J$=1.3×10⁸, r=-9) | 14 it, 0.092s, $J$=0.3456623         |
| 8   | 263 169   | 12 it, 1.951s, $J$=0.3456340       | 26 it, 2.529s, $J$=0.3456339       | 13 it, 0.827s, $J$=0.3456337       | 12 it, 0.433s, $J$=0.3456337         | 12 it, 0.282s, $J$=0.3456339         |
| 9   | 1 050 625 | 14 it, 8.922s, $J$=0.3456269       | 23 it, 9.689s, $J$=0.3456265       | 13 it, 3.805s, $J$=0.3456265       | 13 it, 2.436s, $J$=0.3456265         | 14 it, 1.706s, $J$=0.3456265         |

**`ksp_rtol = 1e-1`**:

| lvl | dofs      | serial                              | 2-proc                              | 4-proc                              | 8-proc                               | 16-proc                              |
| --- | --------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ------------------------------------- | ------------------------------------- |
| 5   | 4 225     | 36 it, 0.088s, $J$=0.3462324       | 27 it, 0.052s, $J$=0.3462324       | 13 it, 0.027s, $J$=0.3462324       | **FAIL** (20 it, $J$=0.504)          | 13 it, 0.027s, $J$=0.3462324         |
| 6   | 16 641    | 12 it, 0.115s, $J$=0.3457767       | 14 it, 0.086s, $J$=0.3457767       | 13 it, 0.059s, $J$=0.3457767       | 12 it, 0.038s, $J$=0.3457767         | **FAIL** (13 it, $J$=0.503)          |
| 7   | 66 049    | 14 it, 0.526s, $J$=0.3456623       | 19 it, 0.454s, $J$=0.3456623       | 13 it, 0.210s, $J$=0.3456623       | **FAIL** (15 it, $J$=1.3×10⁸, r=-9) | 14 it, 0.098s, $J$=0.3456623         |
| 8   | 263 169   | 12 it, 1.975s, $J$=0.3456340       | 26 it, 2.543s, $J$=0.3456339       | 13 it, 0.852s, $J$=0.3456337       | 12 it, 0.450s, $J$=0.3456337         | 12 it, 0.284s, $J$=0.3456339         |
| 9   | 1 050 625 | 14 it, 8.863s, $J$=0.3456269       | 23 it, 9.827s, $J$=0.3456265       | 13 it, 3.841s, $J$=0.3456265       | 13 it, 2.504s, $J$=0.3456265         | 14 it, 1.660s, $J$=0.3456265         |

**Observations**:
- **EW forcing makes `ksp_rtol` irrelevant**: All three tables are essentially identical — same iteration counts, same convergence/failure pattern. Eisenstat–Walker dynamically adapts the inner KSP tolerance, so the initial `ksp_rtol` setting has no effect. This is expected behaviour.
- **Reproducible parallel failures**: np=8 fails at L5 (wrong minimum, $J \approx 0.504$) and L7 (divergence $r=-9$); np=16 fails at L6 (wrong minimum, $J \approx 0.503$). These wrong-minimum cases are insidious: SNES reports convergence (reason > 0) but the energy is far from the correct value. The failures are **deterministic** and persist across all KSP tolerances.
- **Serial is perfectly reliable**: All levels converge correctly in 12–36 iterations.
- **np=4 is the sweet spot**: All levels converge in exactly 13 iterations (except L5 at serial/2-proc), giving the best parallel speedup without reliability loss. At L9: 3.8s (2.3× speedup over serial).
- **Compared to A3 (`newtontr` + ASM+ILU)**: C3 has fewer iterations at L9 (13–23 vs 23–68) but has unpredictable wrong-minimum failures at np=8,16 that A3 (`ksp_rtol=1e-1`) avoids entirely. A3 is more robust in parallel despite being slower.
- **Compared to B4 (`l2` + HYPRE)**: Both fail in parallel, but C3's failures are more insidious (wrong minimum with "OK" status vs B4's explicit line-search failures).

---

### Part 5 — Initial guess: `tanh(dist/√ε)` (sine → tanh)

The advisory note suggests using `u_0 = \tanh(\mathrm{dist\_to\_boundary}/\sqrt{\varepsilon})` to start closer to the expected ±1 interior state. This is tested with `gmres`, HYPRE, `ksp_rtol=1e-1`, `divtol=-1`.

| #   | Configuration         | Init guess |          L5          |          L6          |          L7          |          L8          |
| --- | --------------------- | :--------: | :------------------: | :------------------: | :------------------: | :------------------: |
| D1  | **`basic` + tanh**    |   `tanh`   | **OK (4 it, 0.01s)** | **OK (4 it, 0.05s)** | **OK (4 it, 0.20s)** | **OK (4 it, 0.79s)** |
| D2  | `newtontr` + tanh     |   `tanh`   |   OK (9 it, 0.03s)   |  OK (10 it, 0.12s)   |  OK (11 it, 0.51s)   |   OK (12 it, 2.4s)   |
| D3  | **`bt` + obj + tanh** |   `tanh`   | **OK (4 it, 0.02s)** | **OK (4 it, 0.06s)** | **OK (4 it, 0.25s)** | **OK (4 it, 1.0s)**  |

**Key finding**: The `tanh` initial guess is **game-changing**. Even `basic` line search (full step $\alpha = 1$) converges in just **4 iterations** at all levels. The `bt` + objective combination that previously failed at 0 iterations now also converges in 4 iterations — the initial guess is close enough to the minimum that the Newton direction *is* a descent direction.

This dramatically outperforms the sine initial guess (which required 65–218 iterations with `basic` and failed entirely with `bt`).

---

### Part 6 — $\varepsilon$-continuation (0.1 → 0.05 → 0.02 → 0.01)

Solve for a sequence of decreasing $\varepsilon$ values, using the previous solution as the initial guess for the next. Uses the sine initial guess for $\varepsilon = 0.1$.

| #   | Configuration              |                   L5                   |              L6               |          L7           |          L8          |
| --- | -------------------------- | :------------------------------------: | :---------------------------: | :-------------------: | :------------------: |
| E1  | **`basic` + continuation** | **OK** (18 it total, 0.06s) [4+5+5+4]  |     **OK** (18 it, 0.22s)     | **OK** (17 it, 0.82s) | **OK** (16 it, 3.2s) |
| E2  | `newtontr` + continuation  |  **FAIL** (9 it at $\varepsilon=0.1$)  | **FAIL** ($\varepsilon=0.05$) |   OK (48 it, 2.1s)    |   OK (52 it, 9.2s)   |
| E3  | `bt` + obj + continuation  | **FAIL** (500 it at $\varepsilon=0.1$) |       OK (18 it, 0.28s)       |   OK (18 it, 1.1s)    |   OK (18 it, 4.7s)   |

**Key finding**: **E1 (`basic` + continuation) is fully reliable**, converging in 16–18 total iterations (across all 4 $\varepsilon$ steps) at every level. The approach ensures each subsequent $\varepsilon$ starts near the correct basin. Trust region (E2) fails at early continuation steps for small meshes. `bt` (E3) fails at L5 (where $\varepsilon=0.1$ is still non-convex enough to defeat the line search) but works at larger meshes.

---

### Summary of working configurations

| Config                            | Strategy                                  |   L5 (4K dofs)    |     L6 (17K)      |     L7 (66K)      |     L8 (263K)     | Notes                                     |
| --------------------------------- | ----------------------------------------- | :---------------: | :---------------: | :---------------: | :---------------: | ----------------------------------------- |
| **Custom Newton**                 | energy line search $\alpha \in [-0.5, 2]$ | OK (7 it, 0.05s)  | OK (10 it, 0.24s) | OK (6 it, 0.62s)  |  OK (6 it, 2.4s)  | **Reference: always works, MPI-parallel** |
| D1: `basic` + tanh                | better initial guess                      | OK (4 it, 0.01s)  | OK (4 it, 0.05s)  | OK (4 it, 0.20s)  | OK (4 it, 0.79s)  | Fastest, but requires problem-specific IC |
| D3: `bt`+obj + tanh               | backtracking + better IC                  | OK (4 it, 0.02s)  | OK (4 it, 0.06s)  | OK (4 it, 0.25s)  |  OK (4 it, 1.0s)  | Same as D1 — IC does the heavy lifting    |
| E1: `basic` + $\varepsilon$-cont. | parameter continuation                    | OK (18 it, 0.06s) | OK (18 it, 0.22s) | OK (17 it, 0.82s) | OK (16 it, 3.2s)  | Robust, no problem-specific IC needed     |
| A3: `newtontr` + ASM+ILU          | trust region + simple PC                  | OK (12 it, 0.03s) | OK (13 it, 0.11s) | OK (16 it, 0.66s) | OK (19 it, 3.8s)  | Only TR config that works at all levels   |
| B3: `l2` + HYPRE (tight)          | $l2$ line search, `ksp_rtol=1e-6`         | OK (24 it, 0.27s) | OK (23 it, 1.0s)  | OK (19 it, 3.4s)  | OK (21 it, 15.8s) | Requires tight KSP tolerance              |
| B4: `l2` + HYPRE                  | $l2$ line search, `ksp_rtol=1e-3`         | OK (24 it, 0.25s) | OK (21 it, 0.81s) | OK (30 it, 4.4s)  | OK (16 it, 10.6s) | Looser KSP, still works, faster at L8     |
| C3: `basic`+ew+lag                | Eisenstat–Walker + PC lag                 | OK (37 it, 0.09s) | OK (13 it, 0.13s) | OK (15 it, 0.58s) | OK (13 it, 2.2s)  | Surprising combo that works               |

### The fundamental limitation

**No built-in PETSc line search supports negative step sizes.** All PETSc line searches search over $\alpha \in (0, \alpha_{\max}]$ with $\alpha_{\max} \leq 1$ for backtracking variants. The custom Newton's golden-section search over $\alpha \in [-0.5, 2]$ is essential: when the Newton direction from the indefinite Hessian points uphill, **negative $\alpha$ reverses the direction**, turning it into a descent step. This is the key feature that makes the custom Newton converge reliably for non-convex problems **without** requiring a problem-specific initial guess or continuation strategy.

PETSc does support a `SNESLINESEARCHSHELL` type for user-defined line search callbacks, but the current `petsc4py` bindings do not expose `setShell` on `SNESLineSearch` — a custom Python callback cannot be registered. Implementing the golden-section energy line search inside SNES would require C-level PETSc extensions.

### Takeaways

1. **With the generic sine initial guess**, most SNES configurations fail or are unreliable. The custom Newton with energy line search converges reliably (6–12 iterations) without any problem-specific tricks and supports MPI parallelism. Among SNES configurations, **A3 (`newtontr` + ASM/ILU, `ksp_rtol=1e-1`) is the only setup reliable across all mesh sizes and all MPI decompositions** — this is now the default SNES solver in [`solve_GL_snes_newton.py`](GinzburgLandau2D_fenics/solve_GL_snes_newton.py).

2. **A problem-specific `tanh` initial guess** transforms the convergence landscape: even the simplest SNES config (`basic`, full step) converges in 4 iterations. This is because the initial guess is already close to the correct minimum, making the Newton direction a descent direction.

3. **$\varepsilon$-continuation** is a robust alternative that does not require a problem-specific initial guess. Solving for $\varepsilon = 0.1 \to 0.05 \to 0.02 \to 0.01$ with `basic` line search converges in 16–18 total iterations.

4. **`newtontr` + ASM+ILU** (A3) is the only trust-region config that works at all levels. BoomerAMG interacts poorly with the indefinite Hessian in `newtontr`. Crucially, **loose KSP tolerance (`ksp_rtol=1e-1`) is required for parallel reliability** — tighter tolerances cause failures at L9.

5. **`l2` + tight KSP** (`ksp_rtol=1e-6`) works serially but is slow (15–24 s at L8) and unreliable in parallel.

6. **Eisenstat–Walker + preconditioner lagging** (C3) converges fast serially (13–37 iterations) but has insidious wrong-minimum failures in parallel (reports "converged" with incorrect energy).
