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

| lvl | dofs      | time (serial) | iters | J(u)   | time (4-proc) | iters | J(u)   | time (8-proc) | iters | J(u)   | time (16-proc) | iters | J(u)         |
| --- | --------- | ------------- | ----- | ------ | ------------- | ----- | ------ | ------------- | ----- | ------ | -------------- | ----- | ------------ |
| 5   | 4 225     | 0.244         | 71    | 0.3462 | 0.146         | 48    | 0.3462 | 0.094         | 29    | 0.3462 | 0.164          | 43    | 0.3462       |
| 6   | 16 641    | 1.475         | 114   | 0.3458 | **FAIL**      | —     | —      | **FAIL**      | —     | —      | 0.456          | 77    | 0.3458 (1/3) |
| 7   | 66 049    | 7.496         | 148   | 0.3457 | 1.087         | 45    | 0.3457 | **FAIL**      | —     | —      | **FAIL**       | —     | —            |
| 8   | 263 169   | 45.151        | 218   | 0.3456 | 2.559         | 28    | 0.3456 | **FAIL**      | —     | —      | 2.453          | 68    | 0.3456 (1/3) |
| 9   | 1 050 625 | 57.268        | 65    | 0.3456 | **FAIL**      | —     | —      | **FAIL**      | —     | —      | 7.178          | 39    | 0.3456       |

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

---

## Appendix: Comprehensive SNES Configuration Survey

We tested all reasonable PETSc SNES configurations to determine whether any built-in setting can reliably solve the non-convex Ginzburg-Landau problem. All tests used serial execution with `snes_max_it=500`. The correct minimum energy is $J^* \approx 0.3456$. A result is marked **OK** only if the solver converges (`reason > 0`) *and* reaches this minimum.

Unless otherwise noted, the default initial guess is $u_0(x,y) = \sin(\pi(x-1)/2)\sin(\pi(y-1)/2)$ (a smooth bump that satisfies the Dirichlet BC but is far from the true solution).

Test script: [`tmp_work/test_snes_comprehensive.py`](tmp_work/test_snes_comprehensive.py)

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

---

### Part 4 — Eisenstat–Walker and preconditioner lagging (sine initial guess)

Using `newtonls` + `basic`, `ksp_rtol=1e-1`, HYPRE, `divtol=-1`.

| #   | Configuration                       | Key options               |              L5              |                  L6                   |          L7           |          L8          |
| --- | ----------------------------------- | ------------------------- | :--------------------------: | :-----------------------------------: | :-------------------: | :------------------: |
| C1  | `basic` + `ksp_ew`                  | Eisenstat–Walker          | **FAIL** (500 it, $J=0.857$) |         **FAIL** ($J=0.856$)          | **FAIL** ($J=0.873$)  | **FAIL** ($J=0.844$) |
| C2  | `basic` + `lag_pc=2`                | reuse PC every 2 steps    |         OK (105 it)          | **FAIL** (42 it, $J=1.1 \times 10^6$) |      OK (46 it)       |      OK (62 it)      |
| C3  | **`basic` + `ksp_ew` + `lag_pc=2`** | `fgmres`, `ksp_rtol=1e-6` |    **OK (37 it, 0.09s)**     |         **OK (13 it, 0.13s)**         | **OK (15 it, 0.58s)** | **OK (13 it, 2.2s)** |

**Key finding**: Eisenstat–Walker **alone** (C1) is catastrophic with `ksp_rtol=1e-1` — the initial loose solve never produces a useful Newton direction. But **combined with `lag_pc=2` and tight `ksp_rtol=1e-6`** (C3), it converges reliably and fast (13–37 iterations). Lagging alone (C2) is unreliable.

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

1. **With the generic sine initial guess**, SNES fails or is unreliable with most configurations. The custom Newton with energy line search is the only solver that converges reliably (6–12 iterations) without any problem-specific tricks. It also supports MPI parallelism.

2. **A problem-specific `tanh` initial guess** transforms the convergence landscape: even the simplest SNES config (`basic`, full step) converges in 4 iterations. This is because the initial guess is already close to the correct minimum, making the Newton direction a descent direction.

3. **$\varepsilon$-continuation** is a robust alternative that does not require a problem-specific initial guess. Solving for $\varepsilon = 0.1 \to 0.05 \to 0.02 \to 0.01$ with `basic` line search converges in 16–18 total iterations.

4. **`newtontr` + ASM+ILU** is the only trust-region config that works at all levels. BoomerAMG interacts poorly with the indefinite Hessian in `newtontr`.

5. **`l2` + tight KSP** (`ksp_rtol=1e-6`) works but is slow (15–24 s at L8) because the tight KSP tolerance means expensive linear solves.

6. **Eisenstat–Walker + preconditioner lagging** (C3) is a surprising success: the combination of adaptive KSP tolerance and PC reuse produces a fast, reliable solver (13–37 iterations).
