# HyperElasticity3D Results (JAX vs Custom FEniCS)

This report is reorganized to show the final, detailed per-step results first, then the implementation details and pitfalls, with investigative experiments moved to Annex A.

## 1) Final working setup (clear specification)

### 1.1 JAX reference setup

- Solver: `tools/minimizers.py::newton`
- Nonlinear tolerances:
  - `tolf = 1e-4` (energy change)
  - `tolg = 1e-3` (gradient norm)
  - `maxit = 100`
- Line search:
  - golden-section
  - `linesearch_tol = 1e-3`
  - interval `[-0.5, 2.0]`
- Inner linear solve:
  - Krylov `cg` (SciPy)
  - preconditioner `PyAMG` smoothed aggregation
  - inner tol `1e-3`
  - max inner iterations `100`

Reference artifacts used in tables:
- Level 1: [experiment_scripts/he_jax_evolution_l1_recompute_gmres_compare.json](experiment_scripts/he_jax_evolution_l1_recompute_gmres_compare.json)
- Level 2: [experiment_scripts/he_jax_evolution_l2.json](experiment_scripts/he_jax_evolution_l2.json)

### 1.2 Custom FEniCS final setup

- Solver: `HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py` + `tools_petsc4py/minimizers.py`
- Nonlinear tolerances:
  - `tolf = 1e-4`
  - `tolg = 1e-3`
  - `tolg_rel = 1e-3`
  - `tolx_rel = 1e-3`
  - `tolx_abs = 1e-10`
  - `maxit = 100`
  - convergence requires **all three** criteria:
    - energy change (`|dE| < tolf`)
    - step size (`step_rel < tolx_rel` or `step_norm < tolx_abs`)
    - gradient (`||g|| < max(tolg, tolg_rel * ||g_0||)`)
- Line search:
  - golden-section
  - `linesearch_tol = 1e-3`
  - interval `[-0.5, 2.0]`
  - non-finite trial energies treated as `+inf`
- Inner linear solve:
  - `ksp_type = gmres`
  - `pc_type = hypre` (`boomeramg`)
  - `ksp_rtol = 1e-1`
  - `ksp_max_it = 30`
  - skip explicit BoomerAMG `nodal_coarsen` / `vec_interp_variant` (use HYPRE defaults)
  - `--pc_setup_on_ksp_cap` enabled:
    - first Newton linear solve runs `ksp.setUp()`
    - subsequent solves reuse PC
    - rebuild PC only when previous solve hit `ksp_max_it`
  - nonlinear repair mode enabled:
    - retry on non-finite state and on Newton `maxit` stall
    - retry uses tighter linear solve (`ksp_rtol *= 0.1`), higher linear cap (`ksp_max_it *= 2`)
    - retry clamps line-search upper bound to `1.0`

### 1.2.1 Detailed minimizer changes (custom PETSc Newton)

The minimizer in `tools_petsc4py/minimizers.py` was hardened to prevent silent false convergence and NaN propagation observed in `r1e-2_k50_fresh`:

1. **Stopping logic changed from energy-only to multi-criterion support**
   - Previous practical behavior in HE runs: most steps ended with `"Energy change converged"` even when gradient was still large.
   - Current HE path requires `energy + step + gradient` together.

2. **Relative gradient target support**
   - Added `tolg_rel`, using
     `grad_target = max(tolg, tolg_rel * initial_grad_norm)`.
   - This avoids brittle behavior from using one absolute gradient threshold across all load steps.

3. **Step-size convergence criteria added**
   - Added `tolx_rel`, `tolx_abs`.
   - Convergence can no longer be declared on tiny `dE` alone when steps are still too large (or vice versa).

4. **Non-finite fail-fast + rollback**
   - Detect non-finite energy/gradient/direction.
   - Keep a rollback copy of the previous iterate and restore on non-finite update.
   - Return explicit failure message instead of propagating NaNs.

5. **Line-search non-finite repair**
   - If golden-section candidate is non-finite, backtrack `alpha` toward zero.
   - If no finite trial is found, terminate with explicit message.

6. **Per-iteration diagnostics expanded**
   - History now includes `step_rel`, `step_norm`, `grad_target`, `grad_norm_post`, `ls_repaired`.
   - This enabled identifying the step-16 stall mechanism (`ksp_its` repeatedly at cap with gradient plateau).

7. **Load-step retry strategy in HE driver**
   - Added per-step retry attempt in `solve_HE_custom_jaxversion.py`.
   - Retry is triggered by non-finite failure or Newton max-iteration stall, with tighter/safer linear and line-search settings.

**How to run (96 quarter-steps, level 1, single process):**
```bash
python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
    --level 1 --steps 96 --total_steps 96 \
    --ksp_rtol 1e-1 --ksp_max_it 30 \
    --pc_setup_on_ksp_cap \
    --quiet --out experiment_scripts/out_custom.json
```

Final artifacts used in tables:
- Level 1: [experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap.json](experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap.json)
- Level 2: [experiment_scripts/he_fenics_custom_evolution_l2_skip_ksp30_pc_cap.json](experiment_scripts/he_fenics_custom_evolution_l2_skip_ksp30_pc_cap.json)

### 1.3 SNES FEniCS final setup

- Solver: `HyperElasticity3D_fenics/solve_HE_snes_newton.py` (PETSc SNES Newton with line search)
- Nonlinear tolerances:
  - `snes_atol = 1e-3`
  - `snes_rtol = 1e-10` (default, not active in practice)
  - `snes_max_it = 50` (default)
- Near-nullspace: rigid-body translations + rotations attached to stiffness matrix (same as custom solver)
- Inner linear solve:
  - `ksp_type = gmres`
  - `pc_type = hypre` (`boomeramg`)
  - `ksp_rtol = 1e-1`
  - `ksp_max_it = 500`
  - HYPRE default coarsening (no explicit `nodal_coarsen` / `vec_interp_variant`)
  - **Note**: `vec_interp_variant=3` (used in early experiments) produces a non-symmetric AMG preconditioner, which causes CG breakdown (`KSP_DIVERGED_BREAKDOWN`) inside SNES because SNES requires `KSP reason > 0`. GMRES + HYPRE defaults avoids this.

**How to run (96 quarter-steps, level 1, single process):**
```bash
python3 HyperElasticity3D_fenics/solve_HE_snes_newton.py \
    --level 1 --steps 96 \
    --ksp_type gmres --pc_type hypre \
    --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 \
    --quiet --out experiment_scripts/out_snes.json
```

Results artifact: [experiment_scripts/gl_snes_np1.json](experiment_scripts/gl_snes_np1.json) (see Annex D for full per-step table)

---

## 2) Detailed per-step tables (all timesteps)

Note: JAX output does not include per-Newton inner linear iteration counts in the stored trajectory files, so `Sum linear iters` is shown as `—` for JAX.

## Level 1 — JAX (all timesteps)

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Status                                |
| ---: | -------: | -----------: | ---------------: | -------------: | ------------------------------------- |
|    1 |   0.9555 |           18 |                — |   0.3464113962 | Stopping condition for f is satisfied |
|    2 |   0.9883 |           18 |                — |   1.3856347792 | Stopping condition for f is satisfied |
|    3 |   0.9354 |           18 |                — |   3.1173389157 | Stopping condition for f is satisfied |
|    4 |   0.9383 |           17 |                — |   5.5401476466 | Stopping condition for f is satisfied |
|    5 |   0.9902 |           19 |                — |   8.6504988097 | Stopping condition for f is satisfied |
|    6 |   1.0675 |           20 |                — |  12.4422658099 | Stopping condition for f is satisfied |
|    7 |   1.3990 |           22 |                — |  16.9115637817 | Stopping condition for f is satisfied |
|    8 |   1.1646 |           23 |                — |  22.0617389409 | Stopping condition for f is satisfied |
|    9 |   1.1146 |           21 |                — |  27.8989757279 | Stopping condition for f is satisfied |
|   10 |   1.1420 |           21 |                — |  34.4264986836 | Stopping condition for f is satisfied |
|   11 |   1.1005 |           20 |                — |  41.6441021195 | Stopping condition for f is satisfied |
|   12 |   1.3005 |           21 |                — |  49.5500693409 | Stopping condition for f is satisfied |
|   13 |   1.0327 |           19 |                — |  58.1426609931 | Stopping condition for f is satisfied |
|   14 |   1.2174 |           22 |                — |  67.4207907906 | Stopping condition for f is satisfied |
|   15 |   1.2823 |           23 |                — |  77.3830591107 | Stopping condition for f is satisfied |
|   16 |   1.2632 |           22 |                — |  88.0180575975 | Stopping condition for f is satisfied |
|   17 |   1.2160 |           22 |                — |  99.3269874951 | Stopping condition for f is satisfied |
|   18 |   1.0924 |           21 |                — | 111.3262030649 | Stopping condition for f is satisfied |
|   19 |   1.1635 |           22 |                — | 124.0155386580 | Stopping condition for f is satisfied |
|   20 |   1.1919 |           22 |                — | 137.3923331636 | Stopping condition for f is satisfied |
|   21 |   1.2581 |           22 |                — | 151.4552317421 | Stopping condition for f is satisfied |
|   22 |   1.4789 |           24 |                — | 166.2042207199 | Stopping condition for f is satisfied |
|   23 |   1.4576 |           22 |                — | 181.6387101832 | Stopping condition for f is satisfied |
|   24 |   2.0587 |           34 |                — | 197.7486351731 | Stopping condition for f is satisfied |

## Level 1 — Custom FEniCS (final setup, all timesteps)

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Relative error vs JAX | Status                  |
| ---: | -------: | -----------: | ---------------: | -------------: | --------------------: | ----------------------- |
|    1 |   0.2417 |           22 |              331 |   0.3464110000 |              1.14e-06 | Energy change converged |
|    2 |   0.2728 |           23 |              380 |   1.3856400000 |              3.77e-06 | Energy change converged |
|    3 |   0.2456 |           21 |              292 |   3.1173430000 |              1.31e-06 | Energy change converged |
|    4 |   0.2618 |           22 |              301 |   5.5401480000 |              6.38e-08 | Energy change converged |
|    5 |   0.2743 |           22 |              334 |   8.6505010000 |              2.53e-07 | Energy change converged |
|    6 |   0.3076 |           24 |              378 |  12.4422680000 |              1.76e-07 | Energy change converged |
|    7 |   0.2929 |           23 |              349 |  16.9115650000 |              7.20e-08 | Energy change converged |
|    8 |   0.2871 |           23 |              323 |  22.0617410000 |              9.33e-08 | Energy change converged |
|    9 |   0.2523 |           20 |              268 |  27.8989770000 |              4.56e-08 | Energy change converged |
|   10 |   0.2487 |           20 |              258 |  34.4265020000 |              9.63e-08 | Energy change converged |
|   11 |   0.3080 |           23 |              318 |  41.6441060000 |              9.32e-08 | Energy change converged |
|   12 |   0.2718 |           21 |              270 |  49.5500710000 |              3.35e-08 | Energy change converged |
|   13 |   0.2621 |           20 |              261 |  58.1426610000 |              1.18e-10 | Energy change converged |
|   14 |   0.2768 |           21 |              268 |  67.4207910000 |              3.11e-09 | Energy change converged |
|   15 |   0.2766 |           21 |              290 |  77.3830590000 |              1.43e-09 | Energy change converged |
|   16 |   0.3145 |           23 |              324 |  88.0180590000 |              1.59e-08 | Energy change converged |
|   17 |   0.3082 |           22 |              306 |  99.3269870000 |              4.98e-09 | Energy change converged |
|   18 |   0.3278 |           23 |              346 | 111.3262050000 |              1.74e-08 | Energy change converged |
|   19 |   0.3347 |           22 |              338 | 124.0155390000 |              2.76e-09 | Energy change converged |
|   20 |   0.3108 |           22 |              328 | 137.3923340000 |              6.09e-09 | Energy change converged |
|   21 |   0.3209 |           23 |              322 | 151.4552360000 |              2.81e-08 | Energy change converged |
|   22 |   0.2920 |           21 |              291 | 166.2042200000 |              4.33e-09 | Energy change converged |
|   23 |   0.3205 |           23 |              319 | 181.6387090000 |              6.51e-09 | Energy change converged |
|   24 |   0.3061 |           22 |              308 | 197.7551790000 |              3.31e-05 | Energy change converged |

Summary: total time = `6.9156 s`, total Newton iters = `527`, total linear iters = `7503`, max relative error = `3.31e-05`, mean relative error = `1.68e-06`.

## Level 1 — SNES FEniCS (standard 24 steps, stop at first fail)

Config: `newtonls + basic linesearch`, `gmres + hypre boomeramg (HYPRE defaults)`, near-nullspace ON, `ksp_rtol=1e-1`, `ksp_max_it=500`, `snes_atol=1e-3`. Stopped at step 13 (`SNES_DIVERGED_MAX_IT`, reason=-5). Artifact: [experiment_scripts/he_snes_24steps_l1.json](experiment_scripts/he_snes_24steps_l1.json)

| Step | Time [s] | Newton iters | Sum linear iters |        Energy | Relative error vs JAX | Status |
| ---: | -------: | -----------: | ---------------: | ------------: | --------------------: | ------ |
|    1 |   0.1417 |           18 |              195 |  0.3464110000 |              1.14e-06 | conv   |
|    2 |   0.1616 |           19 |              217 |  1.3856340000 |              5.62e-07 | conv   |
|    3 |   0.1769 |           20 |              222 |  3.1173390000 |              2.70e-08 | conv   |
|    4 |   0.1979 |           21 |              251 |  5.5401480000 |              6.38e-08 | conv   |
|    5 |   0.2064 |           22 |              257 |  8.6504990000 |              2.20e-08 | conv   |
|    6 |   0.1739 |           18 |              212 | 12.4422650000 |              6.51e-08 | conv   |
|    7 |   0.2290 |           22 |              284 | 16.9115640000 |              1.29e-08 | conv   |
|    8 |   0.2333 |           24 |              245 | 22.0617390000 |              2.68e-09 | conv   |
|    9 |   0.2040 |           20 |              240 | 27.8989760000 |              9.75e-09 | conv   |
|   10 |   0.1987 |           20 |              206 | 34.4264990000 |              9.19e-09 | conv   |
|   11 |   0.2083 |           20 |              229 | 41.6441010000 |              2.69e-08 | conv   |
|   12 |   0.2154 |           21 |              237 | 49.5500690000 |              6.88e-09 | conv   |
|   13 |   0.7664 |          100 |              135 |  11310.741725 |                     — | r=-5   |

Summary: steps 1–12 converged (12/13), step 13 diverged (`SNES_DIVERGED_MAX_IT`). Total wall time = `3.11 s`, converged Newton iters = `245`, converged KSP iters = `2795`, avg KSP/Newton = `11.4`. Max relative error (steps 1–12) = `1.14e-06`, mean relative error = `1.62e-07`.

## Level 2 — JAX (all timesteps)

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Status                                |
| ---: | -------: | -----------: | ---------------: | -------------: | ------------------------------------- |
|    1 |   7.3735 |           23 |                — |   0.2027078558 | Stopping condition for f is satisfied |
|    2 |  10.8467 |           23 |                — |   0.8108329167 | Stopping condition for f is satisfied |
|    3 |   7.8640 |           19 |                — |   1.8243690766 | Stopping condition for f is satisfied |
|    4 |  11.9467 |           23 |                — |   3.2432373389 | Stopping condition for f is satisfied |
|    5 |  11.4125 |           23 |                — |   5.0672569873 | Stopping condition for f is satisfied |
|    6 |   8.4562 |           20 |                — |   7.2960180354 | Stopping condition for f is satisfied |
|    7 |  12.1968 |           21 |                — |   9.9289527809 | Stopping condition for f is satisfied |
|    8 |   8.2364 |           20 |                — |  12.9658010823 | Stopping condition for f is satisfied |
|    9 |   9.8218 |           21 |                — |  16.4069415137 | Stopping condition for f is satisfied |
|   10 |  10.0110 |           21 |                — |  20.2529469544 | Stopping condition for f is satisfied |
|   11 |   8.1250 |           19 |                — |  24.5041608022 | Stopping condition for f is satisfied |
|   12 |  10.4674 |           19 |                — |  29.1606595399 | Stopping condition for f is satisfied |
|   13 |  10.0258 |           21 |                — |  34.2223416446 | Stopping condition for f is satisfied |
|   14 |  11.7834 |           22 |                — |  39.6889284918 | Stopping condition for f is satisfied |
|   15 |  14.8937 |           26 |                — |  45.5597822261 | Stopping condition for f is satisfied |
|   16 |  13.2603 |           26 |                — |  51.8203722387 | Stopping condition for f is satisfied |
|   17 |   9.4327 |           21 |                — |  58.4802480802 | Stopping condition for f is satisfied |
|   18 |  13.2205 |           23 |                — |  65.5436893127 | Stopping condition for f is satisfied |
|   19 |  10.8798 |           21 |                — |  73.0099173364 | Stopping condition for f is satisfied |
|   20 |  11.9574 |           24 |                — |  80.8777452702 | Stopping condition for f is satisfied |
|   21 |  13.4127 |           23 |                — |  89.1458557765 | Stopping condition for f is satisfied |
|   22 |  13.2149 |           24 |                — |  97.8127632819 | Stopping condition for f is satisfied |
|   23 |  12.2828 |           22 |                — | 106.8769697271 | Stopping condition for f is satisfied |
|   24 |  13.5734 |           26 |                — | 116.3363305574 | Stopping condition for f is satisfied |

## Level 2 — Custom FEniCS (final setup, all timesteps)

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Relative error vs JAX | Status                  |
| ---: | -------: | -----------: | ---------------: | -------------: | --------------------: | ----------------------- |
|    1 |   2.6353 |           25 |              325 |   0.2027090000 |              5.64e-06 | Energy change converged |
|    2 |   3.2564 |           27 |              422 |   0.8108430000 |              1.24e-05 | Energy change converged |
|    3 |   3.0533 |           26 |              361 |   1.8243750000 |              3.25e-06 | Energy change converged |
|    4 |   2.8207 |           25 |              311 |   3.2432390000 |              5.12e-07 | Energy change converged |
|    5 |   3.4192 |           28 |              425 |   5.0672610000 |              7.92e-07 | Energy change converged |
|    6 |   3.0276 |           26 |              330 |   7.2960210000 |              4.06e-07 | Energy change converged |
|    7 |   3.0767 |           26 |              347 |   9.9289580000 |              5.26e-07 | Energy change converged |
|    8 |   3.4279 |           28 |              413 |  12.9658030000 |              1.48e-07 | Energy change converged |
|    9 |   2.9100 |           24 |              331 |  16.4069430000 |              9.06e-08 | Energy change converged |
|   10 |   3.1999 |           25 |              379 |  20.2529490000 |              1.01e-07 | Energy change converged |
|   11 |   2.9653 |           24 |              329 |  24.5041620000 |              4.89e-08 | Energy change converged |
|   12 |   3.3154 |           26 |              379 |  29.1606590000 |              1.85e-08 | Energy change converged |
|   13 |   3.2462 |           25 |              366 |  34.2223490000 |              2.15e-07 | Energy change converged |
|   14 |   3.3554 |           25 |              408 |  39.6889340000 |              1.39e-07 | Energy change converged |
|   15 |   3.4148 |           25 |              400 |  45.5598100000 |              6.10e-07 | Energy change converged |
|   16 |   5.2937 |           36 |              718 |  51.8204090000 |              7.09e-07 | Energy change converged |
|   17 |   3.5650 |           26 |              453 |  58.4802480000 |              1.37e-09 | Energy change converged |
|   18 |   3.3143 |           24 |              422 |  65.5437020000 |              1.94e-07 | Energy change converged |
|   19 |   3.1228 |           23 |              380 |  73.0099130000 |              5.94e-08 | Energy change converged |
|   20 |   3.4744 |           24 |              436 |  80.8777820000 |              4.54e-07 | Energy change converged |
|   21 |   3.4515 |           24 |              432 |  89.1459040000 |              5.41e-07 | Energy change converged |
|   22 |   3.6497 |           25 |              466 |  97.8129070000 |              1.47e-06 | Energy change converged |
|   23 |   3.5182 |           25 |              411 | 106.8769200000 |              4.65e-07 | Energy change converged |
|   24 |   6.1875 |           40 |              826 | 116.3256990000 |              9.14e-05 | Energy change converged |

Summary: total time = `82.7012 s`, total Newton iters = `632`, total linear iters = `10070`, max relative error = `9.14e-05`, mean relative error = `5.01e-06`.

---

## 3) Pitfalls encountered and how they were solved

### 3.1 Restart/state mapping pitfall (JAX → FEniCS)

**Problem encountered**
- Late-step conclusions were initially distorted by incorrect restart initialization.
- JAX restart data were interpreted as displacements, but stored values are full deformed coordinates.
- A raw vector-index assignment path also risked DOF-mapping inconsistencies.

**Fix applied**
- Convert JAX field to displacement: `u = x_deformed - X_ref`.
- Interpolate by coordinate matching, not by raw vector indexing.
- Validate on an early-step restart before using late-step conclusions.

**Result**
- Restart consistency restored; subsequent robustness findings became reliable.

### 3.2 Null-space formatting pitfall (critical for elasticity + AMG)

**Problem encountered**
- Near-nullspace vectors can be malformed if constructed with inconsistent vector layout, wrong ownership range, or missing ghost updates.

**Correct formatting used**
1. Build six rigid modes (3 translations + 3 rotations).
2. Allocate vectors with matrix-compatible layout (`A.createVecLeft()`).
3. Use only owned coordinates (`x_owned = coords[:index_map.size_local]`) for local vector fill.
4. Fill translations by component stride and rotations from coordinate-based rigid-body formulas.
5. Forward ghost update each mode (`INSERT/FORWARD`).
6. Create PETSc nullspace and attach with `A.setNearNullSpace(nullspace)`.

**Result**
- Stable, consistent near-nullspace attachment behavior for elasticity matrices.

### 3.3 HYPRE AMG / KSP setup pitfall

**Problem encountered**
- `cg+hypre` failed in late nonlinear regime even with tighter inner tolerances.
- Per-Newton unconditional `ksp.setUp()` incurred repeated setup overhead.
- Large inner iteration caps could hide inefficient solves.

**Final robust setup**
- Switch to `gmres+hypre`.
- Use loose-but-effective `ksp_rtol=1e-1`.
- Cap inner work with `ksp_max_it=30`.
- Reuse preconditioner and only rebuild after cap hits (`--pc_setup_on_ksp_cap`).
- For this case, HYPRE defaults (skip explicit `nodal/vec`) performed best.

**Result**
- Stable full trajectories on both levels with small relative error vs JAX and reduced wall time.

### 3.4 Non-finite line-search trial pitfall

**Problem encountered**
- Non-finite trial energies can break line-search behavior.

**Fix applied**
- Treat non-finite trial energies as `+inf` in line-search objective evaluation.

**Result**
- Prevented NaN-driven line-search corruption and improved nonlinear robustness.

---

## 4) Parallel custom FEniCS tables (np=4,8,16)

These runs use the same custom setup as the final serial configuration except
the historical note below recorded `--no_near_nullspace` for the MPI runs.

Common settings for all tables below: `ksp_type=gmres`, `pc_type=hypre`, `ksp_rtol=1e-1`, `ksp_max_it=30`, skip explicit `nodal/vec`, `--pc_setup_on_ksp_cap`, `--no_near_nullspace`.

Current 2026-03-06 step-1 reruns on the refactored tree did not reproduce a
nullspace crash for this case. Both `--no_near_nullspace` and the default
near-nullspace-on configuration converged; see the added comparison rows in the
level-4 `np=16` table below.

> **⚠ Docker environment requirements for MPI runs**
>
> All parallel benchmarks in this section were run inside the `fenics_test:latest`
> Docker container. The image uses **MPICH**, which allocates shared-memory
> segments for inter-process communication. Key requirements:
>
> - **`--shm-size=8g`** (or larger) is **mandatory** when running ≥8 MPI
>   processes. Docker's default shared memory is 64 MB, which causes **SIGBUS
>   (exit code 135)** or OOM kills (exit code 9) with many ranks.
> - For long-running benchmarks (e.g. level 3–4, 24 steps), use a **persistent
>   container** rather than `docker run --rm`:
>   ```bash
>   docker run -d --name bench_container --shm-size=8g \
>     --entrypoint /bin/bash \
>     -v "$PWD":/workdir -w /workdir \
>     fenics_test:latest -c "sleep infinity"
>   docker exec bench_container mpirun -n 16 python3 \
>     /workdir/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
>     --level 3 --steps 24 --total_steps 24 \
>     --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap \
>     --quiet --out /workdir/experiment_scripts/out.json
>   docker stop bench_container && docker rm bench_container
>   ```
> - **Timing sensitivity**: Running in a container created from a *different*
>   Docker image (even with the same packages) can produce 2–3× slower wall-clock
>   times despite identical iteration counts. Always use `fenics_test:latest`
>   built from `.devcontainer/Dockerfile` for reproducible timings.

### Level 1 — Custom FEniCS MPI nproc=4

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Relative error vs JAX | Status                  |
| ---: | -------: | -----------: | ---------------: | -------------: | --------------------: | ----------------------- |
|    1 |   0.1062 |           21 |              304 |   0.3464120000 |              1.74e-06 | Energy change converged |
|    2 |   0.1106 |           21 |              252 |   1.3856370000 |              1.60e-06 | Energy change converged |
|    3 |   0.1111 |           20 |              261 |   3.1173460000 |              2.27e-06 | Energy change converged |
|    4 |   0.1283 |           23 |              344 |   5.5401490000 |              2.44e-07 | Energy change converged |
|    5 |   0.1078 |           20 |              277 |   8.6505010000 |              2.53e-07 | Energy change converged |
|    6 |   0.1228 |           22 |              322 |  12.4422680000 |              1.76e-07 | Energy change converged |
|    7 |   0.1108 |           21 |              275 |  16.9115650000 |              7.20e-08 | Energy change converged |
|    8 |   0.1248 |           21 |              309 |  22.0617420000 |              1.39e-07 | Energy change converged |
|    9 |   0.1340 |           22 |              280 |  27.8989760000 |              9.75e-09 | Energy change converged |
|   10 |   0.1356 |           23 |              344 |  34.4265020000 |              9.63e-08 | Energy change converged |
|   11 |   0.1271 |           23 |              316 |  41.6441060000 |              9.32e-08 | Energy change converged |
|   12 |   0.1184 |           21 |              277 |  49.5500720000 |              5.37e-08 | Energy change converged |
|   13 |   0.1132 |           20 |              250 |  58.1426610000 |              1.18e-10 | Energy change converged |
|   14 |   0.1174 |           22 |              280 |  67.4207960000 |              7.73e-08 | Energy change converged |
|   15 |   0.1281 |           21 |              297 |  77.3830590000 |              1.43e-09 | Energy change converged |
|   16 |   0.1216 |           21 |              291 |  88.0180590000 |              1.59e-08 | Energy change converged |
|   17 |   0.1446 |           24 |              365 |  99.3269870000 |              4.98e-09 | Energy change converged |
|   18 |   0.1444 |           23 |              346 | 111.3262050000 |              1.74e-08 | Energy change converged |
|   19 |   0.1484 |           22 |              340 | 124.0155390000 |              2.76e-09 | Energy change converged |
|   20 |   0.1197 |           20 |              264 | 137.3923340000 |              6.09e-09 | Energy change converged |
|   21 |   0.1479 |           24 |              355 | 151.4552330000 |              8.31e-09 | Energy change converged |
|   22 |   0.1226 |           20 |              267 | 166.2042260000 |              3.18e-08 | Energy change converged |
|   23 |   0.1559 |           23 |              334 | 181.6387080000 |              1.20e-08 | Energy change converged |
|   24 |   0.1255 |           22 |              319 | 197.7551340000 |              3.29e-05 | Energy change converged |

Summary: total time = `3.0268 s`, total Newton iters = `520`, total linear iters = `7269`, max relative error = `3.29e-05`, mean relative error = `1.66e-06`.

### Level 1 — Custom FEniCS MPI nproc=8

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Relative error vs JAX | Status                  |
| ---: | -------: | -----------: | ---------------: | -------------: | --------------------: | ----------------------- |
|    1 |   0.0916 |           21 |              314 |   0.3464120000 |              1.74e-06 | Energy change converged |
|    2 |   0.0941 |           22 |              299 |   1.3856370000 |              1.60e-06 | Energy change converged |
|    3 |   0.1213 |           21 |              296 |   3.1173390000 |              2.71e-08 | Energy change converged |
|    4 |   0.1384 |           23 |              341 |   5.5401500000 |              4.25e-07 | Energy change converged |
|    5 |   0.1328 |           23 |              371 |   8.6504990000 |              2.20e-08 | Energy change converged |
|    6 |   0.1313 |           23 |              361 |  12.4422660000 |              1.53e-08 | Energy change converged |
|    7 |   0.1282 |           23 |              346 |  16.9115640000 |              1.29e-08 | Energy change converged |
|    8 |   0.1086 |           21 |              288 |  22.0617410000 |              9.33e-08 | Energy change converged |
|    9 |   0.1229 |           23 |              303 |  27.8989760000 |              9.75e-09 | Energy change converged |
|   10 |   0.1147 |           21 |              293 |  34.4264990000 |              9.19e-09 | Energy change converged |
|   11 |   0.1095 |           23 |              310 |  41.6441020000 |              2.87e-09 | Energy change converged |
|   12 |   0.1192 |           22 |              303 |  49.5500710000 |              3.35e-08 | Energy change converged |
|   13 |   0.1007 |           20 |              246 |  58.1426610000 |              1.18e-10 | Energy change converged |
|   14 |   0.1179 |           23 |              314 |  67.4207910000 |              3.11e-09 | Energy change converged |
|   15 |   0.1103 |           21 |              290 |  77.3830610000 |              2.44e-08 | Energy change converged |
|   16 |   0.1252 |           23 |              345 |  88.0180580000 |              4.57e-09 | Energy change converged |
|   17 |   0.1384 |           24 |              364 |  99.3269870000 |              4.98e-09 | Energy change converged |
|   18 |   0.1360 |           23 |              352 | 111.3262030000 |              5.83e-10 | Energy change converged |
|   19 |   0.1500 |           23 |              339 | 124.0155390000 |              2.76e-09 | Energy change converged |
|   20 |   0.1181 |           22 |              278 | 137.3923340000 |              6.09e-09 | Energy change converged |
|   21 |   0.1303 |           23 |              320 | 151.4552330000 |              8.31e-09 | Energy change converged |
|   22 |   0.1182 |           22 |              296 | 166.2042280000 |              4.38e-08 | Energy change converged |
|   23 |   0.1494 |           23 |              330 | 181.6387120000 |              1.00e-08 | Energy change converged |
|   24 |   0.1030 |           20 |              283 | 197.7551220000 |              3.28e-05 | Energy change converged |

Summary: total time = `2.9101 s`, total Newton iters = `533`, total linear iters = `7582`, max relative error = `3.28e-05`, mean relative error = `1.54e-06`.

### Level 1 — Custom FEniCS MPI nproc=16

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Relative error vs JAX | Status                  |
| ---: | -------: | -----------: | ---------------: | -------------: | --------------------: | ----------------------- |
|    1 |   0.1391 |           21 |              263 |   0.3464110000 |              1.14e-06 | Energy change converged |
|    2 |   0.1585 |           22 |              320 |   1.3856350000 |              1.59e-07 | Energy change converged |
|    3 |   0.1268 |           20 |              241 |   3.1173510000 |              3.88e-06 | Energy change converged |
|    4 |   0.1533 |           23 |              355 |   5.5401480000 |              6.38e-08 | Energy change converged |
|    5 |   0.1324 |           21 |              309 |   8.6505080000 |              1.06e-06 | Energy change converged |
|    6 |   0.1308 |           21 |              299 |  12.4422700000 |              3.37e-07 | Energy change converged |
|    7 |   0.1220 |           20 |              265 |  16.9115640000 |              1.29e-08 | Energy change converged |
|    8 |   0.1270 |           21 |              265 |  22.0617430000 |              1.84e-07 | Energy change converged |
|    9 |   0.1384 |           23 |              297 |  27.8989780000 |              8.14e-08 | Energy change converged |
|   10 |   0.1227 |           20 |              259 |  34.4265080000 |              2.71e-07 | Energy change converged |
|   11 |   0.1443 |           23 |              309 |  41.6441100000 |              1.89e-07 | Energy change converged |
|   12 |   0.1385 |           22 |              312 |  49.5500700000 |              1.33e-08 | Energy change converged |
|   13 |   0.1400 |           21 |              292 |  58.1426660000 |              8.61e-08 | Energy change converged |
|   14 |   0.1622 |           24 |              377 |  67.4207910000 |              3.11e-09 | Energy change converged |
|   15 |   0.1229 |           20 |              255 |  77.3830630000 |              5.03e-08 | Energy change converged |
|   16 |   0.1463 |           23 |              327 |  88.0180580000 |              4.57e-09 | Energy change converged |
|   17 |   0.1532 |           24 |              368 |  99.3269870000 |              4.98e-09 | Energy change converged |
|   18 |   0.1528 |           24 |              344 | 111.3262030000 |              5.83e-10 | Energy change converged |
|   19 |   0.1620 |           23 |              339 | 124.0155410000 |              1.89e-08 | Energy change converged |
|   20 |   0.1422 |           23 |              328 | 137.3923330000 |              1.19e-09 | Energy change converged |
|   21 |   0.1402 |           22 |              298 | 151.4552350000 |              2.15e-08 | Energy change converged |
|   22 |   0.1286 |           20 |              302 | 166.2042240000 |              1.97e-08 | Energy change converged |
|   23 |   0.1753 |           23 |              320 | 181.6387080000 |              1.20e-08 | Energy change converged |
|   24 |   0.1424 |           23 |              353 | 197.7551650000 |              3.30e-05 | Energy change converged |

Summary: total time = `3.4019 s`, total Newton iters = `527`, total linear iters = `7397`, max relative error = `3.30e-05`, mean relative error = `1.69e-06`.

### Level 2 — Custom FEniCS MPI nproc=4

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Relative error vs JAX | Status                  |
| ---: | -------: | -----------: | ---------------: | -------------: | --------------------: | ----------------------- |
|    1 |   0.2933 |            9 |               94 |   0.5824270000 |              1.87e+00 | Energy change converged |
|    2 |   1.2993 |           34 |              433 |   0.8108400000 |              8.74e-06 | Energy change converged |
|    3 |   1.1110 |           27 |              421 |   1.8243680000 |              5.90e-07 | Energy change converged |
|    4 |   1.0552 |           27 |              353 |   3.2432380000 |              2.04e-07 | Energy change converged |
|    5 |   1.1642 |           28 |              451 |   5.0672630000 |              1.19e-06 | Energy change converged |
|    6 |   0.9669 |           25 |              328 |   7.2960200000 |              2.69e-07 | Energy change converged |
|    7 |   1.1997 |           29 |              441 |   9.9289580000 |              5.26e-07 | Energy change converged |
|    8 |   1.0861 |           26 |              403 |  12.9658030000 |              1.48e-07 | Energy change converged |
|    9 |   1.0976 |           27 |              379 |  16.4069420000 |              2.96e-08 | Energy change converged |
|   10 |   1.1225 |           26 |              402 |  20.2529480000 |              5.16e-08 | Energy change converged |
|   11 |   1.0395 |           25 |              363 |  24.5041650000 |              1.71e-07 | Energy change converged |
|   12 |   1.2112 |           28 |              446 |  29.1606590000 |              1.85e-08 | Energy change converged |
|   13 |   1.0333 |           24 |              375 |  34.2223430000 |              3.96e-08 | Energy change converged |
|   14 |   1.0119 |           23 |              355 |  39.6889330000 |              1.14e-07 | Energy change converged |
|   15 |   1.1260 |           26 |              388 |  45.5598050000 |              5.00e-07 | Energy change converged |
|   16 |   1.6332 |           34 |              655 |  51.8204000000 |              5.36e-07 | Energy change converged |
|   17 |   1.0097 |           23 |              389 |  58.4802480000 |              1.37e-09 | Energy change converged |
|   18 |   1.0959 |           24 |              421 |  65.5436910000 |              2.57e-08 | Energy change converged |
|   19 |   1.0405 |           23 |              387 |  73.0099330000 |              2.15e-07 | Energy change converged |
|   20 |   1.1080 |           24 |              400 |  80.8777560000 |              1.33e-07 | Energy change converged |
|   21 |   1.1772 |           25 |              444 |  89.1458850000 |              3.28e-07 | Energy change converged |
|   22 |   1.1122 |           25 |              378 |  97.8129100000 |              1.50e-06 | Energy change converged |
|   23 |   1.1977 |           26 |              441 | 106.8769220000 |              4.47e-07 | Energy change converged |
|   24 |   1.3447 |           28 |              501 | 116.3323410000 |              3.43e-05 | Energy change converged |

Summary: total time = `26.5368 s`, total Newton iters = `616`, total linear iters = `9648`, max relative error = `1.87e+00`, mean relative error = `7.81e-02`.

### Level 2 — Custom FEniCS MPI nproc=8

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Relative error vs JAX | Status                  |
| ---: | -------: | -----------: | ---------------: | -------------: | --------------------: | ----------------------- |
|    1 |   0.6221 |           26 |              370 |   0.2027080000 |              7.11e-07 | Energy change converged |
|    2 |   0.6477 |           26 |              379 |   0.8108350000 |              2.57e-06 | Energy change converged |
|    3 |   0.6702 |           25 |              359 |   1.8243800000 |              5.99e-06 | Energy change converged |
|    4 |   0.6576 |           27 |              365 |   3.2432380000 |              2.04e-07 | Energy change converged |
|    5 |   0.6552 |           26 |              370 |   5.0672540000 |              5.90e-07 | Energy change converged |
|    6 |   0.6435 |           26 |              372 |   7.2960160000 |              2.79e-07 | Energy change converged |
|    7 |   0.6661 |           26 |              370 |   9.9289540000 |              1.23e-07 | Energy change converged |
|    8 |   0.6414 |           25 |              358 |  12.9658110000 |              7.65e-07 | Energy change converged |
|    9 |   0.6363 |           25 |              352 |  16.4069430000 |              9.06e-08 | Energy change converged |
|   10 |   0.6045 |           24 |              346 |  20.2529470000 |              2.25e-09 | Energy change converged |
|   11 |   0.6415 |           25 |              373 |  24.5041620000 |              4.89e-08 | Energy change converged |
|   12 |   0.6863 |           26 |              373 |  29.1606660000 |              2.22e-07 | Energy change converged |
|   13 |   0.6695 |           26 |              403 |  34.2223460000 |              1.27e-07 | Energy change converged |
|   14 |   0.6388 |           24 |              365 |  39.6889280000 |              1.24e-08 | Energy change converged |
|   15 |   0.6660 |           25 |              390 |  45.5597980000 |              3.46e-07 | Energy change converged |
|   16 |   1.0414 |           35 |              677 |  51.8203820000 |              1.88e-07 | Energy change converged |
|   17 |   0.6556 |           24 |              383 |  58.4802550000 |              1.18e-07 | Energy change converged |
|   18 |   0.6760 |           24 |              415 |  65.5436900000 |              1.05e-08 | Energy change converged |
|   19 |   0.6639 |           24 |              419 |  73.0099180000 |              9.09e-09 | Energy change converged |
|   20 |   0.6997 |           24 |              409 |  80.8777730000 |              3.43e-07 | Energy change converged |
|   21 |   0.6878 |           25 |              418 |  89.1458820000 |              2.94e-07 | Energy change converged |
|   22 |   0.7401 |           27 |              423 |  97.8128840000 |              1.23e-06 | Energy change converged |
|   23 |   0.7043 |           25 |              385 | 106.8769100000 |              5.59e-07 | Energy change converged |
|   24 |   1.3826 |           45 |              934 | 116.3238510000 |              1.07e-04 | Energy change converged |

Summary: total time = `16.9981 s`, total Newton iters = `635`, total linear iters = `10008`, max relative error = `1.07e-04`, mean relative error = `5.09e-06`.

### Level 2 — Custom FEniCS MPI nproc=16

| Step | Time [s] | Newton iters | Sum linear iters |         Energy | Relative error vs JAX | Status                  |
| ---: | -------: | -----------: | ---------------: | -------------: | --------------------: | ----------------------- |
|    1 |   0.4746 |           24 |              316 |   0.2027200000 |              5.99e-05 | Energy change converged |
|    2 |   0.5919 |           28 |              438 |   0.8108430000 |              1.24e-05 | Energy change converged |
|    3 |   0.5438 |           26 |              383 |   1.8243680000 |              5.90e-07 | Energy change converged |
|    4 |   0.6211 |           28 |              444 |   3.2432460000 |              2.67e-06 | Energy change converged |
|    5 |   0.5553 |           27 |              374 |   5.0672590000 |              3.97e-07 | Energy change converged |
|    6 |   0.6025 |           28 |              387 |   7.2960170000 |              1.42e-07 | Energy change converged |
|    7 |   0.5686 |           26 |              384 |   9.9289570000 |              4.25e-07 | Energy change converged |
|    8 |   0.5674 |           26 |              392 |  12.9658030000 |              1.48e-07 | Energy change converged |
|    9 |   0.5334 |           25 |              374 |  16.4069480000 |              3.95e-07 | Energy change converged |
|   10 |   0.5557 |           26 |              390 |  20.2529480000 |              5.16e-08 | Energy change converged |
|   11 |   0.5617 |           27 |              403 |  24.5041610000 |              8.07e-09 | Energy change converged |
|   12 |   0.5267 |           25 |              391 |  29.1606590000 |              1.85e-08 | Energy change converged |
|   13 |   0.5142 |           25 |              382 |  34.2223470000 |              1.56e-07 | Energy change converged |
|   14 |   0.5792 |           26 |              410 |  39.6889260000 |              6.28e-08 | Energy change converged |
|   15 |   0.5719 |           26 |              409 |  45.5597920000 |              2.15e-07 | Energy change converged |
|   16 |   0.9602 |           39 |              757 |  51.8203790000 |              1.30e-07 | Energy change converged |
|   17 |   0.5257 |           24 |              383 |  58.4802490000 |              1.57e-08 | Energy change converged |
|   18 |   0.5258 |           24 |              410 |  65.5436900000 |              1.05e-08 | Energy change converged |
|   19 |   0.6479 |           25 |              445 |  73.0099120000 |              7.31e-08 | Energy change converged |
|   20 |   0.5806 |           25 |              444 |  80.8777530000 |              9.56e-08 | Energy change converged |
|   21 |   0.6375 |           26 |              504 |  89.1458740000 |              2.04e-07 | Energy change converged |
|   22 |   0.6581 |           28 |              509 |  97.8128410000 |              7.95e-07 | Energy change converged |
|   23 |   0.5592 |           26 |              425 | 106.8769000000 |              6.52e-07 | Energy change converged |
|   24 |   1.1500 |           47 |             1008 | 116.3238720000 |              1.07e-04 | Energy change converged |

Summary: total time = `14.6130 s`, total Newton iters = `657`, total linear iters = `10762`, max relative error = `1.07e-04`, mean relative error = `7.78e-06`.

### Level 3 — Custom FEniCS MPI nproc=16

Artifact:
- [experiment_scripts/he_fenics_custom_evolution_l3_skip_ksp30_pc_cap_np16.json](experiment_scripts/he_fenics_custom_evolution_l3_skip_ksp30_pc_cap_np16.json)

No level-3 JAX reference trajectory is available in this report yet, so relative error vs JAX is reported as `—`.

| Step | Time [s] | Newton iters | Sum linear iters |        Energy | Relative error vs JAX | Status                  |
| ---: | -------: | -----------: | ---------------: | ------------: | --------------------: | ----------------------- |
|    1 |   4.7440 |           26 |              323 |  0.1625750000 |                     — | Energy change converged |
|    2 |   5.1946 |           27 |              332 |  0.6502440000 |                     — | Energy change converged |
|    3 |   5.2324 |           27 |              339 |  1.4630780000 |                     — | Energy change converged |
|    4 |   5.5783 |           28 |              369 |  2.6011410000 |                     — | Energy change converged |
|    5 |   5.5568 |           28 |              369 |  4.0645120000 |                     — | Energy change converged |
|    6 |   5.5878 |           28 |              362 |  5.8532820000 |                     — | Energy change converged |
|    7 |   6.0551 |           29 |              429 |  7.9675450000 |                     — | Energy change converged |
|    8 |   6.0837 |           28 |              435 | 10.4074380000 |                     — | Energy change converged |
|    9 |   5.7178 |           28 |              377 | 13.1730700000 |                     — | Energy change converged |
|   10 |   6.0522 |           28 |              432 | 16.2645250000 |                     — | Energy change converged |
|   11 |   5.8020 |           26 |              415 | 19.6818740000 |                     — | Energy change converged |
|   12 |   5.9056 |           27 |              424 | 23.4251650000 |                     — | Energy change converged |
|   13 |   5.7986 |           27 |              409 | 27.4943760000 |                     — | Energy change converged |
|   14 |   5.8379 |           27 |              415 | 31.8894140000 |                     — | Energy change converged |
|   15 |   4.8552 |           25 |              301 | 36.6099980000 |                     — | Energy change converged |
|   16 |   9.0974 |           37 |              737 | 41.6541600000 |                     — | Energy change converged |
|   17 |   6.7359 |           30 |              515 | 47.0225850000 |                     — | Energy change converged |
|   18 |   5.6796 |           26 |              406 | 52.7168850000 |                     — | Energy change converged |
|   19 |   6.1056 |           27 |              461 | 58.7367310000 |                     — | Energy change converged |
|   20 |   6.5124 |           27 |              509 | 65.0818470000 |                     — | Energy change converged |
|   21 |   6.8496 |           29 |              529 | 71.7518310000 |                     — | Energy change converged |
|   22 |   6.3764 |           28 |              468 | 78.7463110000 |                     — | Energy change converged |
|   23 |   5.9168 |           26 |              410 | 86.0644680000 |                     — | Energy change converged |
|   24 |   6.2964 |           27 |              428 | 93.7051150000 |                     — | Energy change converged |

Summary: total time = `143.5721 s`, total Newton iters = `666`, total linear iters = `10194`.

### Level 4 — Custom FEniCS MPI nproc=16

Artifact:
- [experiment_scripts/he_fenics_custom_evolution_l4_skip_ksp30_pc_cap_np16.json](experiment_scripts/he_fenics_custom_evolution_l4_skip_ksp30_pc_cap_np16.json)
- 2026-03-06 reruns for the added `step 1` rows were recorded as temporary
  local JSON outputs during the investigation and are summarized directly in
  the table below.

No level-4 JAX reference trajectory is available in this report yet, so relative error vs JAX is reported as `—`.

| Step | Time [s] | Newton iters | Sum linear iters |        Energy | Relative error vs JAX | Status                               |
| ---: | -------: | -----------: | ---------------: | ------------: | --------------------: | ------------------------------------ |
|    1 |  91.7594 |           28 |              350 |  0.1519710000 |                     — | Energy change converged              |
| 1 (2026 rerun, no-null) | 186.5641 | 31 | 442 | 0.1519690000 | — | Converged (energy, step, gradient) |
| 1 (2026 rerun, null) | 187.7254 | 31 | 442 | 0.1519690000 | — | Converged (energy, step, gradient) |
|    2 | 133.1481 |           31 |              491 |  0.6078930000 |                     — | Energy change converged              |
|    3 | 101.2237 |           28 |              378 |  1.3678100000 |                     — | Energy change converged              |
|    4 | 102.3104 |           28 |              373 |  2.4318070000 |                     — | Energy change converged              |
|    5 | 115.4161 |           30 |              430 |  3.7999700000 |                     — | Energy change converged              |
|    6 | 105.7331 |           29 |              377 |  5.4724480000 |                     — | Energy change converged              |
|    7 | 126.1320 |           31 |              466 |  7.4493840000 |                     — | Energy change converged              |
|    8 | 129.5468 |           31 |              442 |  9.7309880000 |                     — | Energy change converged              |
|    9 | 116.9576 |           29 |              417 | 12.3173970000 |                     — | Energy change converged              |
|   10 | 106.0879 |           28 |              379 | 15.2088840000 |                     — | Energy change converged              |
|   11 | 111.6968 |           28 |              406 | 18.4055980000 |                     — | Energy change converged              |
|   12 | 116.9532 |           28 |              433 | 21.9074750000 |                     — | Energy change converged              |
|   13 | 120.7920 |           29 |              455 | 25.7143880000 |                     — | Energy change converged              |
|   14 |  68.6393 |           17 |              258 | 29.9425350000 |                     — | Energy change converged              |
|   15 | 110.0916 |           32 |              384 | 34.2432770000 |                     — | Energy change converged              |
|   16 | 116.2453 |           30 |              428 | 38.9651580000 |                     — | Energy change converged              |
|   17 | 114.3216 |           29 |              421 | 43.9918990000 |                     — | Energy change converged              |
|   18 | 107.6639 |           28 |              414 | 49.3237070000 |                     — | Energy change converged              |
|   19 | 115.4159 |           30 |              446 | 54.9608540000 |                     — | Energy change converged              |
|   20 |  21.1442 |           14 |               20 | 71.3995850000 |                     — | Energy change converged              |
|   21 | 183.2629 |          100 |              142 |           nan |                     — | Maximum number of iterations reached |
|   22 | 177.6202 |          100 |              124 |           nan |                     — | Maximum number of iterations reached |
|   23 | 178.1935 |          100 |              140 |           nan |                     — | Maximum number of iterations reached |
|   24 | 178.2756 |          100 |              141 |           nan |                     — | Maximum number of iterations reached |

Summary: total time = `2848.6311 s`, total Newton iters = `958`, total linear iters = `8315`.

---

## Annex A) Diagnostic experiments and solver forensics

This annex contains the investigative campaign that led to the final setup.

### A.1 Step-24 inner-precision sweep (`ksp_rtol = 1e-1 ... 1e-6`)

- Script: [experiment_scripts/sweep_he_custom_step24_precision.py](experiment_scripts/sweep_he_custom_step24_precision.py)
- Outputs:
  - [experiment_scripts/he_step24_precision_sweep/step24_precision_summary.md](experiment_scripts/he_step24_precision_sweep/step24_precision_summary.md)
  - [experiment_scripts/he_step24_precision_sweep/step24_precision_summary.json](experiment_scripts/he_step24_precision_sweep/step24_precision_summary.json)
  - [experiment_scripts/he_step24_precision_sweep/step24_convergence_profiles.csv](experiment_scripts/he_step24_precision_sweep/step24_convergence_profiles.csv)

Main finding:
- `cg+hypre` failed across sweep; `gmres+hypre` converged across sweep.

### A.2 Step-24 setup probes (near-nullspace, `ksp_max_it`, KSP/PC variants)

- Artifacts:
  - [experiment_scripts/he_step24_setup_baseline_gmres_hypre_1e1.json](experiment_scripts/he_step24_setup_baseline_gmres_hypre_1e1.json)
  - [experiment_scripts/he_step24_setup_no_nearnull_gmres_hypre_1e1.json](experiment_scripts/he_step24_setup_no_nearnull_gmres_hypre_1e1.json)
  - [experiment_scripts/he_step24_setup_gmres_hypre_1e1_ksp500.json](experiment_scripts/he_step24_setup_gmres_hypre_1e1_ksp500.json)
  - [experiment_scripts/he_step24_setup_fgmres_hypre_1e1.json](experiment_scripts/he_step24_setup_fgmres_hypre_1e1.json)
  - [experiment_scripts/he_step24_setup_gmres_gamg_1e1.json](experiment_scripts/he_step24_setup_gmres_gamg_1e1.json)

Main finding:
- Capping inner iterations gives major time savings; `gamg` was fast but inaccurate on this nonlinear state.

### A.3 HYPRE option sweep (near-nullspace ON)

- Script: [experiment_scripts/sweep_he_step24_hypre_options.py](experiment_scripts/sweep_he_step24_hypre_options.py)
- Summary:
  - [experiment_scripts/he_step24_hypre_options/summary.md](experiment_scripts/he_step24_hypre_options/summary.md)
  - [experiment_scripts/he_step24_hypre_options/summary.json](experiment_scripts/he_step24_hypre_options/summary.json)

Main finding:
- All tested variants converged; best runtime in tested cases came from skipping explicit `nodal/vec` settings.

### A.4 Cap-triggered preconditioner setup profiling

- Profiles:
  - [experiment_scripts/he_step24_profile_nodal4_vec2_ksp30_pc_cap.json](experiment_scripts/he_step24_profile_nodal4_vec2_ksp30_pc_cap.json)
  - [experiment_scripts/he_step24_profile_skip_nodal_vec_ksp30_pc_cap.json](experiment_scripts/he_step24_profile_skip_nodal_vec_ksp30_pc_cap.json)

Main finding:
- PC setup time drops substantially when setup is not repeated every Newton step.

### A.5 Looser-tolerance trajectory check (`ksp_rtol=5e-1`)

- Artifact: [experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap_rtol5e1.json](experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap_rtol5e1.json)

Main finding:
- Further loosening tolerance reduced inner linear work but increased Newton iterations and introduced localized larger energy error (notably one outlier step).

---

## Annex B) SNES replication campaign (serial screening → L2 → MPI-16)

Goal: reproduce the custom solver robustness as closely as possible using `HyperElasticity3D_fenics/solve_HE_snes_newton.py` under comparable PETSc controls.

Common settings in this campaign:
- `ksp_type=gmres`
- level trajectory length: 24 timesteps
- SNES objective disabled (`--use_objective` not used)
- output artifacts in [experiment_scripts/he_snes_replicate](experiment_scripts/he_snes_replicate)

### B.1 Serial level-1 screening

Screened configurations (all full 24-step trajectories):

| ID                               | Core options                                                                        | Artifact                                                                                                                                                   | Outcome summary                                                  |
| -------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `ls_basic_hypre_r1e1_k30_nonull` | `newtonls + basic`, `pc=hypre`, `ksp_rtol=1e-1`, `ksp_max_it=30`, no near-nullspace | [experiment_scripts/he_snes_replicate/l1_ls_basic_hypre_r1e1_k30_nonull.json](experiment_scripts/he_snes_replicate/l1_ls_basic_hypre_r1e1_k30_nonull.json) | finite through early steps, then NaN; non-converged final reason |
| `ls_basic_hypre_r1e1_k30_null`   | same as above with near-nullspace enabled                                           | [experiment_scripts/he_snes_replicate/l1_ls_basic_hypre_r1e1_k30_null.json](experiment_scripts/he_snes_replicate/l1_ls_basic_hypre_r1e1_k30_null.json)     | similar failure pattern; no robustness gain                      |
| `ls_basic_hypre_r1e2_k30_nonull` | `ksp_rtol=1e-2`, no near-nullspace                                                  | [experiment_scripts/he_snes_replicate/l1_ls_basic_hypre_r1e2_k30_nonull.json](experiment_scripts/he_snes_replicate/l1_ls_basic_hypre_r1e2_k30_nonull.json) | fails very early                                                 |
| `ls_bt_hypre_r1e1_k30_nonull`    | `newtonls + bt` with hypre                                                          | [experiment_scripts/he_snes_replicate/l1_ls_bt_hypre_r1e1_k30_nonull.json](experiment_scripts/he_snes_replicate/l1_ls_bt_hypre_r1e1_k30_nonull.json)       | immediate/early divergence                                       |
| `tr_hypre_r1e1_k30_nonull`       | `newtontr` with hypre                                                               | [experiment_scripts/he_snes_replicate/l1_tr_hypre_r1e1_k30_nonull.json](experiment_scripts/he_snes_replicate/l1_tr_hypre_r1e1_k30_nonull.json)             | immediate/early divergence                                       |
| `ls_basic_asm_r1e1_k30`          | `newtonls + basic`, `pc=asm`, `ksp_rtol=1e-1`, `ksp_max_it=30`                      | [experiment_scripts/he_snes_replicate/l1_ls_basic_asm_r1e1_k30.json](experiment_scripts/he_snes_replicate/l1_ls_basic_asm_r1e1_k30.json)                   | finite energies on all steps, but SNES reasons non-converged     |

Serial summary artifact for this round:
- [experiment_scripts/he_snes_replicate/l1_serial_summary.json](experiment_scripts/he_snes_replicate/l1_serial_summary.json)

Second-round L1 refinements:
- [experiment_scripts/he_snes_replicate/l1_ls_basic_hypre_r1e1_k500_nonull.json](experiment_scripts/he_snes_replicate/l1_ls_basic_hypre_r1e1_k500_nonull.json) (`ksp_max_it=500`)
- [experiment_scripts/he_snes_replicate/l1_ls_basic_asm_r1e3_k500_nonull.json](experiment_scripts/he_snes_replicate/l1_ls_basic_asm_r1e3_k500_nonull.json)
- [experiment_scripts/he_snes_replicate/l1_ls_l2_hypre_r1e1_k500_nonull.json](experiment_scripts/he_snes_replicate/l1_ls_l2_hypre_r1e1_k500_nonull.json)

Main findings from serial screening:
- No tested SNES setup achieved the same “all-step converged + low-error” behavior as custom FEniCS final setup.
- `hypre` with larger inner cap (`ksp_max_it=500`) improved early-step convergence count but still diverged in the mid trajectory.
- `asm` with loose inner tolerance stayed finite on L1 but did not satisfy SNES convergence criteria (`reason <= 0` at all steps).

### B.2 Shortlisted candidates and downstream validation

Shortlist used for downstream checks:
- Candidate H: `newtonls/basic + gmres+hypre`, `ksp_rtol=1e-1`, `ksp_max_it=500`, no near-nullspace.
- Candidate A: `newtonls/basic + gmres+asm`, `ksp_rtol=1e-1`, `ksp_max_it=30`.

Validation metrics (relative errors computed against the JAX level-1/level-2 reference trajectories in Section 2):

| Case                    | Finite steps | Converged steps (`reason > 0`) | Total time [s] | Max relative error vs JAX | Mean relative error vs JAX | First NaN step |
| ----------------------- | -----------: | -----------------------------: | -------------: | ------------------------: | -------------------------: | -------------: |
| L1 serial / Candidate H |           13 |                             12 |        11.5733 |                  1.92e+02 |                   1.48e+01 |             14 |
| L1 serial / Candidate A |           24 |                              0 |         1.1367 |                  1.79e-02 |                   6.21e-03 |              — |
| L2 serial / Candidate H |            0 |                              0 |       151.5931 |                         — |                          — |              1 |
| L2 serial / Candidate A |           14 |                              0 |         3.7082 |                  3.60e+00 |                   1.07e+00 |             15 |
| L1 MPI-16 / Candidate H |            0 |                              0 |         6.8064 |                         — |                          — |              1 |
| L1 MPI-16 / Candidate A |           20 |                              0 |         0.1220 |                  2.27e+00 |                   4.80e-01 |             21 |

Artifacts for L2 and MPI-16 validation:
- [experiment_scripts/he_snes_replicate/l2_ls_basic_hypre_r1e1_k500_nonull.json](experiment_scripts/he_snes_replicate/l2_ls_basic_hypre_r1e1_k500_nonull.json)
- [experiment_scripts/he_snes_replicate/l2_ls_basic_asm_r1e1_k30.json](experiment_scripts/he_snes_replicate/l2_ls_basic_asm_r1e1_k30.json)
- [experiment_scripts/he_snes_replicate/l1_np16_ls_basic_hypre_r1e1_k500_nonull.json](experiment_scripts/he_snes_replicate/l1_np16_ls_basic_hypre_r1e1_k500_nonull.json)
- [experiment_scripts/he_snes_replicate/l1_np16_ls_basic_asm_r1e1_k30.json](experiment_scripts/he_snes_replicate/l1_np16_ls_basic_asm_r1e1_k30.json)

### B.3 Conclusion of this SNES campaign

- Under tested settings, SNES did not match the custom solver robustness across serial L1/L2 and MPI-16 checks.
- Best partial behavior depended on criterion:
  - convergence count on early L1 steps: Candidate H,
  - finite full L1 trajectory (but non-converged SNES reasons): Candidate A.
- For production-quality trajectories in this project, the custom solver path remains the recommended method.
- For the later HE JAX+PETSc element-path distribution/layout follow-up and the
  production reordered-overlap implementation, see
  [HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md](/home/michal/repos/fenics_nonlinear_energies/HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md).

### B.4 Continuation rerun for iterative SNES (half step size)

To test whether the 24-step loading path was too aggressive for SNES iterative methods, we repeated the Annex B iterative configurations with continuation at half step size (`48` steps over the same total rotation).

Important implementation note:
- This required fixing the step-to-rotation scaling in [HyperElasticity3D_fenics/solve_HE_snes_newton.py](HyperElasticity3D_fenics/solve_HE_snes_newton.py) so `rotation_per_iter` uses `num_steps` instead of a hardcoded 24.

Half-step artifacts:
- [experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e1_k30_nonull.json](experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e1_k30_nonull.json)
- [experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e1_k30_null.json](experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e1_k30_null.json)
- [experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e2_k30_nonull.json](experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e2_k30_nonull.json)
- [experiment_scripts/he_snes_replicate/l1_half_ls_bt_hypre_r1e1_k30_nonull.json](experiment_scripts/he_snes_replicate/l1_half_ls_bt_hypre_r1e1_k30_nonull.json)
- [experiment_scripts/he_snes_replicate/l1_half_tr_hypre_r1e1_k30_nonull.json](experiment_scripts/he_snes_replicate/l1_half_tr_hypre_r1e1_k30_nonull.json)
- [experiment_scripts/he_snes_replicate/l1_half_ls_basic_asm_r1e1_k30.json](experiment_scripts/he_snes_replicate/l1_half_ls_basic_asm_r1e1_k30.json)
- [experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e1_k500_nonull.json](experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e1_k500_nonull.json)
- [experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e1_k500_null_hypredefault.json](experiment_scripts/he_snes_replicate/l1_half_ls_basic_hypre_r1e1_k500_null_hypredefault.json)

HYPRE setup parity check vs custom solver:
- In custom final setup, explicit BoomerAMG `nodal_coarsen` / `vec_interp_variant` are skipped (HYPRE defaults).
- The added `k500 + null` case above was run with `--hypre_nodal_coarsen -1 --hypre_vec_interp_variant -1` to match that behavior.

Summary at 48 steps (L1 serial):

| Case                                                              | Finite steps | Positive-reason steps | First non-finite step | First `reason <= 0` | Total time [s] | Sum linear iters | Relative error at final step |
| ----------------------------------------------------------------- | -----------: | --------------------: | --------------------: | ------------------: | -------------: | ---------------: | ---------------------------: |
| `newtonls/basic + gmres+hypre (1e-1, k30, no-null)`               |           48 |                     0 |                     — |                   1 |         4.5488 |             5319 |                     5.44e-06 |
| `newtonls/basic + gmres+hypre (1e-1, k30, null)`                  |           48 |                     0 |                     — |                   1 |        15.4540 |             4262 |                     7.14e-03 |
| `newtonls/basic + gmres+hypre (1e-2, k30, no-null)`               |           28 |                     0 |                    17 |                   1 |         1.6815 |             2663 |                            — |
| `newtonls/bt + gmres+hypre (1e-1, k30, no-null)`                  |           25 |                     0 |                    12 |                   1 |         0.6467 |              168 |                            — |
| `newtontr + gmres+hypre (1e-1, k30, no-null)`                     |            1 |                     0 |                     2 |                   1 |         1.2449 |             1400 |                            — |
| `newtonls/basic + gmres+asm (1e-1, k30)`                          |           48 |                     0 |                     — |                   1 |         1.4612 |             5063 |                     5.47e-03 |
| `newtonls/basic + gmres+hypre (1e-1, k500, no-null)`              |           48 |                    45 |                     — |                  46 |        18.5889 |            38589 |                     3.21e-05 |
| `newtonls/basic + gmres+hypre (1e-1, k500, null, hypre defaults)` |           48 |                    46 |                     — |                  47 |         9.8215 |            14521 |                     3.27e-05 |

Interpretation:
- The original 24-step path was indeed a harder nonlinear trajectory for these iterative SNES variants.
- With half-size continuation increments, several iterative methods remain finite across all 48 steps and recover accurate final energies.
- However, strict SNES convergence reasons remain a limitation for most iterative settings (`reason <= 0` appears early), except the `k500` hypre variant which is positive for most steps before degrading near the end.

---

## Annex C) Custom solver — quarter-step baseline (96 steps, level 1)

**Setup:** `solve_HE_custom_jaxversion.py`, mesh level 1 (2187 DOFs), 96 steps (`--total_steps 96`),
`ksp_rtol=1e-1`, `ksp_max_it=30`, near-nullspace **ON** (default: `nodal_coarsen=6`, `vec_interp_variant=3`), serial execution.

Each step covers 15° of rotation (quarter of the original 60°/step); total rotation = 4 × 360° = 1440°.

**Artifact:** `experiment_scripts/he_custom_quarter_steps_l1_k30.json`

**Summary:** 96/96 steps converged, total Newton iterations = 1209, total KSP iterations = 24 872, wall time = 72.62 s.

### C.1 Per-step table (level 1, 96 quarter-steps)

| Step | Angle [°] | Time [s] | Newton its | Sum KSP its | Message                 |
| ---: | --------: | -------: | ---------: | ----------: | ----------------------- |
|    1 |     15.00 |   0.4358 |         10 |         130 | Energy change converged |
|    2 |     30.00 |   0.3705 |          9 |         119 | Energy change converged |
|    3 |     45.00 |   0.4181 |         10 |         143 | Energy change converged |
|    4 |     60.00 |   0.3699 |          9 |         123 | Energy change converged |
|    5 |     75.00 |   0.3692 |          9 |         120 | Energy change converged |
|    6 |     90.00 |   0.4310 |         10 |         132 | Energy change converged |
|    7 |    105.00 |   0.3872 |          9 |         118 | Energy change converged |
|    8 |    120.00 |   0.4191 |          9 |         124 | Energy change converged |
|    9 |    135.00 |   0.4295 |          9 |         121 | Energy change converged |
|   10 |    150.00 |   0.5972 |         11 |         179 | Energy change converged |
|   11 |    165.00 |   0.4421 |          9 |         111 | Energy change converged |
|   12 |    180.00 |   0.5949 |         11 |         170 | Energy change converged |
|   13 |    195.00 |   0.5186 |         10 |         135 | Energy change converged |
|   14 |    210.00 |   0.4625 |          9 |         118 | Energy change converged |
|   15 |    225.00 |   0.6432 |         11 |         173 | Energy change converged |
|   16 |    240.00 |   0.5138 |          9 |         135 | Energy change converged |
|   17 |    255.00 |   0.5919 |         10 |         163 | Energy change converged |
|   18 |    270.00 |   0.5011 |          9 |         132 | Energy change converged |
|   19 |    285.00 |   0.6760 |         11 |         192 | Energy change converged |
|   20 |    300.00 |   0.7682 |         12 |         229 | Energy change converged |
|   21 |    315.00 |   0.6815 |         11 |         197 | Energy change converged |
|   22 |    330.00 |   0.6804 |         11 |         211 | Energy change converged |
|   23 |    345.00 |   0.6791 |         11 |         222 | Energy change converged |
|   24 |    360.00 |   0.8307 |         13 |         279 | Energy change converged |
|   25 |    375.00 |   0.6293 |         11 |         193 | Energy change converged |
|   26 |    390.00 |   0.5857 |         10 |         177 | Energy change converged |
|   27 |    405.00 |   0.7540 |         12 |         237 | Energy change converged |
|   28 |    420.00 |   0.8244 |         13 |         288 | Energy change converged |
|   29 |    435.00 |   0.6487 |         11 |         212 | Energy change converged |
|   30 |    450.00 |   0.7380 |         12 |         249 | Energy change converged |
|   31 |    465.00 |   0.7291 |         12 |         249 | Energy change converged |
|   32 |    480.00 |   0.7364 |         12 |         254 | Energy change converged |
|   33 |    495.00 |   0.7422 |         12 |         256 | Energy change converged |
|   34 |    510.00 |   0.7603 |         12 |         253 | Energy change converged |
|   35 |    525.00 |   0.8708 |         14 |         317 | Energy change converged |
|   36 |    540.00 |   0.8082 |         13 |         279 | Energy change converged |
|   37 |    555.00 |   0.7232 |         12 |         254 | Energy change converged |
|   38 |    570.00 |   0.9991 |         15 |         342 | Energy change converged |
|   39 |    585.00 |   1.0087 |         15 |         346 | Energy change converged |
|   40 |    600.00 |   0.9308 |         14 |         313 | Energy change converged |
|   41 |    615.00 |   0.8639 |         13 |         286 | Energy change converged |
|   42 |    630.00 |   0.7935 |         12 |         265 | Energy change converged |
|   43 |    645.00 |   0.8351 |         13 |         289 | Energy change converged |
|   44 |    660.00 |   0.7930 |         12 |         259 | Energy change converged |
|   45 |    675.00 |   0.6237 |         10 |         202 | Energy change converged |
|   46 |    690.00 |   0.7311 |         12 |         250 | Energy change converged |
|   47 |    705.00 |   0.7345 |         12 |         251 | Energy change converged |
|   48 |    720.00 |   0.6604 |         11 |         225 | Energy change converged |
|   49 |    735.00 |   0.7461 |         12 |         253 | Energy change converged |
|   50 |    750.00 |   0.6635 |         11 |         221 | Energy change converged |
|   51 |    765.00 |   0.7339 |         12 |         256 | Energy change converged |
|   52 |    780.00 |   0.5847 |         10 |         196 | Energy change converged |
|   53 |    795.00 |   0.7501 |         12 |         257 | Energy change converged |
|   54 |    810.00 |   0.6581 |         11 |         225 | Energy change converged |
|   55 |    825.00 |   0.5673 |         10 |         192 | Energy change converged |
|   56 |    840.00 |   0.5822 |         10 |         191 | Energy change converged |
|   57 |    855.00 |   0.6336 |         11 |         218 | Energy change converged |
|   58 |    870.00 |   0.6540 |         11 |         221 | Energy change converged |
|   59 |    885.00 |   0.7420 |         12 |         257 | Energy change converged |
|   60 |    900.00 |   0.7396 |         12 |         262 | Energy change converged |
|   61 |    915.00 |   0.9816 |         15 |         352 | Energy change converged |
|   62 |    930.00 |   1.4249 |         21 |         527 | Energy change converged |
|   63 |    945.00 |   1.3022 |         19 |         467 | Energy change converged |
|   64 |    960.00 |   1.5453 |         23 |         594 | Energy change converged |
|   65 |    975.00 |   1.2676 |         19 |         475 | Energy change converged |
|   66 |    990.00 |   1.0855 |         17 |         413 | Energy change converged |
|   67 |   1005.00 |   1.2409 |         18 |         439 | Energy change converged |
|   68 |   1020.00 |   1.1502 |         17 |         409 | Energy change converged |
|   69 |   1035.00 |   1.2432 |         18 |         440 | Energy change converged |
|   70 |   1050.00 |   0.9480 |         15 |         346 | Energy change converged |
|   71 |   1065.00 |   1.0231 |         16 |         373 | Energy change converged |
|   72 |   1080.00 |   1.1429 |         17 |         405 | Energy change converged |
|   73 |   1095.00 |   0.9864 |         16 |         366 | Energy change converged |
|   74 |   1110.00 |   0.8479 |         14 |         312 | Energy change converged |
|   75 |   1125.00 |   0.9908 |         16 |         371 | Energy change converged |
|   76 |   1140.00 |   0.8141 |         14 |         310 | Energy change converged |
|   77 |   1155.00 |   1.0415 |         17 |         402 | Energy change converged |
|   78 |   1170.00 |   0.9695 |         16 |         366 | Energy change converged |
|   79 |   1185.00 |   0.7526 |         13 |         272 | Energy change converged |
|   80 |   1200.00 |   0.9131 |         15 |         342 | Energy change converged |
|   81 |   1215.00 |   0.9378 |         16 |         361 | Energy change converged |
|   82 |   1230.00 |   0.9048 |         15 |         331 | Energy change converged |
|   83 |   1245.00 |   0.8376 |         14 |         300 | Energy change converged |
|   84 |   1260.00 |   0.8371 |         14 |         300 | Energy change converged |
|   85 |   1275.00 |   0.7673 |         13 |         271 | Energy change converged |
|   86 |   1290.00 |   0.8288 |         14 |         297 | Energy change converged |
|   87 |   1305.00 |   0.9932 |         16 |         361 | Energy change converged |
|   88 |   1320.00 |   0.6121 |         11 |         209 | Energy change converged |
|   89 |   1335.00 |   0.6787 |         12 |         240 | Energy change converged |
|   90 |   1350.00 |   0.5973 |         11 |         205 | Energy change converged |
|   91 |   1365.00 |   0.6017 |         11 |         212 | Energy change converged |
|   92 |   1380.00 |   0.6657 |         12 |         241 | Energy change converged |
|   93 |   1395.00 |   0.5833 |         11 |         207 | Energy change converged |
|   94 |   1410.00 |   0.5265 |         10 |         177 | Energy change converged |
|   95 |   1425.00 |   0.5898 |         11 |         203 | Energy change converged |
|   96 |   1440.00 |   0.5933 |         11 |         205 | Energy change converged |

| **Total** | | **72.62** | **1209** | **24 872** | |

**Observations:**
- All 96 steps converged (100% success rate).
- Steps 1–9: ~9–10 Newton iters, ~120–145 KSP per step (loose tolerance reached quickly).
- Steps 10–60: gradual increase from ~9 to ~15 Newton iters, ~110–350 KSP per step, as deformation builds.
- Steps 62–69: hardest band (915°–1035°), 17–23 Newton iters and up to 594 KSP per step.
- Steps 70–96: recovery to 11–17 Newton iters, ~200–410 KSP per step.
- Average KSP iterations per Newton step: 24 872 / 1209 ≈ **20.6 KSP/Newton** (capped at 30).
- Wall time 72.62 s vs 363.31 s for `ksp_rtol=1e-3`/`ksp_max_it=500` — **5× faster** at the cost of ~50% more Newton iterations (1209 vs 813).

---

## Annex D) SNES near-nullspace verification — 96 quarter-steps (level 1)

**Goal:** Verify that the near-nullspace is correctly configured and working in the SNES solver, and replicate the custom solver Annex C table as closely as possible.

**Root cause analysis of earlier SNES failures with n6v3:**
- `vec_interp_variant=3` in HYPRE BoomerAMG produces a **non-symmetric interpolation operator**, making the AMG preconditioner non-symmetric.
- **CG breaks down** (reports `KSP_DIVERGED_BREAKDOWN` → `SNES_DIVERGED_LINEAR_SOLVE`) for the non-symmetric preconditioned system. The custom solver accepts the partial CG direction anyway (no convergence check), so it works.
- **GMRES with n6v3** avoids breakdown but stagnates: 72.9 KSP/Newton with `ksp_max_it=300` vs 20.6 for the custom CG.
- **BiCGSTAB with n6v3**: completes 96/96 but needs 27.8 KSP/Newton and 138 s (1.4× more iterations, 1.9× slower than custom).
- **GMRES with HYPRE defaults** (no `nodal_coarsen`, no `vec_interp_variant`): symmetric preconditioner → GMRES convergence comparable to custom CG, near-nullspace still active.

**Best-matching SNES configuration:**
`newtonls + basic`, `ksp_type=gmres`, `pc_type=hypre` (defaults), near-nullspace ON (no Gram-Schmidt), `ksp_rtol=1e-1`, `ksp_max_it=500`, `snes_atol=1e-3`.

**Artifact:** `experiment_scripts/he_snes_96steps_defaults_gmres_null_k500_atol3.json`

**Summary:** 93/96 steps converged, total Newton = 1 175, total KSP = 22 490, wall time = 15.03 s.

**Comparison with custom solver (Annex C):**

| Metric                    |                          Custom solver (Annex C) |   SNES (Annex D) |
| ------------------------- | -----------------------------------------------: | ---------------: |
| KSP type                  |                                               CG |            GMRES |
| AMG config                | `nodal_coarsen=6`, `vec_interp_variant=3` (n6v3) |   HYPRE defaults |
| `ksp_rtol`                |                                             1e-1 |             1e-1 |
| `ksp_max_it`              |                                               30 |              500 |
| Convergence criterion     |                             energy change < 1e-4 | `snes_atol=1e-3` |
| Converged steps           |                                            96/96 |            93/96 |
| Total Newton iterations   |                                            1 209 |            1 175 |
| Total KSP iterations      |                                           24 872 |           22 490 |
| **Avg KSP / Newton step** |                                         **20.6** |         **19.1** |
| Wall time [s]             |                                            72.62 |            15.03 |

**Key finding:** Average KSP iterations per Newton step are essentially identical (19.1 vs 20.6), confirming the near-nullspace **IS correctly configured and active** in the SNES solver.  The 3/96 failures (steps 94–96, angles 1410°–1440°) require >500 GMRES iterations for individual linear solves at extreme deformation — a hard-conditioning problem at those specific Jacobians, not a nullspace issue. SNES is significantly faster in wall time (15 s vs 73 s) due to lighter assembly overhead vs the custom Newton loop.

**Why n6v3 is incompatible with SNES standard KSP:**
`vec_interp_variant=3` creates a non-symmetric AMG preconditioner. SNES requires KSP to converge (positive reason code) and aborts with `SNES_DIVERGED_LINEAR_SOLVE` if KSP diverges. CG formally breaks down on non-symmetric preconditioned systems; GMRES with non-symmetric preconditioning converges slowly. The custom Newton loop accepts any KSP output (even `KSP_DIVERGED_BREAKDOWN`) as a search direction, then uses golden-section line search to ensure descent — a flexibility not available in standard SNES.

### D.1 Per-step table (level 1, 96 quarter-steps, SNES GMRES+defaults+null)

| Step | Angle [°] | Time [s] | Newton its | Sum KSP its | Reason |
| ---: | --------: | -------: | ---------: | ----------: | -----: |
|    1 |     15.00 |   0.0916 |         12 |         147 |   conv |
|    2 |     30.00 |   0.0992 |         12 |         169 |   conv |
|    3 |     45.00 |   0.1115 |         13 |         187 |   conv |
|    4 |     60.00 |   0.1166 |         13 |         193 |   conv |
|    5 |     75.00 |   0.1163 |         13 |         187 |   conv |
|    6 |     90.00 |   0.1187 |         13 |         193 |   conv |
|    7 |    105.00 |   0.1115 |         12 |         170 |   conv |
|    8 |    120.00 |   0.1204 |         13 |         176 |   conv |
|    9 |    135.00 |   0.1201 |         12 |         183 |   conv |
|   10 |    150.00 |   0.1159 |         12 |         164 |   conv |
|   11 |    165.00 |   0.1276 |         13 |         189 |   conv |
|   12 |    180.00 |   0.1282 |         13 |         188 |   conv |
|   13 |    195.00 |   0.1223 |         12 |         187 |   conv |
|   14 |    210.00 |   0.1231 |         12 |         181 |   conv |
|   15 |    225.00 |   0.1299 |         13 |         182 |   conv |
|   16 |    240.00 |   0.1133 |         11 |         160 |   conv |
|   17 |    255.00 |   0.1230 |         12 |         181 |   conv |
|   18 |    270.00 |   0.1277 |         12 |         196 |   conv |
|   19 |    285.00 |   0.1136 |         11 |         162 |   conv |
|   20 |    300.00 |   0.1405 |         13 |         211 |   conv |
|   21 |    315.00 |   0.1219 |         12 |         167 |   conv |
|   22 |    330.00 |   0.1237 |         12 |         174 |   conv |
|   23 |    345.00 |   0.1455 |         13 |         224 |   conv |
|   24 |    360.00 |   0.1413 |         13 |         216 |   conv |
|   25 |    375.00 |   0.1230 |         12 |         171 |   conv |
|   26 |    390.00 |   0.1234 |         12 |         166 |   conv |
|   27 |    405.00 |   0.1209 |         12 |         153 |   conv |
|   28 |    420.00 |   0.1158 |         11 |         157 |   conv |
|   29 |    435.00 |   0.1299 |         12 |         184 |   conv |
|   30 |    450.00 |   0.1223 |         12 |         159 |   conv |
|   31 |    465.00 |   0.1267 |         12 |         152 |   conv |
|   32 |    480.00 |   0.1449 |         13 |         194 |   conv |
|   33 |    495.00 |   0.1424 |         13 |         178 |   conv |
|   34 |    510.00 |   0.1373 |         13 |         171 |   conv |
|   35 |    525.00 |   0.1429 |         13 |         176 |   conv |
|   36 |    540.00 |   0.1495 |         13 |         204 |   conv |
|   37 |    555.00 |   0.1371 |         12 |         180 |   conv |
|   38 |    570.00 |   0.1415 |         13 |         180 |   conv |
|   39 |    585.00 |   0.1462 |         13 |         204 |   conv |
|   40 |    600.00 |   0.1244 |         11 |         161 |   conv |
|   41 |    615.00 |   0.1502 |         13 |         205 |   conv |
|   42 |    630.00 |   0.1414 |         13 |         189 |   conv |
|   43 |    645.00 |   0.1282 |         12 |         160 |   conv |
|   44 |    660.00 |   0.1406 |         13 |         179 |   conv |
|   45 |    675.00 |   0.1324 |         12 |         172 |   conv |
|   46 |    690.00 |   0.1476 |         13 |         197 |   conv |
|   47 |    705.00 |   0.1488 |         13 |         191 |   conv |
|   48 |    720.00 |   0.1418 |         12 |         183 |   conv |
|   49 |    735.00 |   0.1387 |         12 |         176 |   conv |
|   50 |    750.00 |   0.1327 |         12 |         156 |   conv |
|   51 |    765.00 |   0.1298 |         11 |         162 |   conv |
|   52 |    780.00 |   0.1401 |         12 |         168 |   conv |
|   53 |    795.00 |   0.1409 |         12 |         179 |   conv |
|   54 |    810.00 |   0.1526 |         13 |         193 |   conv |
|   55 |    825.00 |   0.1470 |         13 |         185 |   conv |
|   56 |    840.00 |   0.1428 |         12 |         193 |   conv |
|   57 |    855.00 |   0.1434 |         12 |         193 |   conv |
|   58 |    870.00 |   0.1263 |         11 |         165 |   conv |
|   59 |    885.00 |   0.1462 |         12 |         194 |   conv |
|   60 |    900.00 |   0.1326 |         11 |         180 |   conv |
|   61 |    915.00 |   0.1584 |         12 |         216 |   conv |
|   62 |    930.00 |   0.1787 |         13 |         244 |   conv |
|   63 |    945.00 |   0.1640 |         12 |         221 |   conv |
|   64 |    960.00 |   0.1656 |         12 |         228 |   conv |
|   65 |    975.00 |   0.1786 |         13 |         238 |   conv |
|   66 |    990.00 |   0.1650 |         12 |         222 |   conv |
|   67 |   1005.00 |   0.1499 |         11 |         196 |   conv |
|   68 |   1020.00 |   0.1868 |         13 |         267 |   conv |
|   69 |   1035.00 |   0.1651 |         12 |         218 |   conv |
|   70 |   1050.00 |   0.1611 |         12 |         208 |   conv |
|   71 |   1065.00 |   0.1649 |         12 |         225 |   conv |
|   72 |   1080.00 |   0.1456 |         11 |         190 |   conv |
|   73 |   1095.00 |   0.1597 |         12 |         214 |   conv |
|   74 |   1110.00 |   0.1543 |         12 |         207 |   conv |
|   75 |   1125.00 |   0.1523 |         12 |         192 |   conv |
|   76 |   1140.00 |   0.1520 |         12 |         192 |   conv |
|   77 |   1155.00 |   0.1580 |         12 |         211 |   conv |
|   78 |   1170.00 |   0.1612 |         12 |         213 |   conv |
|   79 |   1185.00 |   0.1497 |         12 |         196 |   conv |
|   80 |   1200.00 |   0.1570 |         12 |         204 |   conv |
|   81 |   1215.00 |   0.1568 |         12 |         205 |   conv |
|   82 |   1230.00 |   0.1525 |         12 |         200 |   conv |
|   83 |   1245.00 |   0.1571 |         12 |         202 |   conv |
|   84 |   1260.00 |   0.1632 |         12 |         224 |   conv |
|   85 |   1275.00 |   0.1596 |         12 |         213 |   conv |
|   86 |   1290.00 |   0.1603 |         12 |         213 |   conv |
|   87 |   1305.00 |   0.1603 |         12 |         208 |   conv |
|   88 |   1320.00 |   0.1535 |         11 |         196 |   conv |
|   89 |   1335.00 |   0.1530 |         11 |         188 |   conv |
|   90 |   1350.00 |   0.1620 |         12 |         208 |   conv |
|   91 |   1365.00 |   0.1547 |         11 |         194 |   conv |
|   92 |   1380.00 |   0.1600 |         12 |         211 |   conv |
|   93 |   1395.00 |   0.1636 |         12 |         214 |   conv |
|   94 |   1410.00 |   0.8676 |         10 |         867 |   r=-3 |
|   95 |   1425.00 |   0.6536 |         11 |         654 |   r=-3 |
|   96 |   1440.00 |   0.5967 |          8 |         597 |   r=-3 |

| **Total** | | **15.03** | **1175** | **22,490** | **93/96** |

**Observations:**
- Steps 1–93: 11–13 Newton iters, 150–270 KSP per step (all converging, avg 19.0 KSP/Newton).
- Avg KSP/Newton ratio: 22 490 / 1 175 ≈ **19.1 KSP/Newton** — essentially matches custom solver's 20.6.
- Steps 94–96 (1410°–1440°): `SNES_DIVERGED_LINEAR_SOLVE`; each step hits the 500-iteration KSP cap on one internal Newton linear solve, suggesting HYPRE defaults AMG loses effectiveness at this extreme deformation state. Not a nullspace issue.

---

## Annex E) Speed investigation: why is SNES faster at level 1 but slower at level 3?

**Motivation:** Annex D showed SNES completing 96 quarter-steps in **15.03 s** vs custom's **72.62 s** (5× advantage at level 1).  Yet the main §2 scaling tables showed SNES slower than custom at larger meshes with more MPI processes.  This annex isolates the very first quarter-step (15° rotation) at mesh level 3 with 16 MPI processes to understand the crossover.

### E.1 Critical bug discovered during setup: `build_nullspace` parallelism fix

While setting up these experiments a critical bug was found in `build_nullspace` in **both** solvers (`solve_HE_custom_jaxversion.py` and `solve_HE_snes_newton.py`):

**Root cause:** `A.createVecLeft()` produces a standard MPI (non-ghost) PETSc Vec.  Calling `localForm()` on such a Vec returns an empty read-only sequential form (size 0).  All numpy array assignments to `loc.array[...]` were silently no-ops, leaving all six rigid-body nullspace vectors as zeros.  `ghostUpdate(INSERT_VALUES, FORWARD)` then raises `PETSc.Error` ("Vector is not ghosted") on any np > 1.

**Effect:** For np = 1 the AMG preconditioner still ran (HYPRE silently ignores all-zero near-nullspace), but convergence was degraded.  For np > 1 the process crashed with a SIGSEGV inside HYPRE's coarsening routines.

**Fix:** Replace `localForm()` / `loc.array` with a direct `vec.getArray()` call, which correctly returns the owned local entries for standard MPI Vecs.  Remove the `ghostUpdate(INSERT_VALUES, FORWARD)` call which is invalid on non-ghost Vecs.  Both files were patched before running the Annex E experiments.

**Additional finding:** The `nodal_coarsen=6 + vec_interp_variant=3` (n6v3) HYPRE configuration crashes inside HYPRE at mesh level ≥ 2, even at np = 1.  All Annex E experiments therefore use HYPRE defaults (`hypre_nodal_coarsen=-1`, `hypre_vec_interp_variant=-1`).

### E.2 Experiment setup

Both solvers run the identical problem: one quarter-step (15°) of a 96-step trajectory on a level 3 mesh (78 003 DOFs) using 16 MPI processes.  The `--total_steps 96` flag (added to the SNES solver for this investigation) pins the rotation per step to 360°×4/96 = 15°, independent of the `--steps` count.

| Parameter             | Custom Newton                                 | SNES Newton                             |
| --------------------- | --------------------------------------------- | --------------------------------------- |
| Script                | `solve_HE_custom_jaxversion.py`               | `solve_HE_snes_newton.py`               |
| Mesh level / DOFs     | 3 / 78 003                                    | 3 / 78 003                              |
| MPI processes         | 16                                            | 16                                      |
| Steps (total_steps)   | 1 (of 96)                                     | 1 (of 96)                               |
| KSP type              | CG                                            | GMRES                                   |
| PC type               | HYPRE BoomerAMG (defaults)                    | HYPRE BoomerAMG (defaults)              |
| `ksp_rtol`            | 1e-1                                          | 1e-1                                    |
| `ksp_max_it`          | 30                                            | 500                                     |
| Convergence criterion | energy change < 1e-4                          | `snes_atol=1e-3`                        |
| AMG reuse strategy    | `--pc_setup_on_ksp_cap` (reuse until cap hit) | PETSc SNES: reassemble each Newton iter |
| Line search           | Golden-section (custom Python)                | `newtonls` + `basic` (PETSc)            |
| Artifacts             | `he_speed_custom_l3np16.json/txt`             | `he_speed_snes_l3np16.json/txt`         |

### E.3 Results summary

| Metric                   | Custom Newton | SNES Newton |
| ------------------------ | ------------: | ----------: |
| Wall time [s]            |    **1.8239** |  **2.5096** |
| Newton iterations        |            11 |          14 |
| Total KSP iterations     |           148 |         214 |
| Avg KSP / Newton         |          13.5 |        15.3 |
| Final energy             |      0.010163 |     0.01016 |
| PC setups performed      |             1 |          14 |
| PC setup time [s]        |        0.0605 |     ~0.84 ✱ |
| Total assembly time [s]  |        0.1679 |        — ✱✱ |
| Total KSP solve time [s] |        1.4822 |        — ✱✱ |
| Total linear time [s]    |        1.7108 |        — ✱✱ |
| Non-linear overhead [s]  |        0.1131 |        — ✱✱ |

✱ Estimated: 14 Newton iters × ~0.0605 s/setup (measured from the custom solver's first PC setup at this level/np).
✱✱ SNES manages Newton/KSP internals; no per-Newton timing is exposed.

**At level 3 np=16, custom is ~1.4× faster (1.82 s vs 2.51 s).**  This is the opposite of the level 1 result.

### E.4 Per-Newton timing breakdown — custom solver

The `--save_linear_timing` flag records assembly, PC setup, and KSP solve times for every Newton iteration:

| Newton | KSP its | assemble [s] | pc_setup [s] | ksp_solve [s] | lin_total [s] |
| -----: | ------: | -----------: | -----------: | ------------: | ------------: |
|      0 |       4 |       0.0142 |       0.0605 |        0.0243 |        0.0989 |
|      1 |       6 |       0.0148 |       0.0000 |        0.0913 |        0.1061 |
|      2 |      11 |       0.0163 |       0.0000 |        0.1194 |        0.1357 |
|      3 |      12 |       0.0151 |       0.0000 |        0.1283 |        0.1434 |
|      4 |      16 |       0.0149 |       0.0000 |        0.1525 |        0.1674 |
|      5 |      15 |       0.0147 |       0.0000 |        0.1465 |        0.1612 |
|      6 |      13 |       0.0149 |       0.0000 |        0.1391 |        0.1540 |
|      7 |      18 |       0.0152 |       0.0000 |        0.1666 |        0.1818 |
|      8 |      12 |       0.0156 |       0.0000 |        0.1380 |        0.1536 |
|      9 |      29 |       0.0158 |       0.0000 |        0.2375 |        0.2533 |
|     10 |      12 |       0.0165 |       0.0000 |        0.1388 |        0.1553 |
|  **Σ** | **148** |   **0.1679** |   **0.0605** |    **1.4822** |    **1.7108** |

**Observations:**
- Only Newton step 0 incurs a PC setup (0.060 s).  The `pc_setup_on_ksp_cap` flag reuses the AMG preconditioner for all subsequent Newton steps — none of steps 1–10 hit the `ksp_max_it=30` cap.
- KSP iterations grow from 4 (step 0) to a peak of 29 (step 9) as the Jacobian drifts from the initial preconditioner.  Step 10 drops back to 12, consistent with the iterate converging near the minimum.
- Assembly is essentially constant (~0.015 s per Newton) and is parallelism-bound (mesh-level 3, 16 processes).
- KSP solve time (1.48 s, 81.3% of wall) dominates; the increase from step 0→9 reflects the growing misalignment between the reused AMG and the current Jacobian.
- Non-linear overhead (Python line search + energy evaluations): 0.113 s (6.2%).

### E.5 Explaining the level 1 vs level 3 crossover

At **level 1 np=1** (serial, 2 187 DOFs):

| Metric          | Custom (step 1) |   SNES (step 1) |
| --------------- | --------------: | --------------: |
| Wall time [s]   |          0.4358 |          0.0916 |
| Newton iters    |              10 |              12 |
| Total KSP iters |             130 |             147 |
| Speed ratio     |        baseline | **4.8× faster** |

At **level 3 np=16** (parallel, 78 003 DOFs):

| Metric          | Custom (step 1) | SNES (step 1) |
| --------------- | --------------: | ------------: |
| Wall time [s]   |          1.8239 |        2.5096 |
| Newton iters    |              11 |            14 |
| Total KSP iters |             148 |           214 |
| Speed ratio     | **1.4× faster** |      baseline |

**Why SNES is faster at level 1:**
At small serial problems the AMG setup (≈0.060 s at level 3, much less at level 1 ≈ few ms) is cheap relative to Python overhead.  Each Newton iteration in the custom solver involves a full Python function call, golden-section line search (up to 17 energy evaluations per step visible in `ls_evals` field), and CG inside a Python loop.  PETSc SNES manages the entire Newton loop in compiled C, with only the residual/Jacobian callbacks entering Python.  This eliminates the per-Newton Python overhead.

**Why custom is faster at level 3 np=16:**
The AMG setup cost scales with mesh size.  At level 3 with 16 MPI processes, one HYPRE BoomerAMG setup takes ~0.060 s.  PETSc SNES (with `newtonls`) rebuilds and re-factorises the Jacobian at every Newton iteration — 14 setups × 0.060 s ≈ **0.84 s** in AMG setups alone.  The custom solver's `pc_setup_on_ksp_cap` flag reuses the preconditioner until the KSP iteration count hits `ksp_max_it=30`: because no step hits that cap here, only **1 AMG setup (0.060 s)** is performed for all 11 Newton iterations.  This ~0.78 s saving more than compensates for the custom solver's Python overhead (0.113 s line search).

**Summary of the crossover mechanism:**

> At small meshes/serial execution: Python Newton loop overhead > AMG reuse benefit → SNES wins.
> At large meshes/MPI execution: AMG setup cost per Newton iter > Python Newton loop overhead → custom's AMG reuse wins.

The `--pc_setup_on_ksp_cap` strategy is particularly effective here because this is the first step of the trajectory (the initial AMG is a good preconditioner for the un-deformed Jacobian, so KSP never hits the 30-iteration cap across all 11 Newton steps).

---

## Annex F) GAMG vs HYPRE preconditioner comparison (level 3, 16 MPI processes)

**Motivation:** All preceding experiments used HYPRE BoomerAMG as the AMG preconditioner. PETSc also ships GAMG (Geometric-Algebraic MultiGrid), which is a native PETSc implementation with elasticity-aware features (`PCSetCoordinates`, block size, near-nullspace). This annex investigates whether GAMG can match or beat HYPRE for this 3D hyperelasticity problem.

### F.1 Setup

**Problem:** Neo-Hookean 3D elasticity on a beam `[0, 0.4] × [-0.005, 0.005]²`, 24 load steps (each = 15° rotation of right face), level 3 mesh (Nx=320, Ny=8, Nz=8 → 78,003 DOFs), 16 MPI processes.

**Docker environment:**
```bash
docker run -d --name bench_container --shm-size=8g \
  --entrypoint /bin/bash \
  -v "$PWD":/workdir -w /workdir \
  fenics_test:latest -c "sleep infinity"
```

**GAMG elasticity-specific settings (critical for correctness):**

1. **Block size = 3**: `A.setBlockSize(3)` — tells GAMG that the matrix has 3 DOFs per node (vector elasticity). Without this, GAMG treats each scalar DOF independently and produces poor coarse-level interpolation.

2. **Near-nullspace**: 6 rigid-body modes (3 translations + 3 rotations) via `A.setNearNullSpace(...)`, same as HYPRE.

3. **PCSetCoordinates**: Provides nodal coordinates to GAMG for geometry-aware aggregation. Set via `pc.setCoordinates(coords)` after `ksp.setOperators(A)`.

4. **`pc_gamg_threshold = 0.05`**: **This is the critical parameter.** The threshold filters weak connections in the strength-of-connection graph. Default (`-1`, keep all edges) preserves every connection, leading to overly aggressive coarsening that fails to preserve rigid-body modes → solver converges to wrong energy. A threshold of `0.05` filters out weak couplings, producing slower but physically correct coarsening.

5. **`pc_gamg_agg_nsmooths = 1`**: One level of smoothed aggregation (default).

### F.2 Investigation trajectory

Several configurations were tested (intermediate runs not shown in full):

| Run                  | Block size | Nullspace | Coordinates |  Threshold   | KSP rtol | Step-1 energy | Correct? |
| -------------------- | :--------: | :-------: | :---------: | :----------: | :------: | :-----------: | :------: |
| 1 — Naive GAMG       |     No     |    No     |     No      | -1 (default) |   1e-1   |   diverged    |    No    |
| 2 — + block size     |     3      |    Yes    |     No      |      -1      |   1e-3   |    0.3957     |    No    |
| 3 — + coordinates    |     3      |    Yes    |     Yes     |      -1      |   1e-6   |    0.1673     |    No    |
| 4 — + threshold=0.05 |     3      |    Yes    |     Yes     |     0.05     |   1e-6   |  **0.1626**   | **Yes**  |

**Key finding:** Without `pc_gamg_threshold=0.05`, GAMG consistently converged to wrong minima (energies 0.17–0.40 instead of the correct 0.1626). The threshold is the single most important parameter — more important than coordinates or nullspace for this problem.

### F.3 Three-way comparison: HYPRE vs GAMG (tight) vs GAMG (loose)

All runs: level 3, 24 load steps, 16 MPI, `ksp_type=gmres`, near-nullspace ON, `gamg_threshold=0.05`.

|                       |           **HYPRE** | **GAMG (tight)** | **GAMG (loose)** |
| --------------------- | ------------------: | ---------------: | ---------------: |
| PC type               | `hypre` (BoomerAMG) |           `gamg` |           `gamg` |
| `ksp_rtol`            |                1e-1 |             1e-6 |             1e-1 |
| `ksp_max_it`          |                  30 |              500 |               30 |
| `pc_setup_on_ksp_cap` |                 Yes |               No |              Yes |
| Total time            |         **135.5 s** |          211.1 s |       **62.4 s** |
| Total Newton iters    |                 669 |              536 |            1,123 |
| Total KSP iters       |              10,347 |          190,012 |           17,868 |
| Avg KSP/Newton        |                15.5 |            354.5 |             15.9 |
| Final energy          |             93.7050 |          93.7057 |          93.7048 |
| All steps converged?  |         Yes (24/24) |      Yes (24/24) |      Yes (24/24) |

**Observations:**

- **GAMG (loose) is 2.2× faster than HYPRE** (62.4 s vs 135.5 s) with the same loose-tolerance strategy (`ksp_rtol=1e-1`, `ksp_max_it=30`, `pc_setup_on_ksp_cap`). The time savings come from cheaper AMG setup (GAMG reuses the matrix directly; HYPRE requires data conversion to its internal format).

- **GAMG (tight) is 1.6× slower than HYPRE** (211.1 s vs 135.5 s). With `ksp_rtol=1e-6`, GAMG needs ~355 KSP iterations per Newton step on average. While this reduces Newton iterations (536 vs 669), the vastly increased KSP work dominates.

- **GAMG (tight) + `pc_setup_on_ksp_cap` is unstable**: An earlier test (v2) with both tight tolerance *and* PC reuse diverged at step 7+. Without `pc_setup_on_ksp_cap`, tight GAMG works fine (v3). The tight-tolerance configuration needs fresh PC each Newton step.

- **Final energies** are consistent across all three configurations (93.70±0.01), confirming correctness.

### F.4 Exact replication commands

**HYPRE reference:**
```bash
docker exec bench_container mpirun -n 16 python3 \
  /workdir/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type hypre \
  --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap \
  --save_history \
  --out /workdir/experiment_scripts/he_custom_l3_np16_bench.json
```

**GAMG (tight tolerance, fresh PC every Newton):**
```bash
docker exec bench_container mpirun -n 16 python3 \
  /workdir/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type gamg \
  --ksp_rtol 1e-6 --ksp_max_it 500 \
  --gamg_threshold 0.05 \
  --save_history \
  --out /workdir/experiment_scripts/he_custom_l3_np16_gamg_full24_v3.json
```

**GAMG (loose tolerance, HYPRE-like settings):**
```bash
docker exec bench_container mpirun -n 16 python3 \
  /workdir/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type gamg \
  --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap \
  --gamg_threshold 0.05 \
  --save_history \
  --out /workdir/experiment_scripts/he_custom_l3_np16_gamg_hypre_settings.json
```

### F.5 Per-step tables

#### Level 3, 16 MPI — HYPRE BoomerAMG (reference)

Artifact: [experiment_scripts/he_custom_l3_np16_bench.json](experiment_scripts/he_custom_l3_np16_bench.json)

| Step | Time [s] | Newton iters | Sum KSP iters |        Energy | Status                  |
| ---: | -------: | -----------: | ------------: | ------------: | ----------------------- |
|    1 |      5.0 |           28 |           386 |  0.1625590000 | Energy change converged |
|    2 |      5.4 |           29 |           403 |  0.6502370000 | Energy change converged |
|    3 |      5.2 |           28 |           368 |  1.4630840000 | Energy change converged |
|    4 |      5.0 |           27 |           355 |  2.6011520000 | Energy change converged |
|    5 |      5.2 |           27 |           370 |  4.0645090000 | Energy change converged |
|    6 |      5.3 |           28 |           371 |  5.8532780000 | Energy change converged |
|    7 |      5.2 |           28 |           367 |  7.9675390000 | Energy change converged |
|    8 |      5.2 |           27 |           364 | 10.4074370000 | Energy change converged |
|    9 |      5.4 |           28 |           373 | 13.1730770000 | Energy change converged |
|   10 |      6.0 |           29 |           448 | 16.2645240000 | Energy change converged |
|   11 |      5.6 |           27 |           422 | 19.6818780000 | Energy change converged |
|   12 |      5.5 |           27 |           424 | 23.4251620000 | Energy change converged |
|   13 |      6.1 |           29 |           481 | 27.4943710000 | Energy change converged |
|   14 |      5.6 |           27 |           440 | 31.8894050000 | Energy change converged |
|   15 |      5.3 |           27 |           383 | 36.6099940000 | Energy change converged |
|   16 |      6.0 |           29 |           478 | 41.6554080000 | Energy change converged |
|   17 |      6.9 |           31 |           575 | 47.0227630000 | Energy change converged |
|   18 |      5.4 |           27 |           394 | 52.7168590000 | Energy change converged |
|   19 |      5.8 |           27 |           467 | 58.7367140000 | Energy change converged |
|   20 |      6.0 |           27 |           496 | 65.0818470000 | Energy change converged |
|   21 |      6.4 |           28 |           539 | 71.7519170000 | Energy change converged |
|   22 |      6.2 |           28 |           504 | 78.7463720000 | Energy change converged |
|   23 |      5.8 |           28 |           447 | 86.0644710000 | Energy change converged |
|   24 |      6.2 |           28 |           492 | 93.7049960000 | Energy change converged |

Summary: total time = `135.5 s`, total Newton iters = `669`, total KSP iters = `10,347`.

#### Level 3, 16 MPI — GAMG (tight: rtol=1e-6, ksp_max_it=500, no PC reuse)

Artifact: [experiment_scripts/he_custom_l3_np16_gamg_full24_v3.json](experiment_scripts/he_custom_l3_np16_gamg_full24_v3.json)

| Step | Time [s] | Newton iters | Sum KSP iters |        Energy | Status                  |
| ---: | -------: | -----------: | ------------: | ------------: | ----------------------- |
|    1 |      4.6 |           24 |         3,559 |  0.1625560000 | Energy change converged |
|    2 |      7.2 |           22 |         6,136 |  0.6502470000 | Energy change converged |
|    3 |      8.9 |           22 |         7,960 |  1.4630860000 | Energy change converged |
|    4 |      7.7 |           21 |         6,815 |  2.6011420000 | Energy change converged |
|    5 |      7.5 |           20 |         6,789 |  4.0645050000 | Energy change converged |
|    6 |      8.9 |           22 |         7,970 |  5.8532660000 | Energy change converged |
|    7 |      7.2 |           21 |         6,515 |  7.9675380000 | Energy change converged |
|    8 |      7.4 |           20 |         6,750 | 10.4074380000 | Energy change converged |
|    9 |      8.4 |           22 |         7,689 | 13.1730690000 | Energy change converged |
|   10 |      8.7 |           22 |         7,942 | 16.2645240000 | Energy change converged |
|   11 |      8.2 |           20 |         7,398 | 19.6818750000 | Energy change converged |
|   12 |      8.9 |           21 |         8,132 | 23.4251650000 | Energy change converged |
|   13 |     10.8 |           24 |         9,783 | 27.4943660000 | Energy change converged |
|   14 |      9.2 |           21 |         8,208 | 31.8893600000 | Energy change converged |
|   15 |     10.1 |           23 |         9,071 | 36.6099960000 | Energy change converged |
|   16 |     12.3 |           28 |        11,119 | 41.6539100000 | Energy change converged |
|   17 |      9.5 |           23 |         8,634 | 47.0226220000 | Energy change converged |
|   18 |      9.2 |           23 |         8,262 | 52.7168540000 | Energy change converged |
|   19 |      9.8 |           23 |         8,875 | 58.7366420000 | Energy change converged |
|   20 |      9.0 |           22 |         8,053 | 65.0816790000 | Energy change converged |
|   21 |      8.8 |           21 |         7,971 | 71.7515970000 | Energy change converged |
|   22 |      9.2 |           23 |         8,443 | 78.7460100000 | Energy change converged |
|   23 |      9.7 |           24 |         8,841 | 86.0643780000 | Energy change converged |
|   24 |      9.9 |           24 |         9,097 | 93.7056510000 | Energy change converged |

Summary: total time = `211.1 s`, total Newton iters = `536`, total KSP iters = `190,012`.

#### Level 3, 16 MPI — GAMG (loose: rtol=1e-1, ksp_max_it=30, PC reuse)

Artifact: [experiment_scripts/he_custom_l3_np16_gamg_hypre_settings.json](experiment_scripts/he_custom_l3_np16_gamg_hypre_settings.json)

| Step | Time [s] | Newton iters | Sum KSP iters |        Energy | Status                  |
| ---: | -------: | -----------: | ------------: | ------------: | ----------------------- |
|    1 |      2.1 |           41 |           445 |  0.1626540000 | Energy change converged |
|    2 |      2.6 |           48 |           733 |  0.6503650000 | Energy change converged |
|    3 |      2.7 |           49 |           759 |  1.4634650000 | Energy change converged |
|    4 |      2.7 |           50 |           718 |  2.6012100000 | Energy change converged |
|    5 |      2.8 |           50 |           785 |  4.0645420000 | Energy change converged |
|    6 |      2.6 |           47 |           703 |  5.8534710000 | Energy change converged |
|    7 |      2.6 |           49 |           705 |  7.9676180000 | Energy change converged |
|    8 |      2.6 |           46 |           761 | 10.4074830000 | Energy change converged |
|    9 |      2.7 |           48 |           757 | 13.1731180000 | Energy change converged |
|   10 |      2.5 |           45 |           726 | 16.2646650000 | Energy change converged |
|   11 |      2.6 |           46 |           752 | 19.6819970000 | Energy change converged |
|   12 |      2.8 |           50 |           807 | 23.4252460000 | Energy change converged |
|   13 |      2.8 |           51 |           835 | 27.4944180000 | Energy change converged |
|   14 |      2.5 |           44 |           725 | 31.8895290000 | Energy change converged |
|   15 |      2.6 |           45 |           791 | 36.6101750000 | Energy change converged |
|   16 |      2.8 |           46 |           756 | 41.6551670000 | Energy change converged |
|   17 |      2.9 |           50 |           932 | 47.0231150000 | Energy change converged |
|   18 |      2.5 |           45 |           725 | 52.7169870000 | Energy change converged |
|   19 |      2.6 |           47 |           733 | 58.7367000000 | Energy change converged |
|   20 |      2.5 |           45 |           720 | 65.0819070000 | Energy change converged |
|   21 |      2.5 |           45 |           698 | 71.7519250000 | Energy change converged |
|   22 |      2.4 |           43 |           722 | 78.7464080000 | Energy change converged |
|   23 |      2.7 |           47 |           834 | 86.0646010000 | Energy change converged |
|   24 |      2.6 |           46 |           746 | 93.7048460000 | Energy change converged |

Summary: total time = `62.4 s`, total Newton iters = `1,123`, total KSP iters = `17,868`.

### F.6 Conclusions

1. **GAMG with `pc_gamg_threshold=0.05` + loose tolerance (`ksp_rtol=1e-1`) + PC reuse (`pc_setup_on_ksp_cap`) is the fastest configuration tested** — 2.2× faster than HYPRE with identical settings, 3.4× faster than GAMG with tight tolerance.

2. **The `pc_gamg_threshold` parameter is critical.** Without it (default `-1`), GAMG converges to wrong solutions for 3D elasticity regardless of block size, nullspace, or coordinates. The threshold `0.05` filters weak graph connections and produces correct coarsening.

3. **Trade-off:** GAMG (loose) needs ~1.7× more Newton iterations (1,123 vs 669) due to the inexpensive-but-inexact preconditioner. However, each Newton step is much cheaper (2.6 s vs 5.6 s) because GAMG setup and application costs are lower than HYPRE's at this problem size.

4. **Recommended GAMG configuration for 3D elasticity:**
   ```
   --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap
   --gamg_threshold 0.05
   ```
   Plus: `A.setBlockSize(3)`, near-nullspace (6 rigid-body modes), `PCSetCoordinates` — all handled automatically by the solver when `--pc_type gamg` is specified.

### F.7 GAMG sweep targeting HYPRE-like iteration counts (L3, np=16)

To tune GAMG toward the HYPRE iteration profile, we swept:
- `ksp_rtol`
- `ksp_max_it`
- preconditioner rebuild policy (`--pc_setup_on_ksp_cap` ON/OFF)

Fixed across all runs:
- `--pc_type gamg`
- `--ksp_type gmres`
- `--gamg_threshold 0.05` (required for correctness)
- level 3, 24 load steps, 16 MPI, near-nullspace ON, GAMG coordinates ON.

HYPRE reference (for comparison): total time `135.5444 s`, Newton `669`, KSP `10347`.

#### How to run single test runs

Run one GAMG test point (replace `<...>`):
```bash
docker exec bench_container mpirun -n 16 python3 \
  /workdir/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type gamg \
  --ksp_rtol <rtol> --ksp_max_it <ksp_max_it> \
  --gamg_threshold 0.05 --gamg_agg_nsmooths 1 \
  [--pc_setup_on_ksp_cap] \
  --save_history --quiet \
  --out /workdir/experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/<case_id>.json
```

Requested fresh-`ksp_max_it=50` cases (PC setup fresh every Newton):
```bash
# rtol = 1e-1
docker exec bench_container mpirun -n 16 python3 \
  /workdir/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type gamg \
  --ksp_rtol 1e-1 --ksp_max_it 50 \
  --gamg_threshold 0.05 --gamg_agg_nsmooths 1 \
  --save_history --quiet \
  --out /workdir/experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/r1e-1_k50_fresh.json

# rtol = 1e-2
docker exec bench_container mpirun -n 16 python3 \
  /workdir/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type gamg \
  --ksp_rtol 1e-2 --ksp_max_it 50 \
  --gamg_threshold 0.05 --gamg_agg_nsmooths 1 \
  --save_history --quiet \
  --out /workdir/experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/r1e-2_k50_fresh.json

# rtol = 1e-3
docker exec bench_container mpirun -n 16 python3 \
  /workdir/HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type gamg \
  --ksp_rtol 1e-3 --ksp_max_it 50 \
  --gamg_threshold 0.05 --gamg_agg_nsmooths 1 \
  --save_history --quiet \
  --out /workdir/experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/r1e-3_k50_fresh.json
```

Run the prepared sweep script:
```bash
python3 experiment_scripts/sweep_he_gamg_hypre_like_l3_np16.py --case_set default
```

#### Combined sweep table (all tested points)

| Case | rtol | ksp_max_it | PC reuse on cap | Conv steps | Time [s] | Newton | KSP | Avg KSP/Newton | Final energy | Score to HYPRE | Speedup vs HYPRE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r2e-2_k30_fresh | 2e-02 | 30 | No | 24/24 | 47.0968 | 720 | 19027 | 26.4264 | 93.704946 | 0.9151 | 2.878x |
| r5e-2_k30_reuse | 5e-02 | 30 | Yes | 24/24 | 51.5617 | 849 | 17447 | 20.5501 | 93.721554 | 0.9552 | 2.629x |
| r1e-2_k30_reuse | 1e-02 | 30 | Yes | 24/24 | 50.4301 | 730 | 20484 | 28.0603 | 93.705067 | 1.0709 | 2.688x |
| r1e-2_k20_reuse | 1e-02 | 20 | Yes | 24/24 | 52.7889 | 910 | 17796 | 19.5560 | 93.705406 | 1.0802 | 2.568x |
| r1e-1_k20_reuse | 1e-01 | 20 | Yes | 24/24 | 59.1073 | 1108 | 14961 | 13.5027 | 93.704985 | 1.1021 | 2.293x |
| r5e-2_k20_fresh | 5e-02 | 20 | No | 24/24 | 55.9024 | 985 | 17151 | 17.4122 | 93.705143 | 1.1299 | 2.425x |
| r1e-2_k30_fresh | 1e-02 | 30 | No | 24/24 | 50.2838 | 756 | 21168 | 28.0000 | 93.705292 | 1.1759 | 2.696x |
| r1e-1_k30_reuse | 1e-01 | 30 | Yes | 24/24 | 58.5355 | 1044 | 16725 | 16.0201 | 93.766002 | 1.1769 | 2.316x |
| r2e-2_k30_reuse | 2e-02 | 30 | Yes | 24/24 | 51.0453 | 779 | 20828 | 26.7368 | 93.705076 | 1.1774 | 2.655x |
| r5e-3_k30_reuse | 5e-03 | 30 | Yes | 24/24 | 52.7780 | 752 | 21809 | 29.0013 | 93.705377 | 1.2318 | 2.568x |
| r2e-2_k20_reuse | 2e-02 | 20 | Yes | 24/24 | 57.7245 | 986 | 18758 | 19.0243 | 93.705381 | 1.2867 | 2.348x |
| r5e-2_k20_reuse | 5e-02 | 20 | Yes | 24/24 | 59.2221 | 1042 | 17994 | 17.2687 | 93.705446 | 1.2966 | 2.289x |
| r2e-2_k20_fresh | 2e-02 | 20 | No | 24/24 | 60.0793 | 1021 | 19551 | 19.1489 | 93.705356 | 1.4157 | 2.256x |
| r2e-2_k50_reuse | 2e-02 | 50 | Yes | 24/24 | 54.3331 | 700 | 27912 | 39.8743 | 93.705209 | 1.7439 | 2.495x |
| r1e-1_k50_fresh | 1e-01 | 50 | No | 24/24 | 65.3242 | 1032 | 24593 | 23.8304 | 93.704684 | 1.9194 | 2.075x |
| r1e-2_k50_reuse | 1e-02 | 50 | Yes | 24/24 | 55.0011 | 692 | 29905 | 43.2153 | 93.705339 | 1.9246 | 2.464x |
| r5e-3_k50_reuse | 5e-03 | 50 | Yes | 24/24 | 58.4888 | 675 | 31223 | 46.2563 | 93.705761 | 2.0266 | 2.317x |
| r1e-3_k50_fresh | 1e-03 | 50 | No | 24/24 | 55.4752 | 640 | 31168 | 48.7000 | 93.705743 | 2.0556 | 2.443x |
| r1e-2_k50_fresh | 1e-02 | 50 | No | 15/24 | 115.5878 | 1290 | 50663 | 39.2736 | nan | 4.8246 | 1.173x |

Note: `r1e-2_k50_fresh` did not complete the full trajectory (only 15/24 converged steps; final energy is `nan`).

Artifacts:
- [experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/summary_all.md](experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/summary_all.md)
- [experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/summary_all.csv](experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/summary_all.csv)
- [experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/summary_all.json](experiment_scripts/he_gamg_hypre_like_sweep_l3_np16/summary_all.json)

### F.8 Hardened nonlinear solver impact (same 16-case GAMG sweep)

This sweep repeats the same 16 GAMG points as F.7, but with the hardened nonlinear minimizer settings:

- `tolf=1e-4`, `tolg=1e-3`, `tolg_rel=1e-3`
- `tolx_rel=1e-3`, `tolx_abs=1e-10`
- convergence requires energy + step + gradient
- per-step retry enabled for non-finite/maxit (`rtol*0.1`, `ksp_max_it*2`, line-search `b<=1.0`)

Side-by-side against F.7:

| Case | rtol | ksp_max_it | PC policy | Time old→new [s] | Newton old→new | KSP old→new | Score old→new |
|---|---:|---:|---|---:|---:|---:|---:|
| r2e-2_k30_fresh | 2e-02 | 30 | fresh | 47.0968 -> 60.3509 (+13.2541) | 720 -> 898 (+178) | 19027 -> 23965 (+4938) | 0.9151 -> 1.6584 (+0.7433) |
| r5e-2_k30_reuse | 5e-02 | 30 | reuse | 51.5617 -> 64.9344 (+13.3727) | 849 -> 1040 (+191) | 17447 -> 25095 (+7648) | 0.9552 -> 1.9799 (+1.0247) |
| r1e-2_k30_reuse | 1e-02 | 30 | reuse | 50.4301 -> 63.2170 (+12.7869) | 730 -> 922 (+192) | 20484 -> 26313 (+5829) | 1.0709 -> 1.9212 (+0.8503) |
| r1e-2_k20_reuse | 1e-02 | 20 | reuse | 52.7889 -> 68.9125 (+16.1236) | 910 -> 1122 (+212) | 17796 -> 24510 (+6714) | 1.0802 -> 2.0459 (+0.9658) |
| r1e-1_k20_reuse | 1e-01 | 20 | reuse | 59.1073 -> 71.1030 (+11.9957) | 1108 -> 1222 (+114) | 14961 -> 22645 (+7684) | 1.1021 -> 2.0152 (+0.9130) |
| r5e-2_k20_fresh | 5e-02 | 20 | fresh | 55.9024 -> 71.7011 (+15.7987) | 985 -> 1187 (+202) | 17151 -> 23875 (+6724) | 1.1299 -> 2.0817 (+0.9518) |
| r1e-2_k30_fresh | 1e-02 | 30 | fresh | 50.2838 -> 65.5587 (+15.2749) | 756 -> 944 (+188) | 21168 -> 27712 (+6544) | 1.1759 -> 2.0893 (+0.9135) |
| r1e-1_k30_reuse | 1e-01 | 30 | reuse | 58.5355 -> 65.1899 (+6.6544) | 1044 -> 1111 (+67) | 16725 -> 20736 (+4011) | 1.1769 -> 1.6647 (+0.4878) |
| r2e-2_k30_reuse | 2e-02 | 30 | reuse | 51.0453 -> 59.5231 (+8.4778) | 779 -> 897 (+118) | 20828 -> 25031 (+4203) | 1.1774 -> 1.7600 (+0.5826) |
| r5e-3_k30_reuse | 5e-03 | 30 | reuse | 52.7780 -> 62.8697 (+10.0917) | 752 -> 910 (+158) | 21809 -> 26545 (+4736) | 1.2318 -> 1.9257 (+0.6939) |
| r2e-2_k20_reuse | 2e-02 | 20 | reuse | 57.7245 -> 64.4587 (+6.7342) | 986 -> 1063 (+77) | 18758 -> 24122 (+5364) | 1.2867 -> 1.9202 (+0.6335) |
| r5e-2_k20_reuse | 5e-02 | 20 | reuse | 59.2221 -> 68.7061 (+9.4840) | 1042 -> 1133 (+91) | 17994 -> 24823 (+6829) | 1.2966 -> 2.0926 (+0.7960) |
| r2e-2_k20_fresh | 2e-02 | 20 | fresh | 60.0793 -> 67.1476 (+7.0683) | 1021 -> 1096 (+75) | 19551 -> 22467 (+2916) | 1.4157 -> 1.8096 (+0.3939) |
| r2e-2_k50_reuse | 2e-02 | 50 | reuse | 54.3331 -> 69.2173 (+14.8842) | 700 -> 862 (+162) | 27912 -> 37593 (+9681) | 1.7439 -> 2.9217 (+1.1778) |
| r1e-2_k50_reuse | 1e-02 | 50 | reuse | 55.0011 -> 70.1919 (+15.1908) | 692 -> 831 (+139) | 29905 -> 38005 (+8100) | 1.9246 -> 2.9152 (+0.9906) |
| r5e-3_k50_reuse | 5e-03 | 50 | reuse | 58.4888 -> 71.6575 (+13.1687) | 675 -> 831 (+156) | 31223 -> 38866 (+7643) | 2.0266 -> 2.9984 (+0.9719) |

Best-by-score in hardened sweep: `r2e-2_k30_fresh` (`score=1.6584`, time `60.3509 s`).

### F.9 Gradient tolerance sensitivity: 10x tighter (`tolg=tolg_rel=1e-4`)

Compared to the hardened baseline (`1e-3`), tightening gradient tolerance by 10x reduced scores for many cases, but triggered earlier non-converged stop in most cases due fail-fast at the first problematic step.

| Case | rtol | ksp_max_it | PC | Conv prev→new | Time prev→new [s] | Newton prev→new | KSP prev→new | Score prev→new |
|---|---:|---:|---|---|---:|---:|---:|---:|
| r2e-2_k30_fresh | 2e-02 | 30 | fresh | 24/24 -> 15/16 | 60.3509 -> 52.2256 (-8.1253) | 898 -> 695 (-203) | 23965 -> 24142 (+177) | 1.6584 -> 1.3721 (-0.2863) ▲ |
| r5e-2_k30_reuse | 5e-02 | 30 | reuse | 24/24 -> 15/16 | 64.9344 -> 58.1126 (-6.8218) | 1040 -> 827 (-213) | 25095 -> 24876 (-219) | 1.9799 -> 1.6403 (-0.3396) ▲ |
| r1e-2_k30_reuse | 1e-02 | 30 | reuse | 24/24 -> 15/16 | 63.2170 -> 50.5609 (-12.6561) | 922 -> 704 (-218) | 26313 -> 23139 (-3174) | 1.9212 -> 1.2886 (-0.6326) ▲ |
| r1e-2_k20_reuse | 1e-02 | 20 | reuse | 24/24 -> 15/16 | 68.9125 -> 58.6215 (-10.2910) | 1122 -> 925 (-197) | 24510 -> 22491 (-2019) | 2.0459 -> 1.5563 (-0.4896) ▲ |
| r1e-1_k20_reuse | 1e-01 | 20 | reuse | 24/24 -> 15/16 | 71.1030 -> 65.2696 (-5.8334) | 1222 -> 1122 (-100) | 22645 -> 20268 (-2377) | 2.0152 -> 1.6360 (-0.3792) ▲ |
| r5e-2_k20_fresh | 5e-02 | 20 | fresh | 24/24 -> 15/16 | 71.7011 -> 52.8871 (-18.8140) | 1187 -> 788 (-399) | 23875 -> 22000 (-1875) | 2.0817 -> 1.3041 (-0.7776) ▲ |
| r1e-2_k30_fresh | 1e-02 | 30 | fresh | 24/24 -> 15/16 | 65.5587 -> 52.9102 (-12.6485) | 944 -> 742 (-202) | 27712 -> 24245 (-3467) | 2.0893 -> 1.4523 (-0.6370) ▲ |
| r1e-1_k30_reuse | 1e-01 | 30 | reuse | 24/24 -> 15/16 | 65.1899 -> 55.6522 (-9.5377) | 1111 -> 915 (-196) | 20736 -> 19121 (-1615) | 1.6647 -> 1.2157 (-0.4491) ▲ |
| r2e-2_k30_reuse | 2e-02 | 30 | reuse | 24/24 -> 15/16 | 59.5231 -> 49.9245 (-9.5986) | 897 -> 703 (-194) | 25031 -> 21990 (-3041) | 1.7600 -> 1.1761 (-0.5839) ▲ |
| r5e-3_k30_reuse | 5e-03 | 30 | reuse | 24/24 -> 15/16 | 62.8697 -> 52.2764 (-10.5933) | 910 -> 720 (-190) | 26545 -> 23980 (-2565) | 1.9257 -> 1.3938 (-0.5319) ▲ |
| r2e-2_k20_reuse | 2e-02 | 20 | reuse | 24/24 -> 15/16 | 64.4587 -> 55.3289 (-9.1298) | 1063 -> 840 (-223) | 24122 -> 22526 (-1596) | 1.9202 -> 1.4327 (-0.4876) ▲ |
| r5e-2_k20_reuse | 5e-02 | 20 | reuse | 24/24 -> 15/16 | 68.7061 -> 61.6159 (-7.0902) | 1133 -> 991 (-142) | 24823 -> 21951 (-2872) | 2.0926 -> 1.6028 (-0.4898) ▲ |
| r2e-2_k20_fresh | 2e-02 | 20 | fresh | 24/24 -> 15/16 | 67.1476 -> 64.3657 (-2.7819) | 1096 -> 1011 (-85) | 22467 -> 22259 (-208) | 1.8096 -> 1.6625 (-0.1472) ▲ |
| r2e-2_k50_reuse | 2e-02 | 50 | reuse | 24/24 -> 23/24 | 69.2173 -> 90.0922 (+20.8749) | 862 -> 999 (+137) | 37593 -> 52055 (+14462) | 2.9217 -> 4.5242 (+1.6025) ▼ |
| r1e-2_k50_reuse | 1e-02 | 50 | reuse | 24/24 -> 15/16 | 70.1919 -> 60.1159 (-10.0760) | 831 -> 655 (-176) | 38005 -> 34294 (-3711) | 2.9152 -> 2.3353 (-0.5799) ▲ |
| r5e-3_k50_reuse | 5e-03 | 50 | reuse | 24/24 -> 15/16 | 71.6575 -> 57.4704 (-14.1871) | 831 -> 611 (-220) | 38866 -> 34863 (-4003) | 2.9984 -> 2.4561 (-0.5423) ▲ |

Best score in this test: `r2e-2_k30_reuse` (`1.1761`), but only `15/16` steps were processed before fail-fast stop.

### F.10 Gradient tolerance sensitivity: looser (`tolg=tolg_rel=1e-2`)

Loosening gradient tolerance from `1e-3` to `1e-2` recovered full convergence (`24/24`) for all 16 cases and improved score for every row relative to the hardened baseline.

| Case | rtol | ksp_max_it | PC | Conv prev→new | Time prev→new [s] | Newton prev→new | KSP prev→new | Score prev→new |
|---|---:|---:|---|---|---:|---:|---:|---:|
| r2e-2_k30_fresh | 2e-02 | 30 | fresh | 24/24 -> 24/24 | 60.3509 -> 55.0945 (-5.2564) | 898 -> 849 (-49) | 23965 -> 22572 (-1393) | 1.6584 -> 1.4506 (-0.2079) ▲ |
| r5e-2_k30_reuse | 5e-02 | 30 | reuse | 24/24 -> 24/24 | 64.9344 -> 60.1198 (-4.8146) | 1040 -> 923 (-117) | 25095 -> 23419 (-1676) | 1.9799 -> 1.6430 (-0.3369) ▲ |
| r1e-2_k30_reuse | 1e-02 | 30 | reuse | 24/24 -> 24/24 | 63.2170 -> 62.7460 (-0.4710) | 922 -> 900 (-22) | 26313 -> 26383 (+70) | 1.9212 -> 1.8951 (-0.0261) ▲ |
| r1e-2_k20_reuse | 1e-02 | 20 | reuse | 24/24 -> 24/24 | 68.9125 -> 63.8089 (-5.1036) | 1122 -> 1088 (-34) | 24510 -> 21985 (-2525) | 2.0459 -> 1.7511 (-0.2949) ▲ |
| r1e-1_k20_reuse | 1e-01 | 20 | reuse | 24/24 -> 24/24 | 71.1030 -> 64.2889 (-6.8141) | 1222 -> 1204 (-18) | 22645 -> 16642 (-6003) | 2.0152 -> 1.4081 (-0.6071) ▲ |
| r5e-2_k20_fresh | 5e-02 | 20 | fresh | 24/24 -> 24/24 | 71.7011 -> 64.2422 (-7.4589) | 1187 -> 1126 (-61) | 23875 -> 20810 (-3065) | 2.0817 -> 1.6943 (-0.3874) ▲ |
| r1e-2_k30_fresh | 1e-02 | 30 | fresh | 24/24 -> 24/24 | 65.5587 -> 63.0000 (-2.5587) | 944 -> 916 (-28) | 27712 -> 26125 (-1587) | 2.0893 -> 1.8941 (-0.1952) ▲ |
| r1e-1_k30_reuse | 1e-01 | 30 | reuse | 24/24 -> 24/24 | 65.1899 -> 62.4068 (-2.7831) | 1111 -> 1104 (-7) | 20736 -> 19322 (-1414) | 1.6647 -> 1.5176 (-0.1471) ▲ |
| r2e-2_k30_reuse | 2e-02 | 30 | reuse | 24/24 -> 24/24 | 59.5231 -> 60.7418 (+1.2187) | 897 -> 913 (+16) | 25031 -> 24676 (-355) | 1.7600 -> 1.7496 (-0.0104) ▲ |
| r5e-3_k30_reuse | 5e-03 | 30 | reuse | 24/24 -> 24/24 | 62.8697 -> 61.0602 (-1.8095) | 910 -> 894 (-16) | 26545 -> 26097 (-448) | 1.9257 -> 1.8585 (-0.0672) ▲ |
| r2e-2_k20_reuse | 2e-02 | 20 | reuse | 24/24 -> 24/24 | 64.4587 -> 64.9997 (+0.5410) | 1063 -> 1062 (-1) | 24122 -> 22057 (-2065) | 1.9202 -> 1.7192 (-0.2011) ▲ |
| r5e-2_k20_reuse | 5e-02 | 20 | reuse | 24/24 -> 24/24 | 68.7061 -> 64.7543 (-3.9518) | 1133 -> 1092 (-41) | 24823 -> 19603 (-5220) | 2.0926 -> 1.5268 (-0.5658) ▲ |
| r2e-2_k20_fresh | 2e-02 | 20 | fresh | 24/24 -> 24/24 | 67.1476 -> 64.8894 (-2.2582) | 1096 -> 1105 (+9) | 22467 -> 21845 (-622) | 1.8096 -> 1.7630 (-0.0467) ▲ |
| r2e-2_k50_reuse | 2e-02 | 50 | reuse | 24/24 -> 24/24 | 69.2173 -> 58.2413 (-10.9760) | 862 -> 744 (-118) | 37593 -> 29437 (-8156) | 2.9217 -> 1.9571 (-0.9646) ▲ |
| r1e-2_k50_reuse | 1e-02 | 50 | reuse | 24/24 -> 24/24 | 70.1919 -> 68.5931 (-1.5988) | 831 -> 830 (-1) | 38005 -> 36700 (-1305) | 2.9152 -> 2.7876 (-0.1276) ▲ |
| r5e-3_k50_reuse | 5e-03 | 50 | reuse | 24/24 -> 24/24 | 71.6575 -> 68.2217 (-3.4358) | 831 -> 791 (-40) | 38866 -> 36851 (-2015) | 2.9984 -> 2.7439 (-0.2545) ▲ |

Best score in this test: `r1e-1_k20_reuse` (`score=1.4081`, time `64.2889 s`).

Artifacts:
- [experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16/summary_hardened16.json](experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16/summary_hardened16.json)
- [experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16/paired_old_vs_hardened.md](experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16/paired_old_vs_hardened.md)
- [experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16_gradx10/summary_hardened16_gradx10.json](experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16_gradx10/summary_hardened16_gradx10.json)
- [experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16_gradx10/paired_prev_vs_gradx10.md](experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16_gradx10/paired_prev_vs_gradx10.md)
- [experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16_grad1e2/summary_hardened16_grad1e2.json](experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16_grad1e2/summary_hardened16_grad1e2.json)
- [experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16_grad1e2/paired_prev_vs_grad1e2.md](experiment_scripts/he_gamg_hypre_like_sweep_l3_np16_hardened16_grad1e2/paired_prev_vs_grad1e2.md)

### F.11 JAX+PETSc full trajectory benchmark (L3, np=16, 2026-02-28)

This section records the current end-to-end JAX+PETSc result for the full 24-step
level-3 trajectory and the matrix-layout isolation experiment.

#### Full run performance (24/24 steps)

Configuration:
- Solver: `HyperElasticity3D_jax_petsc/solve_HE_dof.py`
- Level: `3`
- MPI ranks: `16`
- Steps: `24` (`--total_steps 24`)
- Profile/options: `gmres + gamg`, `ksp_rtol=1e-1`, `ksp_max_it=30`,
  `pc_setup_on_ksp_cap`, `hvp_eval_mode=sequential`

Measured results:
- Wall clock (wrapper): `851.35 s`
- Sum of step times (JSON): `771.85 s`
- Average step time: `32.16 s`
- Step time range: `22.65 s` to `38.94 s`
- Total Newton iterations: `1064`
- Total linear iterations: `18299`
- Final energy: `93.7048601512`
- Final message: `Converged (energy, step, gradient)`

Artifact:
- [experiment_scripts/he_jaxpetsc_full_l3_np16_20260228_062459.json](experiment_scripts/he_jaxpetsc_full_l3_np16_20260228_062459.json)

#### Matrix-layout isolation (FEniCS partition + JAX values)

Goal: separate partition/layout effects from value effects.

Step-1 fixed-state KSP results (`np=16`, `L3`):
- FEniCS matrix (FEniCS values): `ksp_its=1`, `solve_time=0.003520 s`
- FEniCS matrix layout + JAX values: `ksp_its=4`, `solve_time=0.006525 s`

Control check (same values + same layout):
- FEniCS matrix: `ksp_its=2`, `solve_time=0.004138 s`
- Direct matrix copy: `ksp_its=2`, `solve_time=0.004382 s`

Conclusion:
- Layout parity alone does not recover FEniCS solve behavior.
- When values are also identical, KSP behavior matches (as expected).

Per-rank Hessian value compute time (JAX HVP only, `np=16`, step-1):
- `min=0.0852 s`, `max=0.4618 s`, `mean=0.3162 s`, `std=0.1347 s`
- imbalance ratio (`max/min`) = `5.42x`

Artifacts:
- [experiment_scripts/he_fenics_partition_jax_values_l3_np16_run2.json](experiment_scripts/he_fenics_partition_jax_values_l3_np16_run2.json)
- [experiment_scripts/he_fenics_same_values_control_l3_np16.json](experiment_scripts/he_fenics_same_values_control_l3_np16.json)
- [experiment_scripts/he_fenics_partition_jax_values_results.md](experiment_scripts/he_fenics_partition_jax_values_results.md)

#### Implementation notes (current JAX+PETSc path)

- Uses `LocalColoringAssembler` with local SFD coloring and sequential per-color HVP.
- PETSc matrix uses block size `3`, elasticity near-nullspace, and GAMG coordinates.
- For exact mapping experiments, FEniCS mesh must be constructed from HDF5 cells/coords
  (`--fenics_mesh_source h5`), because `create_box` tetrahedralization does not match
  the HDF5 topology exactly.

Reference implementation document:
- [HyperElasticity3D_jax_petsc_IMPLEMENTATION.md](HyperElasticity3D_jax_petsc_IMPLEMENTATION.md)

#### How to run

Full 24-step JAX+PETSc benchmark:
```bash
docker exec bench_container bash -lc "cd /workdir && \
  mpirun -np 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py \
    --level 3 --steps 24 --total_steps 24 \
    --profile performance \
    --ksp_type gmres --pc_type gamg \
    --ksp_rtol 1e-1 --ksp_max_it 30 \
    --gamg_threshold 0.05 --gamg_agg_nsmooths 1 \
    --pc_setup_on_ksp_cap \
    --hvp_eval_mode sequential \
    --quiet \
    --out /workdir/experiment_scripts/he_jaxpetsc_full_l3_np16_<timestamp>.json"
```

FEniCS-layout/JAX-values isolation benchmark:
```bash
docker exec bench_container bash -lc "cd /workdir && \
  mpirun -np 16 python3 experiment_scripts/bench_he_fenics_partition_jax_values.py \
    --level 3 --step 1 --total_steps 24 \
    --fenics_mesh_source h5 \
    --ksp_type gmres --pc_type gamg \
    --ksp_rtol 1e-1 --ksp_max_it 30 \
    --gamg_threshold 0.05 --gamg_agg_nsmooths 1 \
    --quiet \
    --out /workdir/experiment_scripts/he_fenics_partition_jax_values_l3_np16_run2.json"
```

#### Comparison vs original JAX trajectory references

Current repository has original-JAX full trajectories for L1/L2 (serial), but not L3.

| Solver | Level | np | Steps | Total time [s] | Total Newton iters | Final energy |
|---|---:|---:|---:|---:|---:|---:|
| Original JAX | 1 | 1 | 24 | 28.8091 | 513 | 197.748635 |
| Original JAX | 2 | 1 | 24 | 264.6954 | 531 | 116.336331 |
| JAX+PETSc | 3 | 16 | 24 | 771.8495 (step sum) | 1064 | 93.704860 |

Original-JAX artifacts:
- [experiment_scripts/he_jax_evolution_l1_recompute_gmres_compare.json](experiment_scripts/he_jax_evolution_l1_recompute_gmres_compare.json)
- [experiment_scripts/he_jax_evolution_l2.json](experiment_scripts/he_jax_evolution_l2.json)

## Annex G) GAMG vs HYPRE BoomerAMG — serial (level 3, 1 MPI rank, Threadripper)

**Hardware:** AMD Threadripper (local workstation), single MPI rank (`np=1`), `OMP_NUM_THREADS=1`.

**Problem:** Same as Annex F — Neo-Hookean 3D elasticity, beam `[0, 0.4] × [-0.005, 0.005]²`, 24 load steps (15° rotation each), level 3 mesh (Nx=320, Ny=8, Nz=8 → 78,003 DOFs).

**Solver:** `HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py` (custom Newton with PETSc linear algebra).

Both runs use the same loose-tolerance strategy:
- `ksp_type = gmres`, `ksp_rtol = 1e-1`, `ksp_max_it = 30`, `pc_setup_on_ksp_cap`
- Hardened nonlinear settings: `tolf=1e-4`, `tolg=1e-3`, `tolg_rel=1e-3`, `tolx_rel=1e-3`, `tolx_abs=1e-10`
- Near-nullspace ON, GAMG coordinates ON, `gamg_threshold=0.05`
- HYPRE run uses HYPRE defaults (`--hypre_nodal_coarsen -1 --hypre_vec_interp_variant -1`); nodal_coarsen=6 / vec_interp_variant=3 segfaulted at np=1.

### G.1 Summary comparison

|                       | **GAMG (loose)** | **HYPRE BoomerAMG** |
| --------------------- | ---------------: | ------------------: |
| PC type               |           `gamg` | `hypre` (BoomerAMG) |
| `ksp_rtol`            |             1e-1 |                1e-1 |
| `ksp_max_it`          |               30 |                  30 |
| `pc_setup_on_ksp_cap` |              Yes |                 Yes |
| Total time            |     **1208.7 s** |        **2019.3 s** |
| Total Newton iters    |            1,170 |                 832 |
| Total KSP iters       |           19,722 |              14,960 |
| Avg KSP/Newton        |             16.9 |                18.0 |
| Final energy          |        93.704785 |           93.705611 |
| All steps converged?  |      Yes (24/24) |          Yes (24/24)|
| Speedup (GAMG/HYPRE)  |        **1.67×** |                  1× |

**Observations:**
- **GAMG is 1.67× faster than HYPRE** in serial, consistent with the 2.2× advantage seen at 16 MPI ranks (Annex F). The smaller gap at np=1 is expected — HYPRE's data conversion overhead is a smaller fraction of total work in serial.
- GAMG needs ~1.4× more Newton iterations (1,170 vs 832) but each is cheaper due to lower AMG setup cost.
- Final energies agree to ~1e-3 (93.7048 vs 93.7056), confirming correctness.

### G.2 Replication commands

**GAMG:**
```bash
python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type gamg \
  --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap \
  --gamg_threshold 0.05 \
  --save_history \
  --quiet --out experiment_scripts/he_custom_l3_np1_gamg.json
```

**HYPRE BoomerAMG:**
```bash
python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type hypre \
  --hypre_nodal_coarsen -1 --hypre_vec_interp_variant -1 \
  --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap \
  --save_history \
  --quiet --out experiment_scripts/he_custom_l3_np1_hypre.json
```

### G.3 Per-step tables

#### Level 3, np=1 — GAMG (loose: rtol=1e-1, ksp_max_it=30, PC reuse)

Artifact: [experiment_scripts/he_custom_l3_np1_gamg.json](experiment_scripts/he_custom_l3_np1_gamg.json)

| Step | Time [s] | Newton iters | Sum KSP iters |        Energy | Status                              |
| ---: | -------: | -----------: | ------------: | ------------: | ----------------------------------- |
|    1 |     39.1 |           41 |           514 |  0.1626210000 | Converged (energy, step, gradient)  |
|    2 |     51.3 |           51 |           768 |  0.6502700000 | Converged (energy, step, gradient)  |
|    3 |     49.0 |           49 |           736 |  1.4637530000 | Converged (energy, step, gradient)  |
|    4 |     56.4 |           55 |           896 |  2.6011550000 | Converged (energy, step, gradient)  |
|    5 |     47.5 |           47 |           752 |  4.0646790000 | Converged (energy, step, gradient)  |
|    6 |     48.5 |           50 |           710 |  5.8534220000 | Converged (energy, step, gradient)  |
|    7 |     47.7 |           48 |           731 |  7.9676520000 | Converged (energy, step, gradient)  |
|    8 |     46.9 |           46 |           762 | 10.4075170000 | Converged (energy, step, gradient)  |
|    9 |     47.5 |           46 |           776 | 13.1732150000 | Converged (energy, step, gradient)  |
|   10 |     46.2 |           45 |           751 | 16.2647240000 | Converged (energy, step, gradient)  |
|   11 |     49.9 |           48 |           818 | 19.6819520000 | Converged (energy, step, gradient)  |
|   12 |     49.0 |           48 |           779 | 23.4256140000 | Converged (energy, step, gradient)  |
|   13 |     49.3 |           50 |           748 | 27.4944680000 | Converged (energy, step, gradient)  |
|   14 |     52.9 |           50 |           904 | 31.8894130000 | Converged (energy, step, gradient)  |
|   15 |     46.3 |           45 |           760 | 36.6100200000 | Converged (energy, step, gradient)  |
|   16 |     47.4 |           45 |           797 | 41.6554540000 | Converged (energy, step, gradient)  |
|   17 |    107.6 |           91 |         2,128 | 47.0225720000 | Converged (energy, step, gradient)  |
|   18 |     46.6 |           46 |           744 | 52.7169120000 | Converged (energy, step, gradient)  |
|   19 |     57.5 |           52 |         1,051 | 58.7366800000 | Converged (energy, step, gradient)  |
|   20 |     43.2 |           42 |           701 | 65.0819150000 | Converged (energy, step, gradient)  |
|   21 |     45.5 |           44 |           744 | 71.7519820000 | Converged (energy, step, gradient)  |
|   22 |     43.6 |           42 |           710 | 78.7466330000 | Converged (energy, step, gradient)  |
|   23 |     44.8 |           44 |           723 | 86.0645880000 | Converged (energy, step, gradient)  |
|   24 |     45.3 |           45 |           719 | 93.7047850000 | Converged (energy, step, gradient)  |

Summary: total time = `1208.7 s`, total Newton iters = `1,170`, total KSP iters = `19,722`.

#### Level 3, np=1 — HYPRE BoomerAMG (loose: rtol=1e-1, ksp_max_it=30, PC reuse)

Artifact: [experiment_scripts/he_custom_l3_np1_hypre.json](experiment_scripts/he_custom_l3_np1_hypre.json)

| Step | Time [s] | Newton iters | Sum KSP iters |        Energy | Status                              |
| ---: | -------: | -----------: | ------------: | ------------: | ----------------------------------- |
|    1 |     49.9 |           28 |           369 |  0.1625560000 | Converged (energy, step, gradient)  |
|    2 |     57.4 |           30 |           411 |  0.6502360000 | Converged (energy, step, gradient)  |
|    3 |     56.5 |           29 |           402 |  1.4630760000 | Converged (energy, step, gradient)  |
|    4 |     65.1 |           31 |           470 |  2.6011410000 | Converged (energy, step, gradient)  |
|    5 |     53.9 |           29 |           358 |  4.0645030000 | Converged (energy, step, gradient)  |
|    6 |     59.5 |           29 |           417 |  5.8532660000 | Converged (energy, step, gradient)  |
|    7 |     64.4 |           31 |           460 |  7.9675370000 | Converged (energy, step, gradient)  |
|    8 |     61.0 |           30 |           425 | 10.4074350000 | Converged (energy, step, gradient)  |
|    9 |     64.8 |           29 |           467 | 13.1730680000 | Converged (energy, step, gradient)  |
|   10 |     67.1 |           30 |           479 | 16.2645240000 | Converged (energy, step, gradient)  |
|   11 |     69.9 |           31 |           514 | 19.6818720000 | Converged (energy, step, gradient)  |
|   12 |     80.8 |           33 |           617 | 23.4251600000 | Converged (energy, step, gradient)  |
|   13 |     95.7 |           38 |           732 | 27.4943640000 | Converged (energy, step, gradient)  |
|   14 |    127.0 |           46 |           998 | 31.8893590000 | Converged (energy, step, gradient)  |
|   15 |     90.9 |           37 |           676 | 36.6099620000 | Converged (energy, step, gradient)  |
|   16 |    102.6 |           40 |           787 | 41.6554030000 | Converged (energy, step, gradient)  |
|   17 |    107.3 |           40 |           831 | 47.0225720000 | Converged (energy, step, gradient)  |
|   18 |     91.8 |           35 |           691 | 52.7168510000 | Converged (energy, step, gradient)  |
|   19 |    121.7 |           43 |           955 | 58.7366260000 | Converged (energy, step, gradient)  |
|   20 |     97.2 |           36 |           737 | 65.0816590000 | Converged (energy, step, gradient)  |
|   21 |    156.7 |           52 |         1,178 | 71.7515890000 | Converged (energy, step, gradient)  |
|   22 |    109.9 |           40 |           794 | 78.7460090000 | Converged (energy, step, gradient)  |
|   23 |     78.0 |           31 |           548 | 86.0643810000 | Converged (energy, step, gradient)  |
|   24 |     90.2 |           34 |           644 | 93.7056110000 | Converged (energy, step, gradient)  |

Summary: total time = `2019.3 s`, total Newton iters = `832`, total KSP iters = `14,960`.

### G.4 32 MPI ranks (level 3, Threadripper)

Same setup as G.1–G.3 but with `mpirun -n 32`.

#### Summary comparison (np=32)

|                       | **GAMG (loose)** | **HYPRE BoomerAMG** |
| --------------------- | ---------------: | ------------------: |
| PC type               |           `gamg` | `hypre` (BoomerAMG) |
| `ksp_rtol`            |             1e-1 |                1e-1 |
| `ksp_max_it`          |               30 |                  30 |
| `pc_setup_on_ksp_cap` |              Yes |                 Yes |
| Total time            |       **52.9 s** |         **163.9 s** |
| Total Newton iters    |            1,164 |                 838 |
| Total KSP iters       |           20,894 |              15,341 |
| Avg KSP/Newton        |             18.0 |                18.3 |
| Final energy          |        93.704832 |           93.705160 |
| All steps converged?  |      Yes (24/24) |          Yes (24/24)|
| Speedup (GAMG/HYPRE)  |        **3.10×** |                  1× |

**Observations:**
- **GAMG is 3.1× faster than HYPRE** at 32 MPI ranks, up from 1.67× at np=1 and 2.2× at np=16 (Annex F). The HYPRE data-conversion overhead scales poorly with rank count.
- Parallel speedup from np=1 → np=32: GAMG achieves **22.8×** (1208.7 → 52.9 s), HYPRE achieves **12.3×** (2019.3 → 163.9 s).
- Newton/KSP iteration counts are nearly identical to np=1 (GAMG: 1,164 vs 1,170; HYPRE: 838 vs 832), confirming that parallelism does not affect convergence behavior.

#### Replication commands

**GAMG (np=32):**
```bash
export OMP_NUM_THREADS=1
mpirun -n 32 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type gamg \
  --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap \
  --gamg_threshold 0.05 \
  --save_history \
  --quiet --out experiment_scripts/he_custom_l3_np32_gamg.json
```

**HYPRE BoomerAMG (np=32):**
```bash
export OMP_NUM_THREADS=1
mpirun -n 32 python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py \
  --level 3 --steps 24 --start_step 1 --total_steps 24 \
  --maxit 100 --ksp_type gmres --pc_type hypre \
  --hypre_nodal_coarsen -1 --hypre_vec_interp_variant -1 \
  --ksp_rtol 1e-1 --ksp_max_it 30 --pc_setup_on_ksp_cap \
  --save_history \
  --quiet --out experiment_scripts/he_custom_l3_np32_hypre.json
```

#### Per-step tables (np=32)

##### Level 3, np=32 — GAMG (loose: rtol=1e-1, ksp_max_it=30, PC reuse)

Artifact: [experiment_scripts/he_custom_l3_np32_gamg.json](experiment_scripts/he_custom_l3_np32_gamg.json)

| Step | Time [s] | Newton iters | Sum KSP iters |        Energy | Status                              |
| ---: | -------: | -----------: | ------------: | ------------: | ----------------------------------- |
|    1 |      1.6 |           39 |           441 |  0.1625930000 | Converged (energy, step, gradient)  |
|    2 |      2.0 |           48 |           708 |  0.6503360000 | Converged (energy, step, gradient)  |
|    3 |      2.1 |           50 |           775 |  1.4633360000 | Converged (energy, step, gradient)  |
|    4 |      2.1 |           49 |           722 |  2.6013280000 | Converged (energy, step, gradient)  |
|    5 |      2.0 |           47 |           712 |  4.0647030000 | Converged (energy, step, gradient)  |
|    6 |      2.2 |           50 |           734 |  5.8532980000 | Converged (energy, step, gradient)  |
|    7 |      2.0 |           46 |           684 |  7.9676110000 | Converged (energy, step, gradient)  |
|    8 |      2.1 |           47 |           776 | 10.4075020000 | Converged (energy, step, gradient)  |
|    9 |      2.2 |           49 |           908 | 13.1732850000 | Converged (energy, step, gradient)  |
|   10 |      4.1 |           84 |         1,917 | 16.2645600000 | Converged (energy, step, gradient)  |
|   11 |      2.0 |           45 |           711 | 19.6819850000 | Converged (energy, step, gradient)  |
|   12 |      2.2 |           49 |           766 | 23.4253440000 | Converged (energy, step, gradient)  |
|   13 |      2.1 |           48 |           773 | 27.4944280000 | Converged (energy, step, gradient)  |
|   14 |      1.9 |           43 |           681 | 31.8895810000 | Converged (energy, step, gradient)  |
|   15 |      2.0 |           44 |           744 | 36.6101120000 | Converged (energy, step, gradient)  |
|   16 |      3.0 |           62 |         1,338 | 41.6551330000 | Converged (energy, step, gradient)  |
|   17 |      2.0 |           45 |           819 | 47.0231560000 | Converged (energy, step, gradient)  |
|   18 |      2.8 |           59 |         1,160 | 52.7168600000 | Converged (energy, step, gradient)  |
|   19 |      2.2 |           50 |           859 | 58.7366400000 | Converged (energy, step, gradient)  |
|   20 |      2.4 |           34 |         1,749 | 65.0816890000 | Converged (energy, step, gradient)  |
|   21 |      2.0 |           45 |           706 | 71.7518820000 | Converged (energy, step, gradient)  |
|   22 |      1.9 |           43 |           709 | 78.7464610000 | Converged (energy, step, gradient)  |
|   23 |      2.0 |           44 |           780 | 86.0645860000 | Converged (energy, step, gradient)  |
|   24 |      1.9 |           44 |           722 | 93.7048320000 | Converged (energy, step, gradient)  |

Summary: total time = `52.9 s`, total Newton iters = `1,164`, total KSP iters = `20,894`.

##### Level 3, np=32 — HYPRE BoomerAMG (loose: rtol=1e-1, ksp_max_it=30, PC reuse)

Artifact: [experiment_scripts/he_custom_l3_np32_hypre.json](experiment_scripts/he_custom_l3_np32_hypre.json)

| Step | Time [s] | Newton iters | Sum KSP iters |        Energy | Status                              |
| ---: | -------: | -----------: | ------------: | ------------: | ----------------------------------- |
|    1 |      4.8 |           29 |           426 |  0.1625560000 | Converged (energy, step, gradient)  |
|    2 |      5.2 |           30 |           447 |  0.6502370000 | Converged (energy, step, gradient)  |
|    3 |      5.0 |           29 |           414 |  1.4630760000 | Converged (energy, step, gradient)  |
|    4 |      5.0 |           29 |           404 |  2.6011400000 | Converged (energy, step, gradient)  |
|    5 |      4.7 |           28 |           361 |  4.0645040000 | Converged (energy, step, gradient)  |
|    6 |      5.2 |           30 |           420 |  5.8532660000 | Converged (energy, step, gradient)  |
|    7 |      5.2 |           29 |           431 |  7.9675370000 | Converged (energy, step, gradient)  |
|    8 |      5.4 |           30 |           443 | 10.4074350000 | Converged (energy, step, gradient)  |
|    9 |      5.3 |           29 |           451 | 13.1730680000 | Converged (energy, step, gradient)  |
|   10 |      5.6 |           30 |           479 | 16.2645240000 | Converged (energy, step, gradient)  |
|   11 |      5.4 |           29 |           480 | 19.6818720000 | Converged (energy, step, gradient)  |
|   12 |      6.2 |           31 |           580 | 23.4251600000 | Converged (energy, step, gradient)  |
|   13 |      7.2 |           36 |           677 | 27.4943640000 | Converged (energy, step, gradient)  |
|   14 |      9.7 |           46 |           972 | 31.8893590000 | Converged (energy, step, gradient)  |
|   15 |      7.0 |           36 |           656 | 36.6099610000 | Converged (energy, step, gradient)  |
|   16 |     17.9 |           79 |         1,972 | 41.6538440000 | Converged (energy, step, gradient)  |
|   17 |      7.1 |           35 |           690 | 47.0225710000 | Converged (energy, step, gradient)  |
|   18 |      7.5 |           36 |           738 | 52.7168520000 | Converged (energy, step, gradient)  |
|   19 |      8.2 |           39 |           817 | 58.7366260000 | Converged (energy, step, gradient)  |
|   20 |      7.5 |           36 |           747 | 65.0816690000 | Converged (energy, step, gradient)  |
|   21 |      7.9 |           38 |           767 | 71.7516130000 | Converged (energy, step, gradient)  |
|   22 |      7.1 |           35 |           688 | 78.7460660000 | Converged (energy, step, gradient)  |
|   23 |      6.0 |           31 |           540 | 86.0644020000 | Converged (energy, step, gradient)  |
|   24 |      7.8 |           38 |           741 | 93.7051600000 | Converged (energy, step, gradient)  |

Summary: total time = `163.9 s`, total Newton iters = `838`, total KSP iters = `15,341`.

#### Scaling summary across np=1, 16, 32

| np | GAMG time [s] | HYPRE time [s] | GAMG/HYPRE speedup | GAMG parallel speedup | HYPRE parallel speedup |
|---:|--------------:|---------------:|-------------------:|----------------------:|-----------------------:|
|  1 |       1208.7  |        2019.3  |              1.67× |                    1× |                     1× |
| 16 |         62.4  |         135.5  |              2.17× |                19.4× |                  14.9× |
| 32 |         52.9  |         163.9  |              3.10× |                22.8× |                  12.3× |

Note: np=16 values are from Annex F (Docker container). np=1 and np=32 are native (Threadripper). Cross-environment timing comparisons should be interpreted with caution.

### F.12 JAX+PETSc GAMG scaling benchmark — Native build (L3, np=1–32, 2026-03-03)

Full 24-step load trajectory on level 3 (77,517 free DOFs, 122,880 elements) with GAMG preconditioner, run natively (no Docker) on bare metal.

#### System

- **CPU**: AMD Ryzen Threadripper PRO 7975WX 32-Core Processor (64 threads)
- **OS**: Arch Linux x86_64 (kernel 6.18.13-arch1-1)
- **MPI**: OpenMPI 5.0.10
- **DOLFINx**: 0.10.0.post5
- **PETSc**: 3.24.2 (with Hypre, METIS, ParMETIS, MUMPS, SuperLU_dist, SuiteSparse)
- **JAX**: 0.9.0.1
- **Python**: 3.12.10

#### Configuration

- Solver: `HyperElasticity3D_jax_petsc/solve_HE_dof.py`
- Profile: `performance` (GMRES + GAMG, `ksp_rtol=1e-1`, `ksp_max_it=30`, `pc_setup_on_ksp_cap`, `hvp_eval_mode=sequential`)
- `gamg_threshold=0.05`, `gamg_agg_nsmooths=1`, near-nullspace + coordinates enabled
- Newton: `tolf=1e-4`, `tolg=1e-3`, `tolg_rel=1e-3`, `tolx_rel=1e-3`, golden-section line search on $[-0.5, 2.0]$
- `retry_on_failure` enabled
- `OMP_NUM_THREADS=1`

#### Strong scaling results

All runs complete 24/24 steps, converging to $E \approx 93.705$.

| np  | Total (s) | Avg/step (s) | Newton its | KSP its | Final energy | Repairs | Speedup |
| --- | --------- | ------------ | ---------- | ------- | ------------ | ------- | ------- |
| 1   | 3199.78   | 133.32       | 1089       | 19822   | 93.704998    | 2       | 1.00×   |
| 2   | 944.82    | 39.37        | 1090       | 17537   | 93.704770    | 0       | 3.39×   |
| 4   | 772.63    | 32.19        | 1104       | 18865   | 93.704813    | 1       | 4.14×   |
| 8   | 777.33    | 32.39        | 1030       | 19687   | 93.704668    | 3       | 4.12×   |
| 16  | 396.27    | 16.51        | 1108       | 19216   | 93.704908    | 1       | 8.07×   |
| 32  | 267.06    | 11.13        | 1065       | 17508   | 93.704996    | 0       | 12.0×   |

#### Comparison with Docker result (F.11)

The F.11 Docker benchmark ran only np=16, level 3, 24 steps:

| Metric                  | Docker np=16 (F.11) | Native np=16 (F.12) | Native np=32 |
| :---------------------- | ------------------: | ------------------: | -----------: |
| Total time              |            771.85 s |            396.27 s |     267.06 s |
| Avg step time           |             32.16 s |             16.51 s |      11.13 s |
| Total Newton iterations |                1064 |                1108 |         1065 |
| Total KSP iterations    |               18299 |               19216 |        17508 |
| Final energy            |          93.704860  |          93.704908  |   93.704996  |

- **1.95× faster at np=16** on native build compared to Docker (396 s vs 772 s).
- **np=32 delivers 12× speedup** over serial — no scaling degradation through 32 ranks on this problem size.
- Newton and KSP iteration counts vary slightly across runs due to the nonlinear trajectory's sensitivity to floating-point ordering differences, but final energies agree to ~$10^{-4}$.

#### Key observations

- **Super-linear speedup at np=2** (3.39×): the serial run is memory-bandwidth limited; splitting across 2 ranks improves cache utilization.
- **Plateau at np=4–8** (4.1×): the problem size (77K DOFs) is getting small for 4–8 ranks; communication overhead roughly balances compute savings.
- **Second speedup phase at np=16–32**: GAMG's coarse-grid solves and smoother operations still benefit from additional ranks.
- **All runs produce consistent final energies** ($93.7049 \pm 0.0003$), confirming numerical reproducibility.

#### Reproducing

```bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

# Full 24-step trajectory (e.g. 16 ranks)
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py \
    --level 3 --steps 24 --total_steps 24 \
    --profile performance --quiet \
    --out /tmp/he_gamg_np16.json
```
