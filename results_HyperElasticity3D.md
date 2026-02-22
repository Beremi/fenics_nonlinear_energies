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
  - `maxit = 100`
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

Final artifacts used in tables:
- Level 1: [experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap.json](experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap.json)
- Level 2: [experiment_scripts/he_fenics_custom_evolution_l2_skip_ksp30_pc_cap.json](experiment_scripts/he_fenics_custom_evolution_l2_skip_ksp30_pc_cap.json)

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

These runs use the same custom setup as the final serial configuration except `--no_near_nullspace` was required for MPI runs because current near-nullspace construction crashes in parallel (PETSc SEGV during nullspace build).

Common settings for all tables below: `ksp_type=gmres`, `pc_type=hypre`, `ksp_rtol=1e-1`, `ksp_max_it=30`, skip explicit `nodal/vec`, `--pc_setup_on_ksp_cap`, `--no_near_nullspace`.

### Level 1 — Custom FEniCS MPI nproc=4

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1062 | 21 | 304 | 0.3464120000 | 1.74e-06 | Energy change converged |
| 2 | 0.1106 | 21 | 252 | 1.3856370000 | 1.60e-06 | Energy change converged |
| 3 | 0.1111 | 20 | 261 | 3.1173460000 | 2.27e-06 | Energy change converged |
| 4 | 0.1283 | 23 | 344 | 5.5401490000 | 2.44e-07 | Energy change converged |
| 5 | 0.1078 | 20 | 277 | 8.6505010000 | 2.53e-07 | Energy change converged |
| 6 | 0.1228 | 22 | 322 | 12.4422680000 | 1.76e-07 | Energy change converged |
| 7 | 0.1108 | 21 | 275 | 16.9115650000 | 7.20e-08 | Energy change converged |
| 8 | 0.1248 | 21 | 309 | 22.0617420000 | 1.39e-07 | Energy change converged |
| 9 | 0.1340 | 22 | 280 | 27.8989760000 | 9.75e-09 | Energy change converged |
| 10 | 0.1356 | 23 | 344 | 34.4265020000 | 9.63e-08 | Energy change converged |
| 11 | 0.1271 | 23 | 316 | 41.6441060000 | 9.32e-08 | Energy change converged |
| 12 | 0.1184 | 21 | 277 | 49.5500720000 | 5.37e-08 | Energy change converged |
| 13 | 0.1132 | 20 | 250 | 58.1426610000 | 1.18e-10 | Energy change converged |
| 14 | 0.1174 | 22 | 280 | 67.4207960000 | 7.73e-08 | Energy change converged |
| 15 | 0.1281 | 21 | 297 | 77.3830590000 | 1.43e-09 | Energy change converged |
| 16 | 0.1216 | 21 | 291 | 88.0180590000 | 1.59e-08 | Energy change converged |
| 17 | 0.1446 | 24 | 365 | 99.3269870000 | 4.98e-09 | Energy change converged |
| 18 | 0.1444 | 23 | 346 | 111.3262050000 | 1.74e-08 | Energy change converged |
| 19 | 0.1484 | 22 | 340 | 124.0155390000 | 2.76e-09 | Energy change converged |
| 20 | 0.1197 | 20 | 264 | 137.3923340000 | 6.09e-09 | Energy change converged |
| 21 | 0.1479 | 24 | 355 | 151.4552330000 | 8.31e-09 | Energy change converged |
| 22 | 0.1226 | 20 | 267 | 166.2042260000 | 3.18e-08 | Energy change converged |
| 23 | 0.1559 | 23 | 334 | 181.6387080000 | 1.20e-08 | Energy change converged |
| 24 | 0.1255 | 22 | 319 | 197.7551340000 | 3.29e-05 | Energy change converged |

Summary: total time = `3.0268 s`, total Newton iters = `520`, total linear iters = `7269`, max relative error = `3.29e-05`, mean relative error = `1.66e-06`.

### Level 1 — Custom FEniCS MPI nproc=8

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0916 | 21 | 314 | 0.3464120000 | 1.74e-06 | Energy change converged |
| 2 | 0.0941 | 22 | 299 | 1.3856370000 | 1.60e-06 | Energy change converged |
| 3 | 0.1213 | 21 | 296 | 3.1173390000 | 2.71e-08 | Energy change converged |
| 4 | 0.1384 | 23 | 341 | 5.5401500000 | 4.25e-07 | Energy change converged |
| 5 | 0.1328 | 23 | 371 | 8.6504990000 | 2.20e-08 | Energy change converged |
| 6 | 0.1313 | 23 | 361 | 12.4422660000 | 1.53e-08 | Energy change converged |
| 7 | 0.1282 | 23 | 346 | 16.9115640000 | 1.29e-08 | Energy change converged |
| 8 | 0.1086 | 21 | 288 | 22.0617410000 | 9.33e-08 | Energy change converged |
| 9 | 0.1229 | 23 | 303 | 27.8989760000 | 9.75e-09 | Energy change converged |
| 10 | 0.1147 | 21 | 293 | 34.4264990000 | 9.19e-09 | Energy change converged |
| 11 | 0.1095 | 23 | 310 | 41.6441020000 | 2.87e-09 | Energy change converged |
| 12 | 0.1192 | 22 | 303 | 49.5500710000 | 3.35e-08 | Energy change converged |
| 13 | 0.1007 | 20 | 246 | 58.1426610000 | 1.18e-10 | Energy change converged |
| 14 | 0.1179 | 23 | 314 | 67.4207910000 | 3.11e-09 | Energy change converged |
| 15 | 0.1103 | 21 | 290 | 77.3830610000 | 2.44e-08 | Energy change converged |
| 16 | 0.1252 | 23 | 345 | 88.0180580000 | 4.57e-09 | Energy change converged |
| 17 | 0.1384 | 24 | 364 | 99.3269870000 | 4.98e-09 | Energy change converged |
| 18 | 0.1360 | 23 | 352 | 111.3262030000 | 5.83e-10 | Energy change converged |
| 19 | 0.1500 | 23 | 339 | 124.0155390000 | 2.76e-09 | Energy change converged |
| 20 | 0.1181 | 22 | 278 | 137.3923340000 | 6.09e-09 | Energy change converged |
| 21 | 0.1303 | 23 | 320 | 151.4552330000 | 8.31e-09 | Energy change converged |
| 22 | 0.1182 | 22 | 296 | 166.2042280000 | 4.38e-08 | Energy change converged |
| 23 | 0.1494 | 23 | 330 | 181.6387120000 | 1.00e-08 | Energy change converged |
| 24 | 0.1030 | 20 | 283 | 197.7551220000 | 3.28e-05 | Energy change converged |

Summary: total time = `2.9101 s`, total Newton iters = `533`, total linear iters = `7582`, max relative error = `3.28e-05`, mean relative error = `1.54e-06`.

### Level 1 — Custom FEniCS MPI nproc=16

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1391 | 21 | 263 | 0.3464110000 | 1.14e-06 | Energy change converged |
| 2 | 0.1585 | 22 | 320 | 1.3856350000 | 1.59e-07 | Energy change converged |
| 3 | 0.1268 | 20 | 241 | 3.1173510000 | 3.88e-06 | Energy change converged |
| 4 | 0.1533 | 23 | 355 | 5.5401480000 | 6.38e-08 | Energy change converged |
| 5 | 0.1324 | 21 | 309 | 8.6505080000 | 1.06e-06 | Energy change converged |
| 6 | 0.1308 | 21 | 299 | 12.4422700000 | 3.37e-07 | Energy change converged |
| 7 | 0.1220 | 20 | 265 | 16.9115640000 | 1.29e-08 | Energy change converged |
| 8 | 0.1270 | 21 | 265 | 22.0617430000 | 1.84e-07 | Energy change converged |
| 9 | 0.1384 | 23 | 297 | 27.8989780000 | 8.14e-08 | Energy change converged |
| 10 | 0.1227 | 20 | 259 | 34.4265080000 | 2.71e-07 | Energy change converged |
| 11 | 0.1443 | 23 | 309 | 41.6441100000 | 1.89e-07 | Energy change converged |
| 12 | 0.1385 | 22 | 312 | 49.5500700000 | 1.33e-08 | Energy change converged |
| 13 | 0.1400 | 21 | 292 | 58.1426660000 | 8.61e-08 | Energy change converged |
| 14 | 0.1622 | 24 | 377 | 67.4207910000 | 3.11e-09 | Energy change converged |
| 15 | 0.1229 | 20 | 255 | 77.3830630000 | 5.03e-08 | Energy change converged |
| 16 | 0.1463 | 23 | 327 | 88.0180580000 | 4.57e-09 | Energy change converged |
| 17 | 0.1532 | 24 | 368 | 99.3269870000 | 4.98e-09 | Energy change converged |
| 18 | 0.1528 | 24 | 344 | 111.3262030000 | 5.83e-10 | Energy change converged |
| 19 | 0.1620 | 23 | 339 | 124.0155410000 | 1.89e-08 | Energy change converged |
| 20 | 0.1422 | 23 | 328 | 137.3923330000 | 1.19e-09 | Energy change converged |
| 21 | 0.1402 | 22 | 298 | 151.4552350000 | 2.15e-08 | Energy change converged |
| 22 | 0.1286 | 20 | 302 | 166.2042240000 | 1.97e-08 | Energy change converged |
| 23 | 0.1753 | 23 | 320 | 181.6387080000 | 1.20e-08 | Energy change converged |
| 24 | 0.1424 | 23 | 353 | 197.7551650000 | 3.30e-05 | Energy change converged |

Summary: total time = `3.4019 s`, total Newton iters = `527`, total linear iters = `7397`, max relative error = `3.30e-05`, mean relative error = `1.69e-06`.

### Level 2 — Custom FEniCS MPI nproc=4

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.2933 | 9 | 94 | 0.5824270000 | 1.87e+00 | Energy change converged |
| 2 | 1.2993 | 34 | 433 | 0.8108400000 | 8.74e-06 | Energy change converged |
| 3 | 1.1110 | 27 | 421 | 1.8243680000 | 5.90e-07 | Energy change converged |
| 4 | 1.0552 | 27 | 353 | 3.2432380000 | 2.04e-07 | Energy change converged |
| 5 | 1.1642 | 28 | 451 | 5.0672630000 | 1.19e-06 | Energy change converged |
| 6 | 0.9669 | 25 | 328 | 7.2960200000 | 2.69e-07 | Energy change converged |
| 7 | 1.1997 | 29 | 441 | 9.9289580000 | 5.26e-07 | Energy change converged |
| 8 | 1.0861 | 26 | 403 | 12.9658030000 | 1.48e-07 | Energy change converged |
| 9 | 1.0976 | 27 | 379 | 16.4069420000 | 2.96e-08 | Energy change converged |
| 10 | 1.1225 | 26 | 402 | 20.2529480000 | 5.16e-08 | Energy change converged |
| 11 | 1.0395 | 25 | 363 | 24.5041650000 | 1.71e-07 | Energy change converged |
| 12 | 1.2112 | 28 | 446 | 29.1606590000 | 1.85e-08 | Energy change converged |
| 13 | 1.0333 | 24 | 375 | 34.2223430000 | 3.96e-08 | Energy change converged |
| 14 | 1.0119 | 23 | 355 | 39.6889330000 | 1.14e-07 | Energy change converged |
| 15 | 1.1260 | 26 | 388 | 45.5598050000 | 5.00e-07 | Energy change converged |
| 16 | 1.6332 | 34 | 655 | 51.8204000000 | 5.36e-07 | Energy change converged |
| 17 | 1.0097 | 23 | 389 | 58.4802480000 | 1.37e-09 | Energy change converged |
| 18 | 1.0959 | 24 | 421 | 65.5436910000 | 2.57e-08 | Energy change converged |
| 19 | 1.0405 | 23 | 387 | 73.0099330000 | 2.15e-07 | Energy change converged |
| 20 | 1.1080 | 24 | 400 | 80.8777560000 | 1.33e-07 | Energy change converged |
| 21 | 1.1772 | 25 | 444 | 89.1458850000 | 3.28e-07 | Energy change converged |
| 22 | 1.1122 | 25 | 378 | 97.8129100000 | 1.50e-06 | Energy change converged |
| 23 | 1.1977 | 26 | 441 | 106.8769220000 | 4.47e-07 | Energy change converged |
| 24 | 1.3447 | 28 | 501 | 116.3323410000 | 3.43e-05 | Energy change converged |

Summary: total time = `26.5368 s`, total Newton iters = `616`, total linear iters = `9648`, max relative error = `1.87e+00`, mean relative error = `7.81e-02`.

### Level 2 — Custom FEniCS MPI nproc=8

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.6221 | 26 | 370 | 0.2027080000 | 7.11e-07 | Energy change converged |
| 2 | 0.6477 | 26 | 379 | 0.8108350000 | 2.57e-06 | Energy change converged |
| 3 | 0.6702 | 25 | 359 | 1.8243800000 | 5.99e-06 | Energy change converged |
| 4 | 0.6576 | 27 | 365 | 3.2432380000 | 2.04e-07 | Energy change converged |
| 5 | 0.6552 | 26 | 370 | 5.0672540000 | 5.90e-07 | Energy change converged |
| 6 | 0.6435 | 26 | 372 | 7.2960160000 | 2.79e-07 | Energy change converged |
| 7 | 0.6661 | 26 | 370 | 9.9289540000 | 1.23e-07 | Energy change converged |
| 8 | 0.6414 | 25 | 358 | 12.9658110000 | 7.65e-07 | Energy change converged |
| 9 | 0.6363 | 25 | 352 | 16.4069430000 | 9.06e-08 | Energy change converged |
| 10 | 0.6045 | 24 | 346 | 20.2529470000 | 2.25e-09 | Energy change converged |
| 11 | 0.6415 | 25 | 373 | 24.5041620000 | 4.89e-08 | Energy change converged |
| 12 | 0.6863 | 26 | 373 | 29.1606660000 | 2.22e-07 | Energy change converged |
| 13 | 0.6695 | 26 | 403 | 34.2223460000 | 1.27e-07 | Energy change converged |
| 14 | 0.6388 | 24 | 365 | 39.6889280000 | 1.24e-08 | Energy change converged |
| 15 | 0.6660 | 25 | 390 | 45.5597980000 | 3.46e-07 | Energy change converged |
| 16 | 1.0414 | 35 | 677 | 51.8203820000 | 1.88e-07 | Energy change converged |
| 17 | 0.6556 | 24 | 383 | 58.4802550000 | 1.18e-07 | Energy change converged |
| 18 | 0.6760 | 24 | 415 | 65.5436900000 | 1.05e-08 | Energy change converged |
| 19 | 0.6639 | 24 | 419 | 73.0099180000 | 9.09e-09 | Energy change converged |
| 20 | 0.6997 | 24 | 409 | 80.8777730000 | 3.43e-07 | Energy change converged |
| 21 | 0.6878 | 25 | 418 | 89.1458820000 | 2.94e-07 | Energy change converged |
| 22 | 0.7401 | 27 | 423 | 97.8128840000 | 1.23e-06 | Energy change converged |
| 23 | 0.7043 | 25 | 385 | 106.8769100000 | 5.59e-07 | Energy change converged |
| 24 | 1.3826 | 45 | 934 | 116.3238510000 | 1.07e-04 | Energy change converged |

Summary: total time = `16.9981 s`, total Newton iters = `635`, total linear iters = `10008`, max relative error = `1.07e-04`, mean relative error = `5.09e-06`.

### Level 2 — Custom FEniCS MPI nproc=16

| Step | Time [s] | Newton iters | Sum linear iters | Energy | Relative error vs JAX | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.4746 | 24 | 316 | 0.2027200000 | 5.99e-05 | Energy change converged |
| 2 | 0.5919 | 28 | 438 | 0.8108430000 | 1.24e-05 | Energy change converged |
| 3 | 0.5438 | 26 | 383 | 1.8243680000 | 5.90e-07 | Energy change converged |
| 4 | 0.6211 | 28 | 444 | 3.2432460000 | 2.67e-06 | Energy change converged |
| 5 | 0.5553 | 27 | 374 | 5.0672590000 | 3.97e-07 | Energy change converged |
| 6 | 0.6025 | 28 | 387 | 7.2960170000 | 1.42e-07 | Energy change converged |
| 7 | 0.5686 | 26 | 384 | 9.9289570000 | 4.25e-07 | Energy change converged |
| 8 | 0.5674 | 26 | 392 | 12.9658030000 | 1.48e-07 | Energy change converged |
| 9 | 0.5334 | 25 | 374 | 16.4069480000 | 3.95e-07 | Energy change converged |
| 10 | 0.5557 | 26 | 390 | 20.2529480000 | 5.16e-08 | Energy change converged |
| 11 | 0.5617 | 27 | 403 | 24.5041610000 | 8.07e-09 | Energy change converged |
| 12 | 0.5267 | 25 | 391 | 29.1606590000 | 1.85e-08 | Energy change converged |
| 13 | 0.5142 | 25 | 382 | 34.2223470000 | 1.56e-07 | Energy change converged |
| 14 | 0.5792 | 26 | 410 | 39.6889260000 | 6.28e-08 | Energy change converged |
| 15 | 0.5719 | 26 | 409 | 45.5597920000 | 2.15e-07 | Energy change converged |
| 16 | 0.9602 | 39 | 757 | 51.8203790000 | 1.30e-07 | Energy change converged |
| 17 | 0.5257 | 24 | 383 | 58.4802490000 | 1.57e-08 | Energy change converged |
| 18 | 0.5258 | 24 | 410 | 65.5436900000 | 1.05e-08 | Energy change converged |
| 19 | 0.6479 | 25 | 445 | 73.0099120000 | 7.31e-08 | Energy change converged |
| 20 | 0.5806 | 25 | 444 | 80.8777530000 | 9.56e-08 | Energy change converged |
| 21 | 0.6375 | 26 | 504 | 89.1458740000 | 2.04e-07 | Energy change converged |
| 22 | 0.6581 | 28 | 509 | 97.8128410000 | 7.95e-07 | Energy change converged |
| 23 | 0.5592 | 26 | 425 | 106.8769000000 | 6.52e-07 | Energy change converged |
| 24 | 1.1500 | 47 | 1008 | 116.3238720000 | 1.07e-04 | Energy change converged |

Summary: total time = `14.6130 s`, total Newton iters = `657`, total linear iters = `10762`, max relative error = `1.07e-04`, mean relative error = `7.78e-06`.

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
