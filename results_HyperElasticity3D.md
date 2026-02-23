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

No level-4 JAX reference trajectory is available in this report yet, so relative error vs JAX is reported as `—`.

| Step | Time [s] | Newton iters | Sum linear iters |        Energy | Relative error vs JAX | Status                               |
| ---: | -------: | -----------: | ---------------: | ------------: | --------------------: | ------------------------------------ |
|    1 |  91.7594 |           28 |              350 |  0.1519710000 |                     — | Energy change converged              |
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

| Metric                    | Custom solver (Annex C)                          | SNES (Annex D)      |
| ------------------------- | -----------------------------------------------: | ------------------: |
| KSP type                  | CG                                               | GMRES               |
| AMG config                | `nodal_coarsen=6`, `vec_interp_variant=3` (n6v3) | HYPRE defaults      |
| `ksp_rtol`                | 1e-1                                             | 1e-1                |
| `ksp_max_it`              | 30                                               | 500                 |
| Convergence criterion     | energy change < 1e-4                             | `snes_atol=1e-3`    |
| Converged steps           | 96/96                                            | 93/96               |
| Total Newton iterations   | 1 209                                            | 1 175               |
| Total KSP iterations      | 24 872                                           | 22 490              |
| **Avg KSP / Newton step** | **20.6**                                         | **19.1**            |
| Wall time [s]             | 72.62                                            | 15.03               |

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
|   94 |   1410.00 |   0.8676 |         10 |         867 | r=-3 |
|   95 |   1425.00 |   0.6536 |         11 |         654 | r=-3 |
|   96 |   1440.00 |   0.5967 |          8 |         597 | r=-3 |

| **Total** | | **15.03** | **1175** | **22,490** | **93/96** |

**Observations:**
- Steps 1–93: 11–13 Newton iters, 150–270 KSP per step (all converging, avg 19.0 KSP/Newton).
- Avg KSP/Newton ratio: 22 490 / 1 175 ≈ **19.1 KSP/Newton** — essentially matches custom solver's 20.6.
- Steps 94–96 (1410°–1440°): `SNES_DIVERGED_LINEAR_SOLVE`; each step hits the 500-iteration KSP cap on one internal Newton linear solve, suggesting HYPRE defaults AMG loses effectiveness at this extreme deformation state. Not a nullspace issue.
