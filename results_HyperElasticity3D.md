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
