# HyperElasticity3D Results (JAX vs FEniCS)

## Scope of this report

This report starts with the requested baseline comparison:
1. JAX reference (level 1 and level 2, single time step)
2. FEniCS custom Newton (level 1 and level 2, single time step)
3. Current SNES progress and continuation plan

All runs use one rotation step with angle
\(\theta = 4\cdot 2\pi / 24 \approx 1.0472\) rad.

---

## 1) JAX reference (single step)

Method: JAX Newton + golden-section line search + AMG (elastic near-nullspace), consistent with `example_HyperElasticity3D_jax.ipynb`.

| Mesh level |              Energy | Newton iters | Time [s] | Termination                           |
| ---------- | ------------------: | -----------: | -------: | ------------------------------------- |
| 1          |  0.3464113961964319 |           18 |   1.5005 | Stopping condition for f is satisfied |
| 2          | 0.20270785577653685 |           23 |   7.3096 | Stopping condition for f is satisfied |

Command used:
- `source /tmp/jaxenv/bin/activate.fish`
- `python3 tmp_work/run_he_jax_levels12.py`

---

## 2) FEniCS custom Newton (single step)

Method: `tools_petsc4py/minimizers.py` custom Newton, DOLFINx assembly, HYPRE+CG with 3D elastic near-nullspace.

| Mesh level |   Energy | Newton iters | Time [s] | Termination             |
| ---------- | -------: | -----------: | -------: | ----------------------- |
| 1          | 0.346411 |           18 |   2.6332 | Energy change converged |
| 2          | 0.202709 |           21 |  44.0183 | Energy change converged |

Command used:
- `docker run --rm --entrypoint "" -v "$PWD":/work -w /work fenics_test python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level <1|2> --steps 1 --quiet`

### Match vs JAX

- Level 1: excellent match (`0.346411` vs `0.346411396...`), same iteration count (18).
- Level 2: excellent match (`0.202709` vs `0.202707856...`), iteration count close (21 vs 23).

---

## 3) Time-evolution check (levels 1 and 2, 24 steps) — JAX vs custom FEniCS

Before moving to SNES level-2 work, both reference paths were run across the full rotation evolution.

Outputs:
- JAX: [experiment_scripts/he_jax_evolution_l1.json](experiment_scripts/he_jax_evolution_l1.json)
- FEniCS custom: [experiment_scripts/he_fenics_custom_evolution_l1.json](experiment_scripts/he_fenics_custom_evolution_l1.json)
- JAX (level 2): [experiment_scripts/he_jax_evolution_l2.json](experiment_scripts/he_jax_evolution_l2.json)
- FEniCS custom (level 2): [experiment_scripts/he_fenics_custom_evolution_l2.json](experiment_scripts/he_fenics_custom_evolution_l2.json)

Commands used:
- `source /tmp/jaxenv/bin/activate.fish && PYTHONPATH=. python3 experiment_scripts/run_he_jax_evolution.py --level 1 --steps 24 --out experiment_scripts/he_jax_evolution_l1.json`
- `docker run --rm --entrypoint "" -v "$PWD":/work -w /work fenics_test python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --out experiment_scripts/he_fenics_custom_evolution_l1.json --quiet`
- `source /tmp/jaxenv/bin/activate.fish && PYTHONPATH=. python3 experiment_scripts/run_he_jax_evolution.py --level 2 --steps 24 --out experiment_scripts/he_jax_evolution_l2.json`
- `docker run --rm --entrypoint "" -v "$PWD":/work -w /work fenics_test python3 HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 2 --steps 24 --out experiment_scripts/he_fenics_custom_evolution_l2.json --quiet`

### Level 1 trajectory summary

- JAX converges for all 24 steps (finite energies, standard Newton termination).
- FEniCS custom matches JAX through step 19, then fails from step 20 onward (`iters=100`, `energy=NaN`).
- For steps 1–19, agreement is strong: max relative energy error is ~`1.14e-6`.

### Level 1 selected step comparison

| Step |     JAX energy | FEniCS custom energy | Relative error | JAX iters | FEniCS iters |
| ---: | -------------: | -------------------: | -------------: | --------: | -----------: |
|    1 |   0.3464113962 |             0.346411 |        1.14e-6 |        18 |           18 |
|    6 |  12.4422658099 |            12.442266 |        1.53e-8 |        20 |           18 |
|   12 |  49.5500693409 |            49.550069 |        6.88e-9 |        21 |           19 |
|   18 | 111.3262030649 |           111.326203 |       5.83e-10 |        21 |           19 |
|   19 | 124.0155386580 |           124.015539 |        2.76e-9 |        22 |           18 |
|   20 | 137.3923331636 |                  NaN |              — |        22 |          100 |
|   24 | 197.7486351731 |                  NaN |              — |        34 |          100 |

### Level 2 trajectory summary

- JAX converges for all 24 steps.
- FEniCS custom matches JAX through step 9, then fails from step 10 onward (`iters=100`, `energy=NaN`).
- For steps 1–9, agreement is strong: max relative energy error is ~`5.64e-6`.

### Level 2 selected step comparison

| Step |     JAX energy | FEniCS custom energy | Relative error | JAX iters | FEniCS iters |
| ---: | -------------: | -------------------: | -------------: | --------: | -----------: |
|    1 |   0.2027078558 |             0.202709 |        5.64e-6 |        23 |           21 |
|    3 |   1.8243690766 |             1.824366 |        1.69e-6 |        19 |           21 |
|    6 |   7.2960180354 |             7.296016 |        2.79e-7 |        20 |           21 |
|    9 |  16.4069415137 |            16.406942 |        2.96e-8 |        21 |           22 |
|   10 |  20.2529469544 |                  NaN |              — |        21 |          100 |
|   24 | 116.3363305574 |                  NaN |              — |        26 |          100 |

### Interpretation

- The custom FEniCS implementation is accurate in early evolution on both levels but loses robustness as deformation accumulates.
- Failure happens earlier on level 2 (step 10) than on level 1 (step 20), so stabilization should target level-2 robustness first before further SNES level-2 comparisons.

---

## 4) SNES configuration survey (Phase 2, single-step)

Sweep completed on level 1 using:
- [experiment_scripts/bench_he_snes_phase2.py](experiment_scripts/bench_he_snes_phase2.py)
- Output JSON: [experiment_scripts/he_snes_phase2_l1.json](experiment_scripts/he_snes_phase2_l1.json)

Summary (23 tested configs):
- OK: 14
- Diverged: 9
- Error/Timeout: 0

### Best converged configs (level 1)

All entries below reach the correct reference energy (0.346411) with `reason = 2`.

| Rank | Config                                                | Time [s] | SNES iters | Linear iters |   Energy |
| ---: | ----------------------------------------------------- | -------: | ---------: | -----------: | -------: |
|    1 | `newtonls/basic, gmres+asm, rtol=1e-3, atol=1e-5`     |   2.1204 |         20 |         2271 | 0.346411 |
|    2 | `newtonls/basic, gmres+bjacobi, rtol=1e-3, atol=1e-5` |   2.1722 |         20 |         2285 | 0.346411 |
|    3 | `newtonls/basic, gmres+hypre, rtol=1e-1, atol=1e-5`   |   4.6283 |         21 |         3011 | 0.346411 |
|    4 | `newtonls/basic, gmres+hypre, rtol=1e-2, atol=1e-5`   |   5.4398 |         16 |         3046 | 0.346411 |
|    5 | `newtonls/basic, gmres+hypre, rtol=1e-3, atol=1e-5`   |   6.4739 |         16 |         3046 | 0.346411 |

### Diverged configs (level 1)

| Config                                | Reason |      Energy |
| ------------------------------------- | -----: | ----------: |
| `newtonls/basic, cg+hypre`            |     -3 |   29.849161 |
| `newtonls/bt, cg+hypre, objective`    |     -6 |  369.849167 |
| `newtonls/bt, gmres+hypre, objective` |     -6 |  339.656893 |
| `newtonls/l2, gmres+hypre`            |     -3 | 1952.495817 |
| `newtonls/l2, fgmres+hypre`           |     -3 | 1981.989642 |
| `newtontr, cg+hypre, objective`       |    -11 | 3794.813629 |
| `newtontr, gmres+hypre, objective`    |    -11 | 3641.235898 |
| `newtonls/basic, fgmres+asm`          |     -3 | 6227.886553 |
| `newtonls/basic, fgmres+ilu`          |     -3 | 6227.886553 |

### Level-2 status from spot checks

| Mesh level | Config                        | Status                               |
| ---------- | ----------------------------- | ------------------------------------ |
| 2          | `newtonls/basic, gmres+hypre` | PETSc SIGSEGV                        |
| 2          | `newtonls/basic, gmres+ilu`   | Diverged (`reason=-3`, `energy=NaN`) |
| 2          | `newtonls/basic, preonly+lu`  | Diverged (`reason=-9`, `energy=NaN`) |

Notes:
- [HyperElasticity3D_fenics/solve_HE_snes_newton.py](HyperElasticity3D_fenics/solve_HE_snes_newton.py) is now compatible with DOLFINx 0.10 callback signatures (`alpha=` usage).
- Level 2 still needs a dedicated robustness sweep around the level-1 survivors.

---

## 5) Workflow continuation (as planned)

### Phase 2 (next): focus sweep on level 2 using survivors

Carry these survivors from level 1 into level 2:
- `newtonls/basic, gmres+asm`
- `newtonls/basic, gmres+bjacobi`
- `newtonls/basic, gmres+hypre` with `ksp_rtol in {1e-1,1e-2,1e-3}`

Acceptance criterion remains:
- converged reason > 0
- finite energy
- energy matches custom/JAX reference at the same level

### Phase 3: custom Newton full time evolution (level 1)
- Run multi-step trajectory, record per-step energy/iters/time.
- Compare per-step sequence against JAX.

### Phase 4: SNES full time evolution (level 1)
- Run only surviving single-step SNES candidates over full trajectory.
- Keep configs that stay stable and energy-consistent for all steps.

### Phase 5: validate on level 2
- Repeat full evolution with custom Newton and surviving SNES configs.
- Compare stability, energy agreement, and iteration growth.

---

## 6) Failure investigation: restart from JAX states (level 1)

Goal: identify why custom FEniCS fails in late steps while JAX converges.

### Data generation (JAX)

Added [experiment_scripts/gen_he_jax_testdata.py](experiment_scripts/gen_he_jax_testdata.py) to export restart-ready fields:
- `coords`: nodal coordinates
- `u_full_steps`: full **deformed coordinates** per step (all nodes)

Artifacts:
- [experiment_scripts/he_jax_testdata_l1.npz](experiment_scripts/he_jax_testdata_l1.npz)
- [experiment_scripts/he_jax_testdata_l1.json](experiment_scripts/he_jax_testdata_l1.json)

### Restart tests in custom FEniCS (`maxit=1000`)

Runs:
- Step 20 from JAX step 19 state
- Step 24 from JAX step 23 state

Outputs:
- [experiment_scripts/he_custom_restart_step20_maxit1000.json](experiment_scripts/he_custom_restart_step20_maxit1000.json)
- [experiment_scripts/he_custom_restart_step24_maxit1000.json](experiment_scripts/he_custom_restart_step24_maxit1000.json)

Important correction:
- Initial restart loader was wrong in two ways:
	1) It treated JAX fields as displacements (they are deformed coordinates).
	2) It injected values via raw vector indexing with incorrect vector-DOF assumptions.
- Loader was fixed to interpolate by coordinates and convert to displacement (`u = x_deformed - X_ref`).

Validation of corrected restart mapping:
- Step 2 from JAX step 1 now converges correctly to the reference energy:
	- [experiment_scripts/he_custom_restart_step2_from1_check.json](experiment_scripts/he_custom_restart_step2_from1_check.json)
	- energy `1.385634`, iters `18`.

Corrected restart results:
- Step 20 from JAX step 19, `maxit=1000`:
	- [experiment_scripts/he_custom_restart_step20_maxit1000_fixedinit.json](experiment_scripts/he_custom_restart_step20_maxit1000_fixedinit.json)
	- converged, energy `137.392333`, iters `22`.
- Step 24 from JAX step 23, `maxit=1000`:
	- [experiment_scripts/he_custom_restart_step24_maxit1000_fixedinit.json](experiment_scripts/he_custom_restart_step24_maxit1000_fixedinit.json)
	- failed (`energy=NaN`, max iterations reached).

### Why step 24 fails (root cause)

Observed at step 24 (verbose trace):
- NaN appears immediately after Newton update 1.
- With original line search, `alpha` sticks near upper bound (`~2.0`) once energies become NaN.

Line-search stabilization added:
- `tools_petsc4py/minimizers.py` now treats non-finite trial energies as `+inf` and backtracks to finite steps.

Even with NaN-safe line-search:
- Default `CG + HYPRE` still fails at step 24:
	- [experiment_scripts/he_custom_restart_step24_cg_after_lsfix.json](experiment_scripts/he_custom_restart_step24_cg_after_lsfix.json)
	- no NaN, but diverges to very large energy.

KSP/PC A/B tests at step 24 with same corrected initial state:
- `GMRES + HYPRE`, `ksp_rtol=1e-6`:
	- [experiment_scripts/he_custom_restart_step24_gmres_hypre_r1e6.json](experiment_scripts/he_custom_restart_step24_gmres_hypre_r1e6.json)
	- converged, energy `197.748436` (matches JAX `197.748635`), iters `23`.
- `preonly + LU`:
	- [experiment_scripts/he_custom_restart_step24_preonly_lu.json](experiment_scripts/he_custom_restart_step24_preonly_lu.json)
	- converged, energy `197.748420`, iters `23`.

Conclusion:
- The previous “step 20 + 24 both fail at `maxit=1000`” conclusion was invalid due to restart-state loading bugs.
- After fixing restart loading, step 20 converges and step 24 isolates the true issue.
- Primary failure driver is **linear solve choice** in the custom Newton path at high deformation: `CG+HYPRE` is not robust there, while `GMRES+HYPRE` (tight tolerance) or LU recovers the correct solution.
- Secondary issue was line-search handling of non-finite energies; this is now guarded in the PETSc minimizer.

---

## 7) Step-24 convergence profiles and inner-precision sweep

Requested sweep for level 1, step 24 (restart from step 23) over:
- `ksp_rtol = 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6`
- KSP types: `cg`, `gmres`
- PC type: `hypre`

Automation script:
- [experiment_scripts/sweep_he_custom_step24_precision.py](experiment_scripts/sweep_he_custom_step24_precision.py)

Outputs:
- [experiment_scripts/he_step24_precision_sweep/step24_precision_summary.md](experiment_scripts/he_step24_precision_sweep/step24_precision_summary.md)
- [experiment_scripts/he_step24_precision_sweep/step24_precision_summary.json](experiment_scripts/he_step24_precision_sweep/step24_precision_summary.json)
- [experiment_scripts/he_step24_precision_sweep/step24_convergence_profiles.csv](experiment_scripts/he_step24_precision_sweep/step24_convergence_profiles.csv)

### Sweep summary

| KSP           | ksp_rtol range  | Convergence status                  | Typical final energy                             |
| ------------- | --------------- | ----------------------------------- | ------------------------------------------------ |
| `cg+hypre`    | `1e-1 ... 1e-6` | all runs hit max Newton iters (300) | large non-physical values (`~3.2e6` to `~6.8e7`) |
| `gmres+hypre` | `1e-1 ... 1e-6` | all runs converged                  | `197.7484...`                                    |

### Convergence-profile notes

- Per-iteration profiles now include: energy, `dE`, gradient norm, line-search `alpha`, inner `ksp_its`, and line-search eval count.
- Example diverging profile (`cg`, `1e-3`):
	- [experiment_scripts/he_step24_precision_sweep/step24_cg_rtol_1e-03.json](experiment_scripts/he_step24_precision_sweep/step24_cg_rtol_1e-03.json)
- Example converged profile (`gmres`, `1e-6`):
	- [experiment_scripts/he_step24_precision_sweep/step24_gmres_rtol_1e-06.json](experiment_scripts/he_step24_precision_sweep/step24_gmres_rtol_1e-06.json)

Interpretation:
- Tightening inner tolerance alone does not rescue `cg+hypre` at step 24.
- Changing Krylov method to `gmres` is the key factor for robustness in this late nonlinear regime.

---

## 8) Recompute campaign progress (1:1 comparable tables)

Requested sequence: first level 1, start from JAX, then custom-GMRES and level 2.

### 8.1 JAX level 1 (fresh rerun, serial)

Run command:
- `PYTHONPATH=. /tmp/jaxenv/bin/python experiment_scripts/run_he_jax_evolution.py --level 1 --steps 24 --out experiment_scripts/he_jax_evolution_l1_recompute_gmres_compare.json`

Output:
- [experiment_scripts/he_jax_evolution_l1_recompute_gmres_compare.json](experiment_scripts/he_jax_evolution_l1_recompute_gmres_compare.json)

Runtime summary:
- total measured step time: `28.8091 s`
- slowest step: `step 24`, `2.0587 s`
- fastest step: `step 3`, `0.9354 s`

| Step | Time [s] | Newton iters |         Energy | Message                               |
| ---: | -------: | -----------: | -------------: | ------------------------------------- |
|    1 |   0.9555 |           18 |   0.3464113962 | Stopping condition for f is satisfied |
|    2 |   0.9883 |           18 |   1.3856347792 | Stopping condition for f is satisfied |
|    3 |   0.9354 |           18 |   3.1173389157 | Stopping condition for f is satisfied |
|    4 |   0.9383 |           17 |   5.5401476466 | Stopping condition for f is satisfied |
|    5 |   0.9902 |           19 |   8.6504988097 | Stopping condition for f is satisfied |
|    6 |   1.0675 |           20 |  12.4422658099 | Stopping condition for f is satisfied |
|    7 |   1.3990 |           22 |  16.9115637817 | Stopping condition for f is satisfied |
|    8 |   1.1646 |           23 |  22.0617389409 | Stopping condition for f is satisfied |
|    9 |   1.1146 |           21 |  27.8989757279 | Stopping condition for f is satisfied |
|   10 |   1.1420 |           21 |  34.4264986836 | Stopping condition for f is satisfied |
|   11 |   1.1005 |           20 |  41.6441021195 | Stopping condition for f is satisfied |
|   12 |   1.3005 |           21 |  49.5500693409 | Stopping condition for f is satisfied |
|   13 |   1.0327 |           19 |  58.1426609931 | Stopping condition for f is satisfied |
|   14 |   1.2174 |           22 |  67.4207907906 | Stopping condition for f is satisfied |
|   15 |   1.2823 |           23 |  77.3830591107 | Stopping condition for f is satisfied |
|   16 |   1.2632 |           22 |  88.0180575975 | Stopping condition for f is satisfied |
|   17 |   1.2160 |           22 |  99.3269874951 | Stopping condition for f is satisfied |
|   18 |   1.0924 |           21 | 111.3262030649 | Stopping condition for f is satisfied |
|   19 |   1.1635 |           22 | 124.0155386580 | Stopping condition for f is satisfied |
|   20 |   1.1919 |           22 | 137.3923331636 | Stopping condition for f is satisfied |
|   21 |   1.2581 |           22 | 151.4552317421 | Stopping condition for f is satisfied |
|   22 |   1.4789 |           24 | 166.2042207199 | Stopping condition for f is satisfied |
|   23 |   1.4576 |           22 | 181.6387101832 | Stopping condition for f is satisfied |
|   24 |   2.0587 |           34 | 197.7486351731 | Stopping condition for f is satisfied |

Next in this section:
- level-1 custom rerun with default `gmres+hypre`
- then level-2 JAX and level-2 custom, using the same table format for direct 1:1 comparison.

### 8.2 Custom FEniCS level 1 (fresh rerun, default `gmres+hypre`, serial)

Run command:
- `docker run --rm --entrypoint python3 -v "$PWD":/workspace -w /workspace fenics_test HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --out experiment_scripts/he_fenics_custom_evolution_l1_recompute_gmres_compare.json --quiet`

Output:
- [experiment_scripts/he_fenics_custom_evolution_l1_recompute_gmres_compare.json](experiment_scripts/he_fenics_custom_evolution_l1_recompute_gmres_compare.json)

Runtime summary:
- total measured step time: `725.0245 s`
- slowest step: `step 24`, `85.6920 s`
- fastest step: `step 13`, `4.5460 s`

### 8.3 Step-24 only comparison (requested detailed check)

Scope:
- level 1
- step 24 only
- custom initialized from step-23 restart state
- custom Newton cap set to `maxit=100`

Artifacts:
- JAX step-24 profile: [experiment_scripts/he_jax_step24_profile_l1.json](experiment_scripts/he_jax_step24_profile_l1.json)
- Custom sweep summary (`maxit=100`, `gmres+hypre`):
	- [experiment_scripts/he_step24_precision_sweep_maxit100_gmres/step24_precision_summary.md](experiment_scripts/he_step24_precision_sweep_maxit100_gmres/step24_precision_summary.md)
	- [experiment_scripts/he_step24_precision_sweep_maxit100_gmres/step24_precision_summary.json](experiment_scripts/he_step24_precision_sweep_maxit100_gmres/step24_precision_summary.json)

| Solver                        | Inner tol (`ksp_rtol`) | Time [s] | Newton iters | Linear iters (total) | Linear iters (max / Newton step) |         Energy | Status                                |
| ----------------------------- | ---------------------: | -------: | -----------: | -------------------: | -------------------------------: | -------------: | ------------------------------------- |
| JAX (`cg + PyAMG`)            |                   1e-3 |   2.3378 |           34 |                  664 |                               29 | 197.7486351731 | Stopping condition for f is satisfied |
| Custom FEniCS (`gmres+hypre`) |                   1e-1 |  18.3333 |           24 |                10734 |                            10000 |     197.749038 | Energy change converged               |
| Custom FEniCS (`gmres+hypre`) |                   1e-2 |  19.6718 |           22 |                12122 |                            10000 |     197.748883 | Energy change converged               |
| Custom FEniCS (`gmres+hypre`) |                   1e-3 |  69.0097 |           23 |                44061 |                            10000 |     197.748430 | Energy change converged               |
| Custom FEniCS (`gmres+hypre`) |                   1e-6 |  87.8538 |           23 |                56995 |                            10000 |     197.748428 | Energy change converged               |

Notes:
- All requested custom tolerances converged within `maxit=100`.
- Runtime growth is driven by very large inner GMRES work (several Newton steps hit `ksp_its=10000`).
