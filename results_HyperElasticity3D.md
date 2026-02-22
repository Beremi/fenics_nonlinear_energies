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

### 8.4 Step-24 preconditioner/nullspace check (`ksp_rtol=1e-1`)

To test the hypothesis that preconditioning/nullspace setup is the bottleneck, step 24 was rerun with alternative solver setups (all with `maxit=100`, restart from step 23):

| Setup                                                 | Time [s] | Newton iters |     Energy | Sum inner iters | Max inner iters | Notes                                  |
| ----------------------------------------------------- | -------: | -----------: | ---------: | --------------: | --------------: | -------------------------------------- |
| `gmres+hypre`, near-nullspace ON, `ksp_max_it=10000`  |  18.2403 |           24 | 197.749038 |           10734 |           10000 | baseline                               |
| `gmres+hypre`, near-nullspace OFF, `ksp_max_it=10000` |   3.8189 |           25 | 197.748827 |           10540 |           10000 | much faster than baseline on this case |
| `gmres+hypre`, near-nullspace ON, `ksp_max_it=500`    |   2.5541 |           24 | 197.749042 |            1236 |             500 | same energy quality, much lower cost   |
| `fgmres+hypre`, near-nullspace ON, `ksp_max_it=10000` |  66.6549 |           36 | 197.749054 |           43239 |           10000 | significantly slower                   |
| `gmres+gamg`, near-nullspace ON, `ksp_max_it=10000`   |   0.1653 |           24 | 199.852426 |             123 |              24 | fast but wrong energy (not acceptable) |

Artifacts:
- [experiment_scripts/he_step24_setup_baseline_gmres_hypre_1e1.json](experiment_scripts/he_step24_setup_baseline_gmres_hypre_1e1.json)
- [experiment_scripts/he_step24_setup_no_nearnull_gmres_hypre_1e1.json](experiment_scripts/he_step24_setup_no_nearnull_gmres_hypre_1e1.json)
- [experiment_scripts/he_step24_setup_gmres_hypre_1e1_ksp500.json](experiment_scripts/he_step24_setup_gmres_hypre_1e1_ksp500.json)
- [experiment_scripts/he_step24_setup_fgmres_hypre_1e1.json](experiment_scripts/he_step24_setup_fgmres_hypre_1e1.json)
- [experiment_scripts/he_step24_setup_gmres_gamg_1e1.json](experiment_scripts/he_step24_setup_gmres_gamg_1e1.json)

Takeaway for next iteration:
- The strongest immediate improvement is capping inner iterations (`ksp_max_it`) while keeping `gmres+hypre`.
- In this nonlinear step, the current near-nullspace attachment does not help time-to-solution and should be revisited (or toggled per regime).

### 8.5 HYPRE option sweep with near-nullspace ON (as requested)

Setup fixed for all runs:
- level 1, step 24 (restart from step 23)
- `ksp_type=gmres`, `pc_type=hypre`
- `ksp_rtol=1e-1`, `ksp_max_it=500`, `maxit=100`
- near-nullspace enabled

Sweep artifacts:
- [experiment_scripts/sweep_he_step24_hypre_options.py](experiment_scripts/sweep_he_step24_hypre_options.py)
- [experiment_scripts/he_step24_hypre_options/summary.md](experiment_scripts/he_step24_hypre_options/summary.md)
- [experiment_scripts/he_step24_hypre_options/summary.json](experiment_scripts/he_step24_hypre_options/summary.json)

| Case                      | Time [s] | Newton iters |     Energy | Sum inner iters | Max inner iters |
| ------------------------- | -------: | -----------: | ---------: | --------------: | --------------: |
| baseline `nodal=6, vec=3` |   2.5543 |           24 | 197.749042 |            1236 |             500 |
| `strong_threshold=0.5`    |   2.9402 |           26 | 197.748986 |            2156 |             500 |
| `strong_threshold=0.7`    |   1.4778 |           24 | 197.748889 |            1226 |             500 |
| `coarsen_type=HMIS`       |   0.9583 |           26 | 197.748838 |            1017 |             183 |
| `coarsen_type=PMIS`       |   1.4135 |           26 | 197.748872 |            1877 |             500 |
| `nodal=1, vec=2`          |   1.1751 |           23 | 197.748544 |             809 |             444 |
| `nodal=4, vec=2`          |   0.9087 |           23 | 197.749034 |             533 |             136 |
| skip setting `nodal/vec`  |   0.5017 |           24 | 197.748455 |             821 |             500 |

Interpretation:
- All tested HYPRE variants converged with acceptable energy (~`197.7484` to `197.7490`).
- Best time in this sweep came from leaving `nodal_coarsen` and `vec_interp_variant` unset (HYPRE defaults).
- Among explicit configurations, `nodal=4, vec=2` and `coarsen_type=HMIS` are the strongest candidates.

### 8.6 Detailed step-24 profile: `nodal=4, vec=2` vs skip `nodal/vec`

Requested side-by-side profile using:
- level 1, step 24 (restart from step 23)
- `ksp_type=gmres`, `pc_type=hypre`, `ksp_rtol=1e-1`, `ksp_max_it=500`, `maxit=100`
- with `--save_history --save_linear_timing`

Artifacts:
- `nodal=4, vec=2`: [experiment_scripts/he_step24_profile_nodal4_vec2.json](experiment_scripts/he_step24_profile_nodal4_vec2.json)
- skip `nodal/vec`: [experiment_scripts/he_step24_profile_skip_nodal_vec.json](experiment_scripts/he_step24_profile_skip_nodal_vec.json)

Aggregate comparison over Newton iterations:

| Config           | Step time [s] | Newton iters |     Energy | Sum KSP iters | Avg KSP / Newton | Sum PC setup [s] | Sum solve [s] | Sum linear total [s] |
| ---------------- | ------------: | -----------: | ---------: | ------------: | ---------------: | ---------------: | ------------: | -------------------: |
| `nodal=4, vec=2` |        1.1499 |           24 | 197.748847 |           839 |            34.96 |         0.290183 |      0.743078 |             1.093878 |
| skip `nodal/vec` |        0.4966 |           24 | 197.748455 |           821 |            34.21 |         0.101776 |      0.281270 |             0.443358 |

Per-Newton iteration linear profile (requested detail):

| Newton it | `ksp_its` (`nodal=4,vec=2`) | `pc_setup_time` [s] | `solve_time` [s] | `linear_total_time` [s] | `ksp_its` (skip) | `pc_setup_time` [s] | `solve_time` [s] | `linear_total_time` [s] |
| --------: | --------------------------: | ------------------: | ---------------: | ----------------------: | ---------------: | ------------------: | ---------------: | ----------------------: |
|         1 |                          17 |            0.013180 |         0.016198 |                0.031592 |               28 |            0.004682 |         0.009621 |                0.016528 |
|         2 |                           4 |            0.012047 |         0.004470 |                0.019051 |                7 |            0.004790 |         0.002674 |                0.010002 |
|         3 |                          18 |            0.013246 |         0.016900 |                0.032666 |                5 |            0.004266 |         0.002031 |                0.008824 |
|         4 |                           6 |            0.012148 |         0.005982 |                0.020660 |               19 |            0.004235 |         0.006587 |                0.013355 |
|         5 |                          16 |            0.011894 |         0.014578 |                0.029018 |                7 |            0.004236 |         0.002716 |                0.009504 |
|         6 |                           7 |            0.011899 |         0.006724 |                0.021159 |                6 |            0.004179 |         0.002384 |                0.009113 |
|         7 |                         144 |            0.011971 |         0.125461 |                0.139967 |                6 |            0.004183 |         0.002346 |                0.009052 |
|         8 |                          21 |            0.012164 |         0.020136 |                0.034849 |                9 |            0.004196 |         0.003366 |                0.010102 |
|         9 |                          14 |            0.010947 |         0.011570 |                0.025056 |                9 |            0.004127 |         0.003280 |                0.009948 |
|        10 |                          11 |            0.012071 |         0.010139 |                0.024741 |               12 |            0.004447 |         0.004317 |                0.011273 |
|        11 |                          22 |            0.012661 |         0.020836 |                0.036035 |                9 |            0.004411 |         0.003498 |                0.010441 |
|        12 |                          19 |            0.011810 |         0.016938 |                0.031287 |               15 |            0.004157 |         0.005243 |                0.011922 |
|        13 |                          25 |            0.011760 |         0.021545 |                0.035845 |               10 |            0.004214 |         0.003680 |                0.010398 |
|        14 |                          12 |            0.011424 |         0.010622 |                0.024590 |               16 |            0.004073 |         0.005434 |                0.012009 |
|        15 |                          45 |            0.011215 |         0.037385 |                0.051134 |               11 |            0.004125 |         0.003922 |                0.010554 |
|        16 |                          22 |            0.011659 |         0.019610 |                0.033802 |               19 |            0.004121 |         0.006476 |                0.013107 |
|        17 |                          14 |            0.012231 |         0.012978 |                0.027745 |                9 |            0.004119 |         0.003255 |                0.009885 |
|        18 |                          19 |            0.011914 |         0.016807 |                0.031259 |               29 |            0.004099 |         0.009698 |                0.016328 |
|        19 |                          19 |            0.011925 |         0.016594 |                0.031058 |               10 |            0.004209 |         0.003667 |                0.010420 |
|        20 |                          17 |            0.012448 |         0.015636 |                0.030632 |               29 |            0.004167 |         0.009982 |                0.016668 |
|        21 |                          27 |            0.012783 |         0.025083 |                0.040406 |               15 |            0.004168 |         0.005290 |                0.011976 |
|        22 |                          22 |            0.011862 |         0.019257 |                0.033666 |               26 |            0.004213 |         0.008991 |                0.015712 |
|        23 |                         312 |            0.012085 |         0.271315 |                0.285955 |               15 |            0.004188 |         0.005365 |                0.012104 |
|        24 |                           6 |            0.012839 |         0.006314 |                0.021705 |              500 |            0.004171 |         0.167447 |                0.174133 |

Observation:
- Skipping explicit `nodal/vec` keeps similar total KSP work, but reduces `pc_setup_time` by roughly 3x per Newton step and lowers total linear time by ~2.47x on this case.

### 8.7 Variant requested: setup PC only after cap-hit (`ksp_max_it=30`)

Requested variant behavior:
- use `ksp_max_it` as hard per-solve cap
- do not run `ksp.setUp()` each Newton iteration
- only run `ksp.setUp()` when the **previous** solve hit cap (`ksp_its == ksp_max_it`)

Run settings:
- level 1, step 24 (restart from step 23)
- `ksp_type=gmres`, `pc_type=hypre`, `ksp_rtol=1e-1`, `ksp_max_it=30`, `maxit=100`
- `--pc_setup_on_ksp_cap --save_history --save_linear_timing`

Artifacts:
- `nodal=4, vec=2`: [experiment_scripts/he_step24_profile_nodal4_vec2_ksp30_pc_cap.json](experiment_scripts/he_step24_profile_nodal4_vec2_ksp30_pc_cap.json)
- skip `nodal/vec`: [experiment_scripts/he_step24_profile_skip_nodal_vec_ksp30_pc_cap.json](experiment_scripts/he_step24_profile_skip_nodal_vec_ksp30_pc_cap.json)

Aggregate comparison (this variant):

| Config           | Step time [s] | Newton iters |     Energy | Sum KSP iters | Avg KSP / Newton | Sum PC setup [s] | Sum solve [s] | Sum linear total [s] |
| ---------------- | ------------: | -----------: | ---------: | ------------: | ---------------: | ---------------: | ------------: | -------------------: |
| `nodal=4, vec=2` |        0.8374 |           24 | 197.748941 |           484 |            20.17 |         0.060364 |      0.663018 |             0.783608 |
| skip `nodal/vec` |        0.3418 |           24 | 197.748450 |           351 |            14.62 |         0.004606 |      0.222389 |             0.287599 |

Per-Newton iteration linear profile (`ksp_max_it=30`, cap-triggered setup):

| Newton it | `ksp_its` (`nodal=4,vec=2`) | `pc_setup_time` [s] | `solve_time` [s] | `linear_total_time` [s] | `ksp_its` (skip) | `pc_setup_time` [s] | `solve_time` [s] | `linear_total_time` [s] |
| --------: | --------------------------: | ------------------: | ---------------: | ----------------------: | ---------------: | ------------------: | ---------------: | ----------------------: |
|         1 |                          17 |            0.012971 |         0.015987 |                0.031178 |               28 |            0.004606 |         0.009604 |                0.016440 |
|         2 |                           4 |            0.000000 |         0.016418 |                0.018930 |                7 |            0.000000 |         0.006889 |                0.009432 |
|         3 |                          18 |            0.000000 |         0.030510 |                0.033015 |                5 |            0.000000 |         0.006181 |                0.008708 |
|         4 |                           6 |            0.000000 |         0.018295 |                0.020848 |               19 |            0.000000 |         0.010701 |                0.013220 |
|         5 |                          16 |            0.000000 |         0.026900 |                0.029407 |                7 |            0.000000 |         0.006897 |                0.009430 |
|         6 |                           7 |            0.000000 |         0.018719 |                0.021233 |                6 |            0.000000 |         0.008040 |                0.010564 |
|         7 |                          30 |            0.000000 |         0.038037 |                0.040542 |                6 |            0.000000 |         0.006635 |                0.009244 |
|         8 |                          27 |            0.011468 |         0.022560 |                0.036539 |                9 |            0.000000 |         0.007685 |                0.010265 |
|         9 |                          14 |            0.000000 |         0.022439 |                0.024962 |                9 |            0.000000 |         0.007593 |                0.010178 |
|        10 |                          13 |            0.000000 |         0.024196 |                0.026711 |               12 |            0.000000 |         0.008352 |                0.010862 |
|        11 |                          23 |            0.000000 |         0.034343 |                0.036856 |                9 |            0.000000 |         0.007427 |                0.009943 |
|        12 |                          16 |            0.000000 |         0.024682 |                0.027201 |               15 |            0.000000 |         0.009539 |                0.012061 |
|        13 |                          30 |            0.000000 |         0.039630 |                0.042142 |               10 |            0.000000 |         0.007991 |                0.010523 |
|        14 |                          14 |            0.012008 |         0.013396 |                0.027933 |               16 |            0.000000 |         0.009531 |                0.012061 |
|        15 |                          20 |            0.000000 |         0.029297 |                0.031820 |               11 |            0.000000 |         0.008085 |                0.010619 |
|        16 |                          20 |            0.000000 |         0.028849 |                0.031368 |               19 |            0.000000 |         0.010721 |                0.013251 |
|        17 |                          26 |            0.000000 |         0.036044 |                0.038569 |                9 |            0.000000 |         0.007467 |                0.009996 |
|        18 |                          22 |            0.000000 |         0.030786 |                0.033304 |               29 |            0.000000 |         0.013928 |                0.016458 |
|        19 |                          26 |            0.000000 |         0.034596 |                0.037136 |               10 |            0.000000 |         0.007891 |                0.010422 |
|        20 |                          18 |            0.000000 |         0.028420 |                0.030946 |               29 |            0.000000 |         0.014304 |                0.016844 |
|        21 |                          27 |            0.000000 |         0.038236 |                0.040772 |               15 |            0.000000 |         0.009533 |                0.012073 |
|        22 |                          30 |            0.000000 |         0.038589 |                0.041121 |               26 |            0.000000 |         0.013232 |                0.015762 |
|        23 |                          30 |            0.011981 |         0.026235 |                0.040750 |               15 |            0.000000 |         0.009598 |                0.012144 |
|        24 |                          30 |            0.011936 |         0.025854 |                0.040325 |               30 |            0.000000 |         0.014565 |                0.017099 |

Comparison vs previous end-table setting (`ksp_max_it=500`, setup every Newton step):
- `nodal=4, vec=2`: step time `1.1499 → 0.8374` (faster), sum KSP `839 → 484`, sum PC setup `0.290183 → 0.060364`.
- skip `nodal/vec`: step time `0.4966 → 0.3418` (faster), sum KSP `821 → 351`, sum PC setup `0.101776 → 0.004606`.

### 8.8 Full evolution rerun with this setting (skip `nodal/vec`, `ksp_rtol=1e-1`, `ksp_max_it=30`)

Requested full-trajectory rerun using the new variant:
- `ksp_type=gmres`, `pc_type=hypre`
- skip explicit `hypre_nodal_coarsen`/`hypre_vec_interp_variant` (`-1`, `-1`)
- `ksp_rtol=1e-1`, `ksp_max_it=30`
- `--pc_setup_on_ksp_cap` (PC setup only after previous cap hit; first Newton solve still sets up)

Artifact:
- [experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap.json](experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap.json)

Run summary:
- total step time: `6.9156 s`
- total Newton iterations: `527`
- total inner linear iterations: `7503`
- average inner iterations per Newton step: `14.24`

Per-step metrics (requested table):

| Step | Time [s] | Newton iters | Sum linear iters | Avg linear / Newton |     Energy | Status                  |
| ---: | -------: | -----------: | ---------------: | ------------------: | ---------: | ----------------------- |
|    1 |   0.2417 |           22 |              331 |               15.05 |   0.346411 | Energy change converged |
|    2 |   0.2728 |           23 |              380 |               16.52 |   1.385640 | Energy change converged |
|    3 |   0.2456 |           21 |              292 |               13.90 |   3.117343 | Energy change converged |
|    4 |   0.2618 |           22 |              301 |               13.68 |   5.540148 | Energy change converged |
|    5 |   0.2743 |           22 |              334 |               15.18 |   8.650501 | Energy change converged |
|    6 |   0.3076 |           24 |              378 |               15.75 |  12.442268 | Energy change converged |
|    7 |   0.2929 |           23 |              349 |               15.17 |  16.911565 | Energy change converged |
|    8 |   0.2871 |           23 |              323 |               14.04 |  22.061741 | Energy change converged |
|    9 |   0.2523 |           20 |              268 |               13.40 |  27.898977 | Energy change converged |
|   10 |   0.2487 |           20 |              258 |               12.90 |  34.426502 | Energy change converged |
|   11 |   0.3080 |           23 |              318 |               13.83 |  41.644106 | Energy change converged |
|   12 |   0.2718 |           21 |              270 |               12.86 |  49.550071 | Energy change converged |
|   13 |   0.2621 |           20 |              261 |               13.05 |  58.142661 | Energy change converged |
|   14 |   0.2768 |           21 |              268 |               12.76 |  67.420791 | Energy change converged |
|   15 |   0.2766 |           21 |              290 |               13.81 |  77.383059 | Energy change converged |
|   16 |   0.3145 |           23 |              324 |               14.09 |  88.018059 | Energy change converged |
|   17 |   0.3082 |           22 |              306 |               13.91 |  99.326987 | Energy change converged |
|   18 |   0.3278 |           23 |              346 |               15.04 | 111.326205 | Energy change converged |
|   19 |   0.3347 |           22 |              338 |               15.36 | 124.015539 | Energy change converged |
|   20 |   0.3108 |           22 |              328 |               14.91 | 137.392334 | Energy change converged |
|   21 |   0.3209 |           23 |              322 |               14.00 | 151.455236 | Energy change converged |
|   22 |   0.2920 |           21 |              291 |               13.86 | 166.204220 | Energy change converged |
|   23 |   0.3205 |           23 |              319 |               13.87 | 181.638709 | Energy change converged |
|   24 |   0.3061 |           22 |              308 |               14.00 | 197.755179 | Energy change converged |

### 8.9 Full evolution rerun with increased tolerance + JAX relative errors

Interpretation used for "increased tolerance": looser inner tolerance, from `ksp_rtol=1e-1` to `ksp_rtol=5e-1`.

Rerun settings:
- same skip variant as 8.8 (`gmres+hypre`, skip `nodal/vec`, `ksp_max_it=30`, `--pc_setup_on_ksp_cap`)
- only changed parameter: `ksp_rtol=5e-1`

Artifacts:
- custom rerun: [experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap_rtol5e1.json](experiment_scripts/he_fenics_custom_evolution_l1_skip_ksp30_pc_cap_rtol5e1.json)
- JAX reference: [experiment_scripts/he_jax_evolution_l1_recompute_gmres_compare.json](experiment_scripts/he_jax_evolution_l1_recompute_gmres_compare.json)

Run summary:
- total step time: `7.8835 s`
- total Newton iterations: `732`
- total inner linear iterations: `4705`
- average inner iterations per Newton step: `6.43`
- max relative energy error vs JAX: `4.68e-3` (step 15)
- mean relative energy error vs JAX: `2.19e-4`

Per-step metrics with JAX comparison:

| Step | Time [s] | Newton iters | Sum linear iters | Avg linear / Newton | Custom energy | JAX energy | Relative error | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.2941 | 35 | 183 | 5.23 | 0.346486 | 0.3464113962 | 2.15e-04 | Energy change converged |
| 2 | 0.3251 | 36 | 207 | 5.75 | 1.385651 | 1.3856347792 | 1.17e-05 | Energy change converged |
| 3 | 0.3071 | 32 | 202 | 6.31 | 3.117429 | 3.1173389157 | 2.89e-05 | Energy change converged |
| 4 | 0.3184 | 32 | 203 | 6.34 | 5.540152 | 5.5401476466 | 7.86e-07 | Energy change converged |
| 5 | 0.3117 | 31 | 208 | 6.71 | 8.650536 | 8.6504988097 | 4.30e-06 | Energy change converged |
| 6 | 0.3021 | 30 | 164 | 5.47 | 12.442269 | 12.4422658099 | 2.56e-07 | Energy change converged |
| 7 | 0.2895 | 29 | 158 | 5.45 | 16.911673 | 16.9115637817 | 6.46e-06 | Energy change converged |
| 8 | 0.3296 | 33 | 174 | 5.27 | 22.061766 | 22.0617389409 | 1.23e-06 | Energy change converged |
| 9 | 0.3142 | 31 | 158 | 5.10 | 27.899011 | 27.8989757279 | 1.26e-06 | Energy change converged |
| 10 | 0.3291 | 32 | 171 | 5.34 | 34.426511 | 34.4264986836 | 3.58e-07 | Energy change converged |
| 11 | 0.3421 | 33 | 175 | 5.30 | 41.644106 | 41.6441021195 | 9.32e-08 | Energy change converged |
| 12 | 0.3329 | 31 | 177 | 5.71 | 49.550096 | 49.5500693409 | 5.38e-07 | Energy change converged |
| 13 | 0.3676 | 29 | 177 | 6.10 | 58.142785 | 58.1426609931 | 2.13e-06 | Energy change converged |
| 14 | 0.3508 | 32 | 180 | 5.62 | 67.420796 | 67.4207907906 | 7.73e-08 | Energy change converged |
| 15 | 0.1506 | 13 | 91 | 7.00 | 77.745148 | 77.3830591107 | 4.68e-03 | Energy change converged |
| 16 | 0.4215 | 35 | 313 | 8.94 | 88.018062 | 88.0180575975 | 5.00e-08 | Energy change converged |
| 17 | 0.3405 | 29 | 229 | 7.90 | 99.327001 | 99.3269874951 | 1.36e-07 | Energy change converged |
| 18 | 0.2499 | 22 | 148 | 6.73 | 111.355575 | 111.3262030649 | 2.64e-04 | Energy change converged |
| 19 | 0.3402 | 29 | 226 | 7.79 | 124.015546 | 124.0155386580 | 5.92e-08 | Energy change converged |
| 20 | 0.3610 | 30 | 250 | 8.33 | 137.392354 | 137.3923331636 | 1.52e-07 | Energy change converged |
| 21 | 0.3499 | 30 | 216 | 7.20 | 151.455258 | 151.4552317421 | 1.73e-07 | Energy change converged |
| 22 | 0.3848 | 33 | 238 | 7.21 | 166.204272 | 166.2042207199 | 3.09e-07 | Energy change converged |
| 23 | 0.3371 | 30 | 173 | 5.77 | 181.639341 | 181.6387101832 | 3.47e-06 | Energy change converged |
| 24 | 0.4337 | 35 | 284 | 8.11 | 197.754043 | 197.7486351731 | 2.73e-05 | Energy change converged |
