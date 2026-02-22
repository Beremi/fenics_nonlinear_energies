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

## 3) SNES configuration survey (Phase 2, single-step)

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

## 4) Workflow continuation (as planned)

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
