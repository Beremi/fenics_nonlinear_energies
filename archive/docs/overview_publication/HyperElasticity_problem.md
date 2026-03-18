# HyperElasticity Problem Overview

## Mathematical Model

The maintained HyperElasticity benchmark minimises the compressible Neo-Hookean
stored energy

$$
\Pi(y)=\int_\Omega C_1\bigl(\operatorname{tr}(F^T F)-3-2\ln J\bigr)
+ D_1(J-1)^2\,dx,
\qquad F = \nabla y, \quad J = \det F,
$$

with $C_1 = 38461538.461538464$ and $D_1 = 83333333.33333333$.
The unknown is the deformation map $y = X + u$, so the solver tracks a
three-component displacement field while preserving the nonlinear elastic
energy structure exactly at the discrete level.

## Geometry, Boundary Conditions, And Setup

- geometry: 3D cantilever beam on $[0,0.4] \times [-0.005,0.005]^2$
- left face: clamped
- right face: prescribed rotating boundary motion
- load path: maintained `24`-step and `96`-step trajectories
- benchmark intent: compare two maintained distributed trust-region paths on a
  genuinely vector-valued large-deformation problem while keeping a pure-JAX
  serial reference on the same energy model

## Discretization And Mesh Source

The maintained benchmark uses vector-valued first-order tetrahedral finite
elements on the canonical meshes under `data/meshes/HyperElasticity/`. The
distributed PETSc paths use block-aware `xyz`-grouped free-DOF ordering,
three-by-three block structure, GAMG coordinates, and rigid-body near-nullspace
vectors so that the elasticity-like linear systems remain scalable on the
largest maintained meshes.

## Maintained Implementations

| implementation | role in overview |
| --- | --- |
| FEniCS custom trust-region Newton | authoritative suite + showcase parity |
| FEniCS SNES | retained comparison point, excluded from parity due failure |
| JAX+PETSc element Hessian | authoritative suite + showcase parity + sample render |
| pure JAX serial | authoritative serial reference + showcase parity |

## Showcase Sample Result

The publication image is exported from a dedicated maintained JAX+PETSc element
render run at level `4` on `32` MPI ranks. The converged maintained
implementations still agree on the shared parity showcase case at level `1`
with `24` load steps to a relative tolerance of approximately `2.778e-05`.
The viewpoint is chosen automatically from the beam's principal-extent axis, so
the render looks straight down the beam length while preserving equal aspect
ratio and orthographic projection.

![HyperElasticity showcase deformed-shape preview](img/png/hyperelasticity/hyperelasticity_sample_state.png)

PDF: [Showcase deformed-shape PDF](img/pdf/hyperelasticity/hyperelasticity_sample_state.pdf)

![HyperElasticity energy-vs-level preview](img/png/hyperelasticity/hyperelasticity_energy_levels.png)

PDF: [Energy-vs-level PDF](img/pdf/hyperelasticity/hyperelasticity_energy_levels.pdf)

## Energy Table Across Levels

The table below uses the maintained `24`-step reference path at `np=1`.

| level | FEniCS custom | JAX+PETSc element | pure JAX serial |
| --- | --- | --- | --- |
| 1 | 197.775 | 197.755 | 197.750 |
| 2 | 116.334 | 116.324 | 116.324 |
| 3 | 93.705 | 93.705 | 93.705 |
| 4 |  |  |  |

## Caveats And Repaired Issues

- `replications/2026-03-16_maintained_refresh/issues/he_jax_petsc_trust_region_cli_flags.md`
- `replications/2026-03-16_maintained_refresh/issues/he_suite_resume_restart.md`

## Commands Used

```bash
./.venv/bin/python -u experiments/runners/run_he_final_suite_best.py --out-dir replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best --no-seed-known-results
```

```bash
./.venv/bin/python -u experiments/runners/run_he_pure_jax_suite_best.py --out-dir replications/2026-03-16_maintained_refresh/runs/hyperelasticity/pure_jax_suite_best
```

```bash
./.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out overview/img/runs/hyperelasticity/showcase/fenics_custom/output.json
```

```bash
./.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --trust_max_reject 6 --nproc 1 --quiet --out overview/img/runs/hyperelasticity/showcase/jax_petsc_element/output.json
```

```bash
./.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py --level 1 --steps 24 --total_steps 24 --quiet --out overview/img/runs/hyperelasticity/showcase/jax_serial/output.json --state-out overview/img/runs/hyperelasticity/showcase/jax_serial/state.npz
```

```bash
mpiexec -n 32 ./.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 4 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --no-pc_setup_on_ksp_cap --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --tolf 1e-4 --tolg 1e-3 --tolg_rel 1e-3 --tolx_rel 1e-3 --tolx_abs 1e-10 --maxit 100 --linesearch_a -0.5 --linesearch_b 2.0 --linesearch_tol 1e-1 --use_trust_region --trust_radius_init 0.5 --trust_radius_min 1e-8 --trust_radius_max 1e6 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --trust_max_reject 6 --trust_subproblem_line_search --nproc 1 --quiet --out overview/img/runs/hyperelasticity/sample_render/jax_petsc_element_l4_np32/output.json --state-out overview/img/runs/hyperelasticity/sample_render/jax_petsc_element_l4_np32/state.npz
```

```bash
./.venv/bin/python overview/img/scripts/build_hyperelasticity_data.py
```

```bash
./.venv/bin/python overview/img/scripts/build_hyperelasticity_figures.py
```
