# Hyperelasticity Model Card

## Mathematical Model

Compressible Neo-Hookean beam benchmark with a rotating right-face displacement boundary condition applied over a fixed sequence of load steps.

## Geometry And Setup

3D cantilever beam, clamped left face, rotating right-face boundary motion, tracked over 24 or 96 load steps.

## Discretization And Mesh Source

Vector P1 finite elements on maintained tetrahedral meshes from the shared HE mesh/data loader.

## Maintained Implementations

| id | implementation | canonical command |
| --- | --- | --- |
| fenics_custom | FEniCS custom Newton | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_custom/output.json |
| fenics_snes | FEniCS SNES | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_snes/output.json |
| jax_serial | pure JAX serial | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py --level 1 --steps 24 --total_steps 24 --quiet --out replications/2026-03-16_maintained_refresh/runs/examples/he_jax_serial/output.json |
| jax_petsc_element | JAX+PETSc element Hessian | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/runs/examples/he_jax_petsc_element/output.json |

## Sample Outputs

### FEniCS custom Newton

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_custom`

```json
{
  "case_id": null,
  "completed_steps": 24,
  "family": "hyperelasticity",
  "final_energy": 197.755003,
  "free_dofs": 0,
  "implementation": "fenics_custom",
  "level": 1,
  "nprocs": 1,
  "result": "completed",
  "setup_time_s": 0.016251,
  "total_dofs": 2187,
  "total_linear_iters": 0,
  "total_newton_iters": 798,
  "total_steps": 24,
  "wall_time_s": 7.7897
}
```

### FEniCS SNES

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_snes`

```json
{
  "case_id": null,
  "completed_steps": 24,
  "family": "hyperelasticity",
  "final_energy": NaN,
  "free_dofs": 0,
  "implementation": "fenics_snes",
  "level": 1,
  "nprocs": 1,
  "result": "failed",
  "setup_time_s": 0.0,
  "total_dofs": 2187,
  "total_linear_iters": 5064,
  "total_newton_iters": 1445,
  "total_steps": 24,
  "wall_time_s": 0.0
}
```

### pure JAX serial

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/he_jax_serial`

```json
{
  "case_id": null,
  "completed_steps": 24,
  "family": "hyperelasticity",
  "final_energy": 197.74950930751214,
  "free_dofs": 2133,
  "implementation": "jax_serial",
  "level": 1,
  "nprocs": 1,
  "result": "completed",
  "setup_time_s": 0.38579492806456983,
  "total_dofs": 2187,
  "total_linear_iters": 2284,
  "total_newton_iters": 559,
  "total_steps": 24,
  "wall_time_s": 29.952550678979605
}
```

### JAX+PETSc element Hessian

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/he_jax_petsc_element`

```json
{
  "case_id": null,
  "completed_steps": 24,
  "family": "hyperelasticity",
  "final_energy": 197.75458224162878,
  "free_dofs": 2133,
  "implementation": "jax_petsc_element",
  "level": 1,
  "nprocs": 1,
  "result": "completed",
  "setup_time_s": 0.570325,
  "total_dofs": 2187,
  "total_linear_iters": 10595,
  "total_newton_iters": 793,
  "total_steps": 24,
  "wall_time_s": 8.49785
}
```

## Replicated Outputs

- Maintained suite: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best`
- Direct speed comparison: `comparisons/hyperelasticity/direct_speed.md`
- Suite scaling summary: `comparisons/hyperelasticity/suite_scaling.md`
- Example runs: `runs/examples/`

## Caveats And Issues

- `he_jax_petsc_trust_region_cli_flags.md`
- `he_suite_resume_restart.md`
