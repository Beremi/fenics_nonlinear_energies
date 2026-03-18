# p-Laplace Model Card

## Mathematical Model

$J(u)=\int_\Omega \frac{1}{p}\lvert\nabla u\rvert^p\,dx - \int_\Omega f u\,dx$ with $p=3$, $f=-10$, and homogeneous Dirichlet boundary conditions on the unit square.

## Geometry And Setup

Unit-square scalar elliptic benchmark with zero displacement on the full boundary.

## Discretization And Mesh Source

P1 Lagrange FE on structured triangular meshes loaded from the maintained mesh pipeline under `data/meshes/`.

## Maintained Implementations

| id | implementation | canonical command |
| --- | --- | --- |
| fenics_custom | FEniCS custom Newton | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_custom/output.json |
| fenics_snes | FEniCS SNES | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_snes/output.json |
| jax_serial | pure JAX serial | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax/solve_pLaplace_jax_newton.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_serial/output.json |
| jax_petsc_element | JAX+PETSc element Hessian | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_petsc_element/output.json |
| jax_petsc_local_sfd | JAX+PETSc local-SFD Hessian | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_local_sfd/run01/output.json |

## Sample Outputs

### FEniCS custom Newton

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_custom`

```json
{
  "case_id": null,
  "family": "plaplace",
  "final_energy": -7.9429687187,
  "implementation": "fenics_custom",
  "level": 5,
  "nprocs": 1,
  "reason": "Converged (energy, step, gradient)",
  "result": "completed",
  "setup_time_s": 0.015,
  "total_dofs": 3201,
  "total_linear_iters": 15,
  "total_newton_iters": 5,
  "wall_time_s": 0.0779
}
```

### FEniCS SNES

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_snes`

```json
{
  "case_id": null,
  "family": "plaplace",
  "final_energy": -7.943,
  "implementation": "fenics_snes",
  "level": 5,
  "nprocs": 1,
  "reason": 2,
  "result": "completed",
  "setup_time_s": 0.0,
  "total_dofs": 3201,
  "total_linear_iters": 0,
  "total_newton_iters": 10,
  "wall_time_s": 0.067
}
```

### pure JAX serial

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_serial`

```json
{
  "case_id": null,
  "family": "plaplace",
  "final_energy": -7.943,
  "implementation": "jax_serial",
  "level": 5,
  "nprocs": 1,
  "reason": "Stopping condition for f is satisfied",
  "result": "completed",
  "setup_time_s": 0.2521,
  "total_dofs": 2945,
  "total_linear_iters": 0,
  "total_newton_iters": 6,
  "wall_time_s": 0.1049
}
```

### JAX+PETSc element Hessian

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_petsc_element`

```json
{
  "case_id": null,
  "family": "plaplace",
  "final_energy": -7.9429687193,
  "implementation": "jax_petsc_element",
  "level": 5,
  "nprocs": 1,
  "reason": "Converged (energy, step, gradient)",
  "result": "completed",
  "setup_time_s": 0.2264,
  "total_dofs": 3201,
  "total_linear_iters": 17,
  "total_newton_iters": 6,
  "wall_time_s": 0.0801
}
```

### JAX+PETSc local-SFD Hessian

Command leaf: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_local_sfd/run01`

```json
{
  "case_id": "l5_np1",
  "family": "plaplace",
  "final_energy": -7.9429687193,
  "implementation": "jax_petsc_local_sfd",
  "level": 5,
  "nprocs": 1,
  "reason": "Converged (energy, step, gradient)",
  "result": "completed",
  "setup_time_s": 0.5331,
  "total_dofs": 3201,
  "total_linear_iters": 17,
  "total_newton_iters": 6,
  "wall_time_s": 0.178
}
```

## Replicated Outputs

- Maintained suite: `replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite`
- Direct speed comparison: `comparisons/plaplace/direct_speed.md`
- Suite scaling summary: `comparisons/plaplace/suite_scaling.md`
- Example runs: `runs/examples/`

## Caveats And Issues

- `plaplace_fenics_snes_parallel_mesh_construction.md`
