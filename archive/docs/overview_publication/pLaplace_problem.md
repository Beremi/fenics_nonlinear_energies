# pLaplace Problem Overview

## Mathematical Model

We solve the nonlinear scalar p-Laplace minimisation problem

$$
E(u)=\int_\Omega \frac{1}{p}\lvert \nabla u \rvert^p\,dx - \int_\Omega f u\,dx,
\qquad p=3,\quad f=-10,
$$

on the unit square $\Omega=(0,1)^2$ with homogeneous Dirichlet data
$u=0$ on $\partial\Omega$. The Euler-Lagrange equation is the nonlinear
diffusion problem

$$
-\nabla \cdot \left(\lvert \nabla u \rvert^{p-2} \nabla u \right) = f,
$$

so the benchmark couples a strongly nonlinear constitutive law with a fixed
second-order elliptic geometry.

## Geometry, Boundary Conditions, And Setup

- domain: unit square
- boundary condition: homogeneous Dirichlet on the full boundary
- forcing: constant negative load $f=-10$
- benchmark hierarchy: maintained mesh levels $5,6,7,8,9$
- benchmark intent: compare custom nonlinear solvers against JAX-derived
  Hessian paths on the same finite-element problem

## Discretization And Mesh Source

The maintained implementation uses first-order Lagrange finite elements on the
canonical triangular meshes in `data/meshes/pLaplace/`. The problem is one
degree of freedom per node, so mesh refinement directly increases the free-DOF
count used in the solver and scaling figures.

## Maintained Implementations

| implementation | role in overview |
| --- | --- |
| FEniCS custom Newton | authoritative suite + showcase state export |
| FEniCS SNES | showcase comparison only |
| pure JAX serial | showcase comparison only |
| JAX+PETSc element Hessian | authoritative suite + showcase comparison |
| JAX+PETSc local-SFD Hessian | authoritative suite + showcase comparison |

## Showcase Sample Result

The publication showcase field is exported from the serial FEniCS custom Newton
rerun at level `5`. The converged maintained implementations agree on the final
scalar energy on this shared showcase case to a relative tolerance of
approximately `3.938e-06`.

![pLaplace showcase solution preview](img/png/plaplace/plaplace_sample_state.png)

PDF: [Showcase solution PDF](img/pdf/plaplace/plaplace_sample_state.pdf)

![pLaplace energy-vs-level preview](img/png/plaplace/plaplace_energy_levels.png)

PDF: [Energy-vs-level PDF](img/pdf/plaplace/plaplace_energy_levels.pdf)

## Energy Table Across Levels

The level table below uses the authoritative maintained benchmark suite at
`np=1`.

| level | FEniCS custom | JAX+PETSc element | JAX+PETSc local-SFD |
| --- | --- | --- | --- |
| 5 | -7.942969 | -7.942969 | -7.942969 |
| 6 | -7.954564 | -7.954564 | -7.954564 |
| 7 | -7.958292 | -7.958292 | -7.958292 |
| 8 | -7.959556 | -7.959556 | -7.959556 |
| 9 | -7.960005 | -7.960004 | -7.960004 |

## Caveats And Repaired Issues

- `replications/2026-03-16_maintained_refresh/issues/plaplace_fenics_snes_parallel_mesh_construction.md`

## Commands Used

```bash
./.venv/bin/python -u experiments/runners/run_plaplace_final_suite.py --out-dir replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite
```

```bash
./.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json overview/img/runs/plaplace/showcase/fenics_custom/output.json --state-out overview/img/runs/plaplace/showcase/fenics_custom/state.npz
```

```bash
./.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json overview/img/runs/plaplace/showcase/fenics_snes/output.json
```

```bash
./.venv/bin/python -u src/problems/plaplace/jax/solve_pLaplace_jax_newton.py --levels 5 --quiet --json overview/img/runs/plaplace/showcase/jax_serial/output.json
```

```bash
./.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json overview/img/runs/plaplace/showcase/jax_petsc_element/output.json
```

```bash
./.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json overview/img/runs/plaplace/showcase/jax_petsc_local_sfd/output.json
```

```bash
./.venv/bin/python overview/img/scripts/build_plaplace_data.py
```

```bash
./.venv/bin/python overview/img/scripts/build_plaplace_figures.py
```
