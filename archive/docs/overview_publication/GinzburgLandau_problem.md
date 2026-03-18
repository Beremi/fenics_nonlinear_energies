# GinzburgLandau Problem Overview

## Mathematical Model

We solve the non-convex scalar Ginzburg-Landau energy minimisation problem

$$
E(u)=\int_\Omega \frac{\varepsilon}{2}\lvert \nabla u \rvert^2
+ \frac{1}{4}(u^2-1)^2\,dx,
\qquad \varepsilon = 10^{-2},
$$

on $\Omega=[-1,1]^2$ with homogeneous Dirichlet boundary data $u=0$ on
$\partial\Omega$. The benchmark sits in the non-convex regime where the
double-well potential competes against the gradient regularisation, so the
nonlinear solves must pass through indefinite local curvature while still
converging to the same discrete minimiser.

## Geometry, Boundary Conditions, And Setup

- domain: square $[-1,1]^2$
- boundary condition: homogeneous Dirichlet on the full boundary
- benchmark hierarchy: maintained mesh levels $5,6,7,8,9$
- difficulty: non-convex double-well potential with indefinite Hessian regions
- benchmark intent: compare custom Newton logic against JAX-derived PETSc
  Hessian paths on a fixed non-convex scalar model

## Discretization And Mesh Source

All maintained implementations use first-order Lagrange finite elements on the
canonical triangular meshes in `data/meshes/GinzburgLandau/`. As in pLaplace,
the problem is one degree of freedom per node, so the mesh-size timing figure
tracks free-DOF growth directly.

## Maintained Implementations

| implementation | role in overview |
| --- | --- |
| FEniCS custom Newton | authoritative suite + showcase state export |
| FEniCS SNES trust-region | showcase comparison only |
| JAX+PETSc element Hessian | authoritative suite + showcase comparison |
| JAX+PETSc local-SFD Hessian | authoritative suite + showcase comparison |

## Showcase Sample Result

The publication showcase field is exported from the serial FEniCS custom Newton
rerun at level `5`. The converged maintained implementations agree on the final
showcase energy to a relative tolerance of approximately `3.237e-05`.

![GinzburgLandau showcase solution preview](img/png/ginzburg_landau/ginzburg_landau_sample_state.png)

PDF: [Showcase solution PDF](img/pdf/ginzburg_landau/ginzburg_landau_sample_state.pdf)

![GinzburgLandau energy-vs-level preview](img/png/ginzburg_landau/ginzburg_landau_energy_levels.png)

PDF: [Energy-vs-level PDF](img/pdf/ginzburg_landau/ginzburg_landau_energy_levels.pdf)

## Energy Table Across Levels

The level table below uses the authoritative maintained benchmark suite at
`np=1`.

| level | FEniCS custom | JAX+PETSc element | JAX+PETSc local-SFD |
| --- | --- | --- | --- |
| 5 | 0.346232 | 0.346231 | 0.346231 |
| 6 | 0.345777 | 0.345777 | 0.345777 |
| 7 | 0.345662 | 0.345662 | 0.345662 |
| 8 | 0.345634 | 0.345634 | 0.345634 |
| 9 | 0.345626 | 0.345626 | 0.345626 |

## Caveats And Repaired Issues

- `replications/2026-03-16_maintained_refresh/issues/gl_jax_petsc_direct_output_schema.md`

## Commands Used

```bash
./.venv/bin/python -u experiments/runners/run_gl_final_suite.py --out-dir replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite
```

```bash
./.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json overview/img/runs/ginzburg_landau/showcase/fenics_custom/output.json --state-out overview/img/runs/ginzburg_landau/showcase/fenics_custom/state.npz
```

```bash
./.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json overview/img/runs/ginzburg_landau/showcase/fenics_snes/output.json
```

```bash
./.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out overview/img/runs/ginzburg_landau/showcase/jax_petsc_element/output.json
```

```bash
./.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out overview/img/runs/ginzburg_landau/showcase/jax_petsc_local_sfd/output.json
```

```bash
./.venv/bin/python overview/img/scripts/build_ginzburg_landau_data.py
```

```bash
./.venv/bin/python overview/img/scripts/build_ginzburg_landau_figures.py
```
