# GinzburgLandau Implementation Comparison

## Maintained Implementation Roster

| implementation | comparison role |
| --- | --- |
| FEniCS custom Newton | authoritative suite + showcase parity |
| FEniCS SNES trust-region | showcase parity only |
| JAX+PETSc element Hessian | authoritative suite + showcase parity |
| JAX+PETSc local-SFD Hessian | authoritative suite + showcase parity |

## Shared-Case Result Equivalence

The shared showcase case is level `5` at `np=1`. Only converged
implementations are included below.

| implementation | energy | rel. diff vs ref | Newton | linear | wall [s] |
| --- | --- | --- | --- | --- | --- |
| FEniCS custom | 0.346232 | 0.000 | 8 | 33 | 0.1090 |
| FEniCS SNES | 0.346200 | 0.000 | 10 | 0 | 0.0354 |
| JAX+PETSc element | 0.346231 | 0.000 | 8 | 33 | 0.0928 |
| JAX+PETSc local-SFD | 0.346231 | 0.000 | 8 | 33 | 0.2012 |

## Scaling And Speed Comparison

![GinzburgLandau finest-mesh strong scaling preview](img/png/ginzburg_landau/ginzburg_landau_strong_scaling.png)

PDF: [GinzburgLandau strong-scaling PDF](img/pdf/ginzburg_landau/ginzburg_landau_strong_scaling.pdf)

![GinzburgLandau time-vs-mesh-size preview](img/png/ginzburg_landau/ginzburg_landau_mesh_timing.png)

PDF: [GinzburgLandau time-vs-mesh-size PDF](img/pdf/ginzburg_landau/ginzburg_landau_mesh_timing.pdf)
- raw direct speed source: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/direct_speed.csv`
- authoritative maintained scaling source: `replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite/summary.json`

Finest maintained suite scaling (`level 9`):

| implementation | ranks | time [s] | Newton | linear | energy |
| --- | --- | --- | --- | --- | --- |
| FEniCS custom | 1 | 24.4818 | 7 | 22 | 0.345626 |
| FEniCS custom | 2 | 19.6249 | 11 | 42 | 0.345626 |
| FEniCS custom | 4 | 6.2648 | 6 | 22 | 0.345626 |
| FEniCS custom | 8 | 5.1989 | 8 | 32 | 0.345626 |
| FEniCS custom | 16 | 3.2231 | 10 | 33 | 0.345626 |
| FEniCS custom | 32 | 1.8193 | 8 | 37 | 0.345626 |
| JAX+PETSc element | 1 | 15.5759 | 9 | 33 | 0.345626 |
| JAX+PETSc element | 2 | 8.3460 | 7 | 29 | 0.345626 |
| JAX+PETSc element | 4 | 6.2776 | 7 | 27 | 0.345626 |
| JAX+PETSc element | 8 | 6.3502 | 8 | 35 | 0.345626 |
| JAX+PETSc element | 16 | 3.3699 | 8 | 38 | 0.345626 |
| JAX+PETSc element | 32 | 2.1695 | 7 | 39 | 0.345626 |
| JAX+PETSc local-SFD | 1 | 19.4136 | 9 | 33 | 0.345626 |
| JAX+PETSc local-SFD | 2 | 10.4006 | 7 | 29 | 0.345626 |
| JAX+PETSc local-SFD | 4 | 8.3310 | 7 | 27 | 0.345626 |
| JAX+PETSc local-SFD | 8 | 8.6518 | 8 | 35 | 0.345626 |
| JAX+PETSc local-SFD | 16 | 4.4495 | 8 | 38 | 0.345626 |
| JAX+PETSc local-SFD | 32 | 2.7186 | 7 | 39 | 0.345626 |

Fixed-rank mesh-size timing (`32` ranks):

| implementation | level | free DOFs | ranks | time [s] | energy |
| --- | --- | --- | --- | --- | --- |
| FEniCS custom | 5 | 4225 | 32 | 0.3347 | 0.346232 |
| FEniCS custom | 6 | 16630 | 32 | 0.3426 | 0.345777 |
| FEniCS custom | 7 | 66049 | 32 | 0.4332 | 0.345662 |
| FEniCS custom | 8 | 263169 | 32 | 0.7205 | 0.345634 |
| FEniCS custom | 9 | 1050285 | 32 | 1.8193 | 0.345626 |
| JAX+PETSc element | 5 | 3969 | 32 | 0.4161 | 0.346231 |
| JAX+PETSc element | 6 | 16129 | 32 | 0.4588 | 0.345777 |
| JAX+PETSc element | 7 | 65025 | 32 | 0.4982 | 0.345662 |
| JAX+PETSc element | 8 | 261121 | 32 | 12.8838 | 0.444651 |
| JAX+PETSc element | 9 | 1046529 | 32 | 2.1695 | 0.345626 |
| JAX+PETSc local-SFD | 5 | 3969 | 32 | 0.6114 | 0.346231 |
| JAX+PETSc local-SFD | 6 | 16129 | 32 | 0.5811 | 0.345777 |
| JAX+PETSc local-SFD | 7 | 65025 | 32 | 0.6723 | 0.345662 |
| JAX+PETSc local-SFD | 8 | 261121 | 32 | 13.5188 | 0.444651 |
| JAX+PETSc local-SFD | 9 | 1046529 | 32 | 2.7186 | 0.345626 |

## Notes On Exclusions

- All four maintained GinzburgLandau implementations converged on the shared
  level-`5` serial showcase case, so the parity table contains the full working
  roster.
- The authoritative maintained final suite still tracks only the custom FEniCS
  and JAX+PETSc paths; FEniCS SNES remains a showcase-only reference here.

## Raw Outputs And Figures

- replication suite: `replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite`
- publication reruns: `overview/img/runs/ginzburg_landau/showcase`
- curated figures: `overview/img/pdf/ginzburg_landau/` and `overview/img/png/ginzburg_landau/`

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
