# pLaplace Implementation Comparison

## Maintained Implementation Roster

| implementation | comparison role |
| --- | --- |
| FEniCS custom Newton | authoritative suite + showcase parity |
| FEniCS SNES | showcase parity only |
| pure JAX serial | showcase parity only |
| JAX+PETSc element Hessian | authoritative suite + showcase parity |
| JAX+PETSc local-SFD Hessian | authoritative suite + showcase parity |

## Shared-Case Result Equivalence

The shared showcase case is level `5` at `np=1`. Only converged
implementations are included below.

| implementation | energy | rel. diff vs ref | Newton | linear | wall [s] |
| --- | --- | --- | --- | --- | --- |
| FEniCS custom | -7.942969 | 0.000 | 5 | 15 | 0.0780 |
| FEniCS SNES | -7.943000 | 0.000 | 10 | 0 | 0.0672 |
| pure JAX serial | -7.943000 | 0.000 | 6 | 0 | 0.0976 |
| JAX+PETSc element | -7.942969 | 0.000 | 6 | 17 | 0.0800 |
| JAX+PETSc local-SFD | -7.942969 | 0.000 | 6 | 17 | 0.1799 |

## Scaling And Speed Comparison

![pLaplace finest-mesh strong scaling preview](img/png/plaplace/plaplace_strong_scaling.png)

PDF: [pLaplace strong-scaling PDF](img/pdf/plaplace/plaplace_strong_scaling.pdf)

![pLaplace time-vs-mesh-size preview](img/png/plaplace/plaplace_mesh_timing.png)

PDF: [pLaplace time-vs-mesh-size PDF](img/pdf/plaplace/plaplace_mesh_timing.pdf)
- raw direct speed source: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/direct_speed.csv`
- authoritative maintained scaling source: `replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite/summary.json`

Finest maintained suite scaling (`level 9`):

| implementation | ranks | time [s] | Newton | linear | energy |
| --- | --- | --- | --- | --- | --- |
| FEniCS custom | 1 | 13.6448 | 5 | 10 | -7.960005 |
| FEniCS custom | 2 | 7.2662 | 5 | 10 | -7.960005 |
| FEniCS custom | 4 | 4.8780 | 6 | 12 | -7.960006 |
| FEniCS custom | 8 | 2.9963 | 6 | 12 | -7.960006 |
| FEniCS custom | 16 | 1.5145 | 6 | 12 | -7.960006 |
| FEniCS custom | 32 | 0.9705 | 6 | 12 | -7.960006 |
| JAX+PETSc element | 1 | 10.1229 | 6 | 11 | -7.960004 |
| JAX+PETSc element | 2 | 5.8477 | 6 | 11 | -7.960005 |
| JAX+PETSc element | 4 | 3.8280 | 6 | 11 | -7.960005 |
| JAX+PETSc element | 8 | 2.7677 | 6 | 11 | -7.960005 |
| JAX+PETSc element | 16 | 1.5238 | 6 | 11 | -7.960005 |
| JAX+PETSc element | 32 | 1.1604 | 6 | 11 | -7.960003 |
| JAX+PETSc local-SFD | 1 | 11.3635 | 6 | 11 | -7.960004 |
| JAX+PETSc local-SFD | 2 | 6.4908 | 6 | 11 | -7.960005 |
| JAX+PETSc local-SFD | 4 | 4.2553 | 6 | 11 | -7.960005 |
| JAX+PETSc local-SFD | 8 | 3.1809 | 6 | 11 | -7.960005 |
| JAX+PETSc local-SFD | 16 | 1.7922 | 6 | 11 | -7.960005 |
| JAX+PETSc local-SFD | 32 | 1.4434 | 6 | 11 | -7.960003 |

Fixed-rank mesh-size timing (`32` ranks):

| implementation | level | free DOFs | ranks | time [s] | energy |
| --- | --- | --- | --- | --- | --- |
| FEniCS custom | 5 | 3194 | 32 | 0.1384 | -7.942969 |
| FEniCS custom | 6 | 12545 | 32 | 0.1767 | -7.954564 |
| FEniCS custom | 7 | 49651 | 32 | 0.2630 | -7.958292 |
| FEniCS custom | 8 | 197555 | 32 | 0.4151 | -7.959556 |
| FEniCS custom | 9 | 788421 | 32 | 0.9705 | -7.960006 |
| JAX+PETSc element | 5 | 2945 | 32 | 0.2418 | -7.942969 |
| JAX+PETSc element | 6 | 12033 | 32 | 0.3491 | -7.954564 |
| JAX+PETSc element | 7 | 48641 | 32 | 0.4302 | -7.958292 |
| JAX+PETSc element | 8 | 195585 | 32 | 0.5230 | -7.959556 |
| JAX+PETSc element | 9 | 784385 | 32 | 1.1604 | -7.960003 |
| JAX+PETSc local-SFD | 5 | 2945 | 32 | 0.4392 | -7.942969 |
| JAX+PETSc local-SFD | 6 | 12033 | 32 | 0.5146 | -7.954564 |
| JAX+PETSc local-SFD | 7 | 48641 | 32 | 0.5686 | -7.958292 |
| JAX+PETSc local-SFD | 8 | 195585 | 32 | 0.7572 | -7.959556 |
| JAX+PETSc local-SFD | 9 | 784385 | 32 | 1.4434 | -7.960003 |

## Notes On Exclusions

- No pLaplace showcase implementation had to be excluded: all five maintained
  paths converged on the shared level-`5` serial comparison case.

## Raw Outputs And Figures

- replication suite: `replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite`
- publication reruns: `overview/img/runs/plaplace/showcase`
- curated figures: `overview/img/pdf/plaplace/` and `overview/img/png/plaplace/`

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
