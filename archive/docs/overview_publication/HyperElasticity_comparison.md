# HyperElasticity Implementation Comparison

## Maintained Implementation Roster

| implementation | comparison role |
| --- | --- |
| FEniCS custom trust-region Newton | authoritative suite + showcase parity |
| FEniCS SNES | excluded from parity; fails on the showcase case |
| JAX+PETSc element Hessian | authoritative suite + showcase parity + fine-mesh scaling |
| pure JAX serial | authoritative serial reference + showcase parity only |

## Shared-Case Result Equivalence

The shared showcase case is level `1` with `24` load steps at `np=1`. Only
working implementations are included in the parity table.

| implementation | completed steps | energy | rel. diff vs ref | Newton | linear | wall [s] |
| --- | --- | --- | --- | --- | --- | --- |
| FEniCS custom | 24 | 197.755 | 0.000 | 798 | 0 | 7.959 |
| JAX+PETSc element | 24 | 197.755 | 0.000 | 793 | 10595 | 8.541 |
| pure JAX serial | 24 | 197.750 | 0.000 | 559 | 0 | 28.936 |

## Scaling And Speed Comparison

![HyperElasticity finest-mesh strong scaling preview](img/png/hyperelasticity/hyperelasticity_strong_scaling.png)

PDF: [HyperElasticity strong-scaling PDF](img/pdf/hyperelasticity/hyperelasticity_strong_scaling.pdf)

![HyperElasticity time-vs-mesh-size preview](img/png/hyperelasticity/hyperelasticity_mesh_timing.png)

PDF: [HyperElasticity time-vs-mesh-size PDF](img/pdf/hyperelasticity/hyperelasticity_mesh_timing.pdf)
- raw direct speed source: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/direct_speed.csv`
- authoritative maintained scaling source: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best/summary.json`

Finest maintained strong scaling (`level 4`, `24` steps):

| implementation | ranks | time [s] | Newton | linear | energy |
| --- | --- | --- | --- | --- | --- |
| FEniCS custom | 8 | 1229.146 | 701 | 16157 | 87.723 |
| FEniCS custom | 16 | 601.867 | 685 | 15896 | 87.723 |
| FEniCS custom | 32 | 334.199 | 726 | 16003 | 87.723 |
| JAX+PETSc element | 8 | 1886.441 | 814 | 18824 | 87.722 |
| JAX+PETSc element | 16 | 907.637 | 792 | 18836 | 87.722 |
| JAX+PETSc element | 32 | 486.783 | 820 | 18880 | 87.722 |

Fixed-rank mesh-size timing (`32` ranks, `24` steps):

| implementation | level | total DOFs | ranks | time [s] | energy |
| --- | --- | --- | --- | --- | --- |
| FEniCS custom | 1 | 2187 | 32 | 10.913 | 197.773 |
| FEniCS custom | 2 | 12075 | 32 | 14.619 | 116.326 |
| FEniCS custom | 3 | 78003 | 32 | 39.575 | 93.705 |
| FEniCS custom | 4 | 555747 | 32 | 334.199 | 87.723 |
| JAX+PETSc element | 1 | 2187 | 32 | 11.806 | 197.755 |
| JAX+PETSc element | 2 | 12075 | 32 | 15.613 | 116.333 |
| JAX+PETSc element | 3 | 78003 | 32 | 45.313 | 93.705 |
| JAX+PETSc element | 4 | 555747 | 32 | 486.783 | 87.722 |

## Notes On Exclusions

- FEniCS SNES is excluded from the parity table because it fails on the
  shared showcase case in the maintained direct comparison data.
- Pure JAX is intentionally excluded from the scaling figures: the maintained
  pure-JAX path is single-process only and a new `level 4` serial rerun would
  add a large extra cost without changing the distributed comparison.

## Raw Outputs And Figures

- MPI suite: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best`
- pure-JAX suite: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/pure_jax_suite_best`
- publication reruns: `overview/img/runs/hyperelasticity/showcase`
- curated figures: `overview/img/pdf/hyperelasticity/` and `overview/img/png/hyperelasticity/`

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
