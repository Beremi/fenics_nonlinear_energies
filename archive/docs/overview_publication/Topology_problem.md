# Topology Problem Overview

## Mathematical Model

The maintained topology workflow solves a reduced compliance-minimisation
problem with a density field, phase-field regularisation, a proximal move
penalty, and staircase SIMP continuation. The mechanics step solves the
equilibrium problem for a fixed density field, then the design step minimises a
scalar FE energy with frozen mechanics data. In shorthand, the design update
minimises

$$
\mathcal{J}(\theta, z)
= C(\theta, u)
+ \lambda_{V} \int_\Omega \theta\,dx
+ \alpha \int_\Omega \left(\frac{\ell}{2}\lvert \nabla \theta \rvert^2
+ \frac{1}{\ell}W(\theta)\right) dx
+ \frac{\mu}{2}\| z-z_{\mathrm{old}} \|^2,
$$

with a target volume fraction of `0.4` and maintained continuation in the SIMP
penalisation parameter $p$. This keeps the mechanics subproblem in the same
energy-minimisation setting as the repository's elasticity solvers while the
design update looks like a data-driven Ginzburg-Landau problem on a fixed
scalar graph.

## Geometry, Boundary Conditions, And Setup

- geometry: 2D cantilever beam of length `2.0` and height `1.0`
- left boundary: clamped support
- right-edge traction patch: `load_fraction = 0.2`
- maintained reference meshes:
  - serial reference: `192 x 96`
  - parallel fine benchmark / scaling: `768 x 384`
- maintained active path: distributed JAX+PETSc topology solve
- maintained reference path: pure-JAX serial solve retained for smaller
  comparison and formulation sanity checks

## Discretization And Mesh Source

The maintained topology path uses structured triangular displacement/design
meshes with separate free-DOF layouts for mechanics and design. The mechanics
phase uses a vector elasticity operator with PETSc `fgmres + gamg`, rigid-body
near-nullspace enrichment, and fixed-rank distributed assembly. The design
phase uses a distributed gradient-based solve on the same mesh family with a
fixed continuation schedule in $p$.

## Maintained Implementations

| implementation | role in overview |
| --- | --- |
| pure JAX serial | serial reference benchmark and showcase state |
| parallel JAX+PETSc | fine-grid final benchmark and scaling study |

## Showcase Sample Result

The finished parallel fine-grid campaign now provides the publication density
figure, the objective-history figure, and the curated density-evolution
animation, while the serial reference remains the smaller direct-comparison
baseline. The finished parallel final state shown below is the maintained
`768 x 384`, `32`-rank run rather than a partially converged demonstration.

![Topology final density preview](img/png/topology/topology_final_density.png)

PDF: [Final density PDF](img/pdf/topology/topology_final_density.pdf)

![Topology objective-history preview](img/png/topology/topology_objective_history.png)

PDF: [Objective-history PDF](img/pdf/topology/topology_objective_history.pdf)

![Topology parallel final density evolution](img/gif/topology/topology_parallel_final_evolution.gif)

## Resolution / Objective Table

| label | mesh | ranks | result | outer | compliance | volume | wall [s] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| serial_reference | 192x96 | 1 | completed | 121 | 4.1557 | 0.4000 | 10.377 |
| parallel_final | 768x384 | 32 | completed | 66 | 9.9074 | 0.3749 | 21.208 |
| parallel_scaling_r1 | 768x384 | 1 | completed | 65 | 9.1559 | 0.3884 | 202.373 |
| parallel_scaling_r2 | 768x384 | 2 | completed | 72 | 8.9473 | 0.3932 | 137.676 |
| parallel_scaling_r4 | 768x384 | 4 | completed | 69 | 9.1684 | 0.3850 | 95.117 |
| parallel_scaling_r8 | 768x384 | 8 | completed | 67 | 9.2178 | 0.3932 | 82.163 |
| parallel_scaling_r16 | 768x384 | 16 | completed | 66 | 9.6859 | 0.3798 | 44.123 |
| parallel_scaling_r32 | 768x384 | 32 | completed | 66 | 9.9074 | 0.3749 | 20.366 |

## Caveats And Repaired Issues

- No new solver/path issues were discovered during the maintained replication refresh.

## Commands Used

```bash
./.venv/bin/python -u experiments/analysis/generate_report_assets.py --asset-dir replications/2026-03-16_maintained_refresh/runs/topology/serial_reference --report-path replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/report.md
```

```bash
mpiexec -n 32 ./.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 768 --ny 384 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 32 --load_pad_cells 32 --volume_fraction_target 0.4 --theta_min 1e-6 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 10.0 --p_increment 0.2 --continuation_interval 1 --outer_maxit 2000 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --print_outer_iterations --save_outer_state_history --outer_snapshot_stride 2 --outer_snapshot_dir replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/frames --json_out replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/parallel_full_run.json --state_out replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/parallel_full_state.npz
```

```bash
./.venv/bin/python -u experiments/analysis/generate_parallel_scaling_stallstop_report.py --asset-dir replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling --report-path replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/report.md
```

```bash
mpiexec -n 8 ./.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 8 --load_pad_cells 8 --volume_fraction_target 0.4 --theta_min 1e-6 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 10.0 --p_increment 0.2 --continuation_interval 1 --outer_maxit 2000 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --json_out overview/img/runs/topology/mesh_scaling/nx192_ny96_np8/output.json
```

```bash
mpiexec -n 8 ./.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 384 --ny 192 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-6 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 10.0 --p_increment 0.2 --continuation_interval 1 --outer_maxit 2000 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --json_out overview/img/runs/topology/mesh_scaling/nx384_ny192_np8/output.json
```

```bash
mpiexec -n 8 ./.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 768 --ny 384 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 32 --load_pad_cells 32 --volume_fraction_target 0.4 --theta_min 1e-6 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 10.0 --p_increment 0.2 --continuation_interval 1 --outer_maxit 2000 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --json_out overview/img/runs/topology/mesh_scaling/nx768_ny384_np8/output.json
```

```bash
./.venv/bin/python overview/img/scripts/build_topology_data.py
```

```bash
./.venv/bin/python overview/img/scripts/build_topology_figures.py
```
