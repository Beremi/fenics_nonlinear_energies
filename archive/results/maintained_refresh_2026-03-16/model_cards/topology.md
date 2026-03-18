# Topology Optimisation Model Card

## Mathematical Model

Compliance-constrained cantilever topology optimisation with a density field, phase-field regularisation, proximal move penalty, and staircase SIMP continuation.

## Geometry And Setup

Cantilever beam with clamped left support, right-edge traction patch, target volume fraction 0.4.

## Discretization And Mesh Source

Structured triangular displacement/design meshes with separate free-DOF graphs for mechanics and design.

## Maintained Implementations

| id | implementation | canonical command |
| --- | --- | --- |
| jax_serial | pure JAX serial reference | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_jax.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 0.001 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --mechanics_maxit 200 --design_maxit 400 --tolf 1e-06 --tolg 0.001 --ksp_rtol 0.01 --ksp_max_it 80 --save_outer_state_history --quiet --json_out replications/2026-03-16_maintained_refresh/runs/examples/topology_serial_reference/report_run.json --state_out replications/2026-03-16_maintained_refresh/runs/examples/topology_serial_reference/report_state.npz |
| jax_parallel | parallel JAX+PETSc | /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/runs/examples/topology_parallel_reference/report_run.json --state_out replications/2026-03-16_maintained_refresh/runs/examples/topology_parallel_reference/report_state.npz |

## Sample Outputs

### pure JAX serial reference

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/topology_serial_reference`

```json
{
  "case_id": null,
  "design_free_dofs": 16205,
  "displacement_free_dofs": 37248,
  "family": "topology",
  "final_compliance": 4.155705577015529,
  "final_p_penal": 4.0,
  "final_volume_fraction": 0.40000000000006014,
  "implementation": "jax_serial",
  "nprocs": 1,
  "nx": 192,
  "ny": 96,
  "outer_iterations": 121,
  "result": "completed",
  "setup_time_s": 1.6043147649616003,
  "total_design_iters": 0,
  "total_linear_iters": 0,
  "total_newton_iters": 0,
  "wall_time_s": 9.730779257020913
}
```

### parallel JAX+PETSc

Command leaf: `replications/2026-03-16_maintained_refresh/runs/examples/topology_parallel_reference`

```json
{
  "case_id": null,
  "design_free_dofs": 16171,
  "displacement_free_dofs": 37248,
  "family": "topology",
  "final_compliance": 29.79524941155877,
  "final_p_penal": 2.0,
  "final_volume_fraction": 0.3962743681922173,
  "implementation": "jax_parallel",
  "nprocs": 1,
  "nx": 192,
  "ny": 96,
  "outer_iterations": 180,
  "result": "max_outer_iterations",
  "setup_time_s": 0.9975787110161036,
  "total_design_iters": 2946,
  "total_linear_iters": 3245,
  "total_newton_iters": 0,
  "wall_time_s": 38.27372991293669
}
```

## Replicated Outputs

- Maintained suite: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling`
- Direct speed comparison: `comparisons/topology/direct_speed.md`
- Topology scaling links: `comparisons/topology/scaling_links.md`
- Example runs: `runs/examples/`

## Caveats And Issues

- No campaign-specific issues were recorded.
