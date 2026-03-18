# Topology Implementation Comparison

## Maintained Implementation Roster

| implementation | comparison role |
| --- | --- |
| pure JAX serial | shared-case parity reference on `192 x 96` |
| parallel JAX+PETSc | direct-status comparison on `192 x 96`; fine-grid scaling on `768 x 384` |

## Shared-Case Result Equivalence

Only the serial pure-JAX implementation completes the maintained shared
`192 x 96` direct-comparison case. The parallel implementation is therefore
excluded from the parity table and reported in the status table below instead.

Completed shared-case results:

| implementation | ranks | status | compliance | volume | wall [s] |
| --- | --- | --- | --- | --- | --- |
| pure JAX serial | 1 | completed | 4.1557 | 0.4000 | 9.575 |

## Direct Comparison Status Table

| implementation | ranks | status | wall [s] | compliance | volume |
| --- | --- | --- | --- | --- | --- |
| parallel JAX+PETSc | 1 | max_outer_iterations | 38.399 | 29.7952 | 0.3963 |
| pure JAX serial | 1 | completed | 9.575 | 4.1557 | 0.4000 |
| parallel JAX+PETSc | 2 | max_outer_iterations | 31.180 | 29.6444 | 0.3946 |
| parallel JAX+PETSc | 4 | max_outer_iterations | 22.550 | 29.4616 | 0.3919 |

## Scaling And Speed Comparison

![Topology finest-mesh strong scaling preview](img/png/topology/topology_strong_scaling.png)

PDF: [Topology strong-scaling PDF](img/pdf/topology/topology_strong_scaling.pdf)

![Topology time-vs-mesh-size preview](img/png/topology/topology_mesh_timing.png)

PDF: [Topology time-vs-mesh-size PDF](img/pdf/topology/topology_mesh_timing.pdf)
- raw direct comparison source: `replications/2026-03-16_maintained_refresh/comparisons/topology/direct_speed.csv`
- maintained fine-grid scaling source: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/scaling_summary.csv`
- maintained fixed-rank mesh-size sweep source: `overview/img/runs/topology/mesh_scaling`

Fine-grid parallel scaling (`768 x 384`):

| ranks | result | outer | p | volume | compliance | wall [s] | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | completed | 65 | 4.20 | 0.3884 | 9.1559 | 202.373 | 1.000 |
| 2 | completed | 72 | 5.60 | 0.3932 | 8.9473 | 137.676 | 1.470 |
| 4 | completed | 69 | 4.60 | 0.3850 | 9.1684 | 95.117 | 2.128 |
| 8 | completed | 67 | 7.00 | 0.3932 | 9.2178 | 82.163 | 2.463 |
| 16 | completed | 66 | 5.80 | 0.3798 | 9.6859 | 44.123 | 4.587 |
| 32 | completed | 66 | 6.60 | 0.3749 | 9.9074 | 20.366 | 9.937 |

Fixed-rank mesh-size timing (`8` ranks):

| mesh | free DOFs | ranks | result | outer | compliance | volume | wall [s] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 192x96 | 54763 | 8 | completed | 237 | 11.4941 | 0.4106 | 20.068 |
| 384x192 | 218007 | 8 | completed | 73 | 9.2296 | 0.3870 | 23.282 |
| 768x384 | 870001 | 8 | completed | 67 | 9.2178 | 0.3932 | 82.879 |

## Notes On Exclusions

- The parallel JAX+PETSc path is excluded from the shared-case parity table
  because the maintained `192 x 96` direct-comparison runs terminate at
  `max_outer_iterations` for ranks `1`, `2`, and `4`.
- The parallel path is still the maintained fine-grid benchmark and scaling
  implementation; its validated `768 x 384` results are reported here
  separately from the serial reference parity case.

## Raw Outputs And Figures

- serial reference: `replications/2026-03-16_maintained_refresh/runs/topology/serial_reference`
- parallel final: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_final`
- parallel scaling: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling`
- curated figures: `overview/img/pdf/topology/` and `overview/img/png/topology/`

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
