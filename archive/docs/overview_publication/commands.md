# Overview Command Catalog

These commands produced the publication overview package under `overview/`.

## Publication Reruns

```bash
./.venv/bin/python overview/img/scripts/build_publication_reruns.py --resume
```

## Data Extraction

### pLaplace

```bash
./.venv/bin/python overview/img/scripts/build_plaplace_data.py
```

### GinzburgLandau

```bash
./.venv/bin/python overview/img/scripts/build_ginzburg_landau_data.py
```

### HyperElasticity

```bash
./.venv/bin/python overview/img/scripts/build_hyperelasticity_data.py
```

### Topology

```bash
./.venv/bin/python overview/img/scripts/build_topology_data.py
```

## Figure Generation

### pLaplace

```bash
./.venv/bin/python overview/img/scripts/build_plaplace_figures.py
```

### GinzburgLandau

```bash
./.venv/bin/python overview/img/scripts/build_ginzburg_landau_figures.py
```

### HyperElasticity

```bash
./.venv/bin/python overview/img/scripts/build_hyperelasticity_figures.py
```

### Topology

```bash
./.venv/bin/python overview/img/scripts/build_topology_figures.py
```

## Overview Page Generation

```bash
./.venv/bin/python overview/img/scripts/build_overview_pages.py
```

## Source Replication Campaign

- replication root: `replications/2026-03-16_maintained_refresh`
- pLaplace suite: `./.venv/bin/python -u experiments/runners/run_plaplace_final_suite.py --out-dir replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite`
- GinzburgLandau suite: `./.venv/bin/python -u experiments/runners/run_gl_final_suite.py --out-dir replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite`
- HyperElasticity MPI suite: `./.venv/bin/python -u experiments/runners/run_he_final_suite_best.py --out-dir replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best --no-seed-known-results`
- HyperElasticity pure-JAX suite: `./.venv/bin/python -u experiments/runners/run_he_pure_jax_suite_best.py --out-dir replications/2026-03-16_maintained_refresh/runs/hyperelasticity/pure_jax_suite_best`
- topology serial report: `./.venv/bin/python -u experiments/analysis/generate_report_assets.py --asset-dir replications/2026-03-16_maintained_refresh/runs/topology/serial_reference --report-path replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/report.md`
- topology parallel final report: `mpiexec -n 32 ./.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 768 --ny 384 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 32 --load_pad_cells 32 --volume_fraction_target 0.4 --theta_min 1e-6 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 10.0 --p_increment 0.2 --continuation_interval 1 --outer_maxit 2000 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --print_outer_iterations --save_outer_state_history --outer_snapshot_stride 2 --outer_snapshot_dir replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/frames --json_out replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/parallel_full_run.json --state_out replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/parallel_full_state.npz`
- topology parallel scaling report: `./.venv/bin/python -u experiments/analysis/generate_parallel_scaling_stallstop_report.py --asset-dir replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling --report-path replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/report.md`
