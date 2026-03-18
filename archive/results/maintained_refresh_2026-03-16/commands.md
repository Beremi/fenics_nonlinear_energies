# Commands

## Examples

### gl_fenics_custom

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/gl_fenics_custom/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/runs/examples/gl_fenics_custom/output.json
```

### gl_fenics_snes

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/gl_fenics_snes/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/runs/examples/gl_fenics_snes/output.json
```

### gl_jax_petsc_element

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/gl_jax_petsc_element/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/runs/examples/gl_jax_petsc_element/output.json
```

### he_fenics_custom

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_custom/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_custom/output.json
```

### he_fenics_snes

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_snes/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_snes/output.json
```

### he_jax_petsc_element

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/he_jax_petsc_element/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/runs/examples/he_jax_petsc_element/output.json
```

### he_jax_serial

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax/solve_HE_jax_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/he_jax_serial/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py --level 1 --steps 24 --total_steps 24 --quiet --out replications/2026-03-16_maintained_refresh/runs/examples/he_jax_serial/output.json
```

### plaplace_fenics_custom

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_custom/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_custom/output.json
```

### plaplace_fenics_snes

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_snes/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_snes/output.json
```

### plaplace_jax_petsc_element

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_petsc_element/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_petsc_element/output.json
```

### plaplace_jax_serial

- family: `plaplace`
- source: `src/problems/plaplace/jax/solve_pLaplace_jax_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_serial/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax/solve_pLaplace_jax_newton.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_serial/output.json
```

### topology_parallel_reference

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/topology_parallel_reference/report_run.json, replications/2026-03-16_maintained_refresh/runs/examples/topology_parallel_reference/report_state.npz`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/runs/examples/topology_parallel_reference/report_run.json --state_out replications/2026-03-16_maintained_refresh/runs/examples/topology_parallel_reference/report_state.npz
```

### topology_serial_reference

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_jax.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/examples/topology_serial_reference/report_run.json, replications/2026-03-16_maintained_refresh/runs/examples/topology_serial_reference/report_state.npz`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_jax.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 0.001 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --mechanics_maxit 200 --design_maxit 400 --tolf 1e-06 --tolg 0.001 --ksp_rtol 0.01 --ksp_max_it 80 --save_outer_state_history --quiet --json_out replications/2026-03-16_maintained_refresh/runs/examples/topology_serial_reference/report_run.json --state_out replications/2026-03-16_maintained_refresh/runs/examples/topology_serial_reference/report_state.npz
```

## Speed

### gl_l5_np1_fenics_custom_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_custom/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_custom/run01/output.json
```

### gl_l5_np1_fenics_custom_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_custom/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_custom/run02/output.json
```

### gl_l5_np1_fenics_custom_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_custom/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_custom/run03/output.json
```

### gl_l5_np1_fenics_snes_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_snes/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_snes/run01/output.json
```

### gl_l5_np1_fenics_snes_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_snes/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_snes/run02/output.json
```

### gl_l5_np1_fenics_snes_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_snes/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/fenics_snes/run03/output.json
```

### gl_l5_np1_jax_petsc_element_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_element/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_element/run01/output.json
```

### gl_l5_np1_jax_petsc_element_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_element/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_element/run02/output.json
```

### gl_l5_np1_jax_petsc_element_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_element/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_element/run03/output.json
```

### gl_l5_np1_jax_petsc_local_sfd_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_local_sfd/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_local_sfd/run01/output.json
```

### gl_l5_np1_jax_petsc_local_sfd_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_local_sfd/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_local_sfd/run02/output.json
```

### gl_l5_np1_jax_petsc_local_sfd_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_local_sfd/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np1/jax_petsc_local_sfd/run03/output.json
```

### gl_l5_np2_fenics_custom_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_custom/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_custom/run01/output.json
```

### gl_l5_np2_fenics_custom_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_custom/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_custom/run02/output.json
```

### gl_l5_np2_fenics_custom_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_custom/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_custom/run03/output.json
```

### gl_l5_np2_fenics_snes_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_snes/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_snes/run01/output.json
```

### gl_l5_np2_fenics_snes_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_snes/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_snes/run02/output.json
```

### gl_l5_np2_fenics_snes_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_snes/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/fenics_snes/run03/output.json
```

### gl_l5_np2_jax_petsc_element_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_element/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_element/run01/output.json
```

### gl_l5_np2_jax_petsc_element_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_element/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_element/run02/output.json
```

### gl_l5_np2_jax_petsc_element_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_element/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_element/run03/output.json
```

### gl_l5_np2_jax_petsc_local_sfd_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_local_sfd/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_local_sfd/run01/output.json
```

### gl_l5_np2_jax_petsc_local_sfd_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_local_sfd/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_local_sfd/run02/output.json
```

### gl_l5_np2_jax_petsc_local_sfd_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_local_sfd/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np2/jax_petsc_local_sfd/run03/output.json
```

### gl_l5_np4_fenics_custom_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_custom/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_custom/run01/output.json
```

### gl_l5_np4_fenics_custom_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_custom/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_custom/run02/output.json
```

### gl_l5_np4_fenics_custom_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_custom/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_custom/run03/output.json
```

### gl_l5_np4_fenics_snes_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_snes/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_snes/run01/output.json
```

### gl_l5_np4_fenics_snes_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_snes/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_snes/run02/output.json
```

### gl_l5_np4_fenics_snes_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_snes/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/fenics_snes/run03/output.json
```

### gl_l5_np4_jax_petsc_element_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_element/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_element/run01/output.json
```

### gl_l5_np4_jax_petsc_element_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_element/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_element/run02/output.json
```

### gl_l5_np4_jax_petsc_element_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_element/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode element --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_element/run03/output.json
```

### gl_l5_np4_jax_petsc_local_sfd_run01

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_local_sfd/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_local_sfd/run01/output.json
```

### gl_l5_np4_jax_petsc_local_sfd_run02

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_local_sfd/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_local_sfd/run02/output.json
```

### gl_l5_np4_jax_petsc_local_sfd_run03

- family: `ginzburg_landau`
- source: `src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_local_sfd/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --profile reference --assembly_mode element --local_hessian_mode sfd_local --element_reorder_mode block_xyz --local_coloring --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/ginzburg_landau/raw/l5_np4/jax_petsc_local_sfd/run03/output.json
```

### he_l1_steps24_np1_fenics_custom_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_custom/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_custom/run01/output.json
```

### he_l1_steps24_np1_fenics_custom_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_custom/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_custom/run02/output.json
```

### he_l1_steps24_np1_fenics_custom_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_custom/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_custom/run03/output.json
```

### he_l1_steps24_np1_fenics_snes_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_snes/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_snes/run01/output.json
```

### he_l1_steps24_np1_fenics_snes_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_snes/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_snes/run02/output.json
```

### he_l1_steps24_np1_fenics_snes_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_snes/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/fenics_snes/run03/output.json
```

### he_l1_steps24_np1_jax_petsc_element_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_petsc_element/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_petsc_element/run01/output.json
```

### he_l1_steps24_np1_jax_petsc_element_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_petsc_element/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_petsc_element/run02/output.json
```

### he_l1_steps24_np1_jax_petsc_element_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_petsc_element/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_petsc_element/run03/output.json
```

### he_l1_steps24_np1_jax_serial_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax/solve_HE_jax_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_serial/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py --level 1 --steps 24 --total_steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_serial/run01/output.json
```

### he_l1_steps24_np1_jax_serial_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax/solve_HE_jax_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_serial/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py --level 1 --steps 24 --total_steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_serial/run02/output.json
```

### he_l1_steps24_np1_jax_serial_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax/solve_HE_jax_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_serial/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py --level 1 --steps 24 --total_steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np1/jax_serial/run03/output.json
```

### he_l1_steps24_np2_fenics_custom_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_custom/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_custom/run01/output.json
```

### he_l1_steps24_np2_fenics_custom_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_custom/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_custom/run02/output.json
```

### he_l1_steps24_np2_fenics_custom_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_custom/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_custom/run03/output.json
```

### he_l1_steps24_np2_fenics_snes_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_snes/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_snes/run01/output.json
```

### he_l1_steps24_np2_fenics_snes_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_snes/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_snes/run02/output.json
```

### he_l1_steps24_np2_fenics_snes_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_snes/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/fenics_snes/run03/output.json
```

### he_l1_steps24_np2_jax_petsc_element_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/jax_petsc_element/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/jax_petsc_element/run01/output.json
```

### he_l1_steps24_np2_jax_petsc_element_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/jax_petsc_element/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/jax_petsc_element/run02/output.json
```

### he_l1_steps24_np2_jax_petsc_element_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/jax_petsc_element/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np2/jax_petsc_element/run03/output.json
```

### he_l1_steps24_np4_fenics_custom_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_custom/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_custom/run01/output.json
```

### he_l1_steps24_np4_fenics_custom_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_custom/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_custom/run02/output.json
```

### he_l1_steps24_np4_fenics_custom_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_custom/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total-steps 24 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_custom/run03/output.json
```

### he_l1_steps24_np4_fenics_snes_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_snes/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_snes/run01/output.json
```

### he_l1_steps24_np4_fenics_snes_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_snes/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_snes/run02/output.json
```

### he_l1_steps24_np4_fenics_snes_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_snes/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/fenics/solve_HE_snes_newton.py --level 1 --steps 24 --total_steps 24 --ksp_type gmres --pc_type hypre --ksp_rtol 1e-1 --ksp_max_it 500 --snes_atol 1e-3 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/fenics_snes/run03/output.json
```

### he_l1_steps24_np4_jax_petsc_element_run01

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/jax_petsc_element/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/jax_petsc_element/run01/output.json
```

### he_l1_steps24_np4_jax_petsc_element_run02

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/jax_petsc_element/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/jax_petsc_element/run02/output.json
```

### he_l1_steps24_np4_jax_petsc_element_run03

- family: `hyperelasticity`
- source: `src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/jax_petsc_element/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --profile performance --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz --local_hessian_mode element --local_coloring --use_trust_region --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 --trust_eta_expand 0.75 --nproc 1 --quiet --out replications/2026-03-16_maintained_refresh/comparisons/hyperelasticity/raw/l1_steps24_np4/jax_petsc_element/run03/output.json
```

### plaplace_l5_np1_fenics_custom_run01

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_custom/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_custom/run01/output.json
```

### plaplace_l5_np1_fenics_custom_run02

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_custom/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_custom/run02/output.json
```

### plaplace_l5_np1_fenics_custom_run03

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_custom/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_custom/run03/output.json
```

### plaplace_l5_np1_fenics_snes_run01

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_snes/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_snes/run01/output.json
```

### plaplace_l5_np1_fenics_snes_run02

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_snes/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_snes/run02/output.json
```

### plaplace_l5_np1_fenics_snes_run03

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_snes/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/fenics_snes/run03/output.json
```

### plaplace_l5_np1_jax_petsc_element_run01

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_element/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_element/run01/output.json
```

### plaplace_l5_np1_jax_petsc_element_run02

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_element/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_element/run02/output.json
```

### plaplace_l5_np1_jax_petsc_element_run03

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_element/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_element/run03/output.json
```

### plaplace_l5_np1_jax_petsc_local_sfd_run01

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_local_sfd/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_local_sfd/run01/output.json
```

### plaplace_l5_np1_jax_petsc_local_sfd_run02

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_local_sfd/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_local_sfd/run02/output.json
```

### plaplace_l5_np1_jax_petsc_local_sfd_run03

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_local_sfd/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_petsc_local_sfd/run03/output.json
```

### plaplace_l5_np1_jax_serial_run01

- family: `plaplace`
- source: `src/problems/plaplace/jax/solve_pLaplace_jax_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_serial/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax/solve_pLaplace_jax_newton.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_serial/run01/output.json
```

### plaplace_l5_np1_jax_serial_run02

- family: `plaplace`
- source: `src/problems/plaplace/jax/solve_pLaplace_jax_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_serial/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax/solve_pLaplace_jax_newton.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_serial/run02/output.json
```

### plaplace_l5_np1_jax_serial_run03

- family: `plaplace`
- source: `src/problems/plaplace/jax/solve_pLaplace_jax_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_serial/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax/solve_pLaplace_jax_newton.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np1/jax_serial/run03/output.json
```

### plaplace_l5_np2_fenics_custom_run01

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_custom/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_custom/run01/output.json
```

### plaplace_l5_np2_fenics_custom_run02

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_custom/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_custom/run02/output.json
```

### plaplace_l5_np2_fenics_custom_run03

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_custom/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_custom/run03/output.json
```

### plaplace_l5_np2_fenics_snes_run01

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_snes/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_snes/run01/output.json
```

### plaplace_l5_np2_fenics_snes_run02

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_snes/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_snes/run02/output.json
```

### plaplace_l5_np2_fenics_snes_run03

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_snes/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/fenics_snes/run03/output.json
```

### plaplace_l5_np2_jax_petsc_element_run01

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_element/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_element/run01/output.json
```

### plaplace_l5_np2_jax_petsc_element_run02

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_element/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_element/run02/output.json
```

### plaplace_l5_np2_jax_petsc_element_run03

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_element/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_element/run03/output.json
```

### plaplace_l5_np2_jax_petsc_local_sfd_run01

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_local_sfd/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_local_sfd/run01/output.json
```

### plaplace_l5_np2_jax_petsc_local_sfd_run02

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_local_sfd/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_local_sfd/run02/output.json
```

### plaplace_l5_np2_jax_petsc_local_sfd_run03

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_local_sfd/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np2/jax_petsc_local_sfd/run03/output.json
```

### plaplace_l5_np4_fenics_custom_run01

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_custom/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_custom/run01/output.json
```

### plaplace_l5_np4_fenics_custom_run02

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_custom/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_custom/run02/output.json
```

### plaplace_l5_np4_fenics_custom_run03

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_custom/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_custom/run03/output.json
```

### plaplace_l5_np4_fenics_snes_run01

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_snes/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_snes/run01/output.json
```

### plaplace_l5_np4_fenics_snes_run02

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_snes/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_snes/run02/output.json
```

### plaplace_l5_np4_fenics_snes_run03

- family: `plaplace`
- source: `src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_snes/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py --levels 5 --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/fenics_snes/run03/output.json
```

### plaplace_l5_np4_jax_petsc_element_run01

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_element/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_element/run01/output.json
```

### plaplace_l5_np4_jax_petsc_element_run02

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_element/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_element/run02/output.json
```

### plaplace_l5_np4_jax_petsc_element_run03

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_element/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode element --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_element/run03/output.json
```

### plaplace_l5_np4_jax_petsc_local_sfd_run01

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_local_sfd/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_local_sfd/run01/output.json
```

### plaplace_l5_np4_jax_petsc_local_sfd_run02

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_local_sfd/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_local_sfd/run02/output.json
```

### plaplace_l5_np4_jax_petsc_local_sfd_run03

- family: `plaplace`
- source: `src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_local_sfd/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --profile reference --assembly-mode element --local-hessian-mode sfd_local --element-reorder-mode block_xyz --local-coloring --nproc 1 --quiet --json replications/2026-03-16_maintained_refresh/comparisons/plaplace/raw/l5_np4/jax_petsc_local_sfd/run03/output.json
```

### topology_nx192_ny96_np1_jax_parallel_run01

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_parallel/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_parallel/run01/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_parallel/run01/state.npz
```

### topology_nx192_ny96_np1_jax_parallel_run02

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_parallel/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_parallel/run02/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_parallel/run02/state.npz
```

### topology_nx192_ny96_np1_jax_parallel_run03

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_parallel/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_parallel/run03/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_parallel/run03/state.npz
```

### topology_nx192_ny96_np1_jax_serial_run01

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_jax.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_serial/run01/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_jax.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 0.001 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --mechanics_maxit 200 --design_maxit 400 --tolf 1e-06 --tolg 0.001 --ksp_rtol 0.01 --ksp_max_it 80 --save_outer_state_history --quiet --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_serial/run01/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_serial/run01/state.npz
```

### topology_nx192_ny96_np1_jax_serial_run02

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_jax.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_serial/run02/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_jax.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 0.001 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --mechanics_maxit 200 --design_maxit 400 --tolf 1e-06 --tolg 0.001 --ksp_rtol 0.01 --ksp_max_it 80 --save_outer_state_history --quiet --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_serial/run02/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_serial/run02/state.npz
```

### topology_nx192_ny96_np1_jax_serial_run03

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_jax.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_serial/run03/output.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_jax.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 0.001 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --mechanics_maxit 200 --design_maxit 400 --tolf 1e-06 --tolg 0.001 --ksp_rtol 0.01 --ksp_max_it 80 --save_outer_state_history --quiet --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_serial/run03/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np1/jax_serial/run03/state.npz
```

### topology_nx192_ny96_np2_jax_parallel_run01

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np2/jax_parallel/run01/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np2/jax_parallel/run01/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np2/jax_parallel/run01/state.npz
```

### topology_nx192_ny96_np2_jax_parallel_run02

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np2/jax_parallel/run02/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np2/jax_parallel/run02/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np2/jax_parallel/run02/state.npz
```

### topology_nx192_ny96_np2_jax_parallel_run03

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np2/jax_parallel/run03/output.json`

```bash
/usr/bin/mpiexec -n 2 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np2/jax_parallel/run03/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np2/jax_parallel/run03/state.npz
```

### topology_nx192_ny96_np4_jax_parallel_run01

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np4/jax_parallel/run01/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np4/jax_parallel/run01/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np4/jax_parallel/run01/state.npz
```

### topology_nx192_ny96_np4_jax_parallel_run02

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np4/jax_parallel/run02/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np4/jax_parallel/run02/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np4/jax_parallel/run02/state.npz
```

### topology_nx192_ny96_np4_jax_parallel_run03

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np4/jax_parallel/run03/output.json`

```bash
/usr/bin/mpiexec -n 4 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 192 --ny 96 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 16 --load_pad_cells 16 --volume_fraction_target 0.4 --theta_min 1e-3 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 4.0 --p_increment 0.5 --continuation_interval 20 --outer_maxit 180 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --save_outer_state_history --json_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np4/jax_parallel/run03/output.json --state_out replications/2026-03-16_maintained_refresh/comparisons/topology/raw/nx192_ny96_np4/jax_parallel/run03/state.npz
```

## Suites

### gl_final_figures

- family: `ginzburg_landau`
- source: `experiments/analysis/generate_gl_final_report_figures.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_report_figures/gl_scaling.png, replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_report_figures/gl_dof_runtime_np8.png, replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_report_figures/gl_convergence_l9_np32.png`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/analysis/generate_gl_final_report_figures.py --summary-json replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite/summary.json --asset-dir replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_report_figures
```

### gl_final_suite

- family: `ginzburg_landau`
- source: `experiments/runners/run_gl_final_suite.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite/summary.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/runners/run_gl_final_suite.py --out-dir replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite
```

### he_final_figures

- family: `hyperelasticity`
- source: `experiments/analysis/generate_he_final_report_figures.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_report_figures/he_scaling_24.png, replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_report_figures/he_scaling_96.png, replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_report_figures/he_dof_runtime_np8_24.png, replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_report_figures/he_dof_runtime_np8_96.png`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/analysis/generate_he_final_report_figures.py --summary-json replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best/summary.json --asset-dir replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_report_figures
```

### he_final_suite_best

- family: `hyperelasticity`
- source: `experiments/runners/run_he_final_suite_best.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best/summary.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/runners/run_he_final_suite_best.py --out-dir replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best --no-seed-known-results
```

### he_pure_jax_suite_best

- family: `hyperelasticity`
- source: `experiments/runners/run_he_pure_jax_suite_best.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/pure_jax_suite_best/summary.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/runners/run_he_pure_jax_suite_best.py --out-dir replications/2026-03-16_maintained_refresh/runs/hyperelasticity/pure_jax_suite_best
```

### plaplace_final_figures

- family: `plaplace`
- source: `experiments/analysis/generate_plaplace_final_report_figures.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/plaplace/final_report_figures/plaplace_scaling.png, replications/2026-03-16_maintained_refresh/runs/plaplace/final_report_figures/plaplace_dof_runtime_np8.png, replications/2026-03-16_maintained_refresh/runs/plaplace/final_report_figures/plaplace_convergence_l9_np32.png`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/analysis/generate_plaplace_final_report_figures.py --summary-json replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite/summary.json --asset-dir replications/2026-03-16_maintained_refresh/runs/plaplace/final_report_figures
```

### plaplace_final_suite

- family: `plaplace`
- source: `experiments/runners/run_plaplace_final_suite.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite/summary.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/runners/run_plaplace_final_suite.py --out-dir replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite
```

### topology_parallel_final_report

- family: `topology`
- source: `experiments/analysis/generate_parallel_full_report.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/parallel_full_outer_history.csv, replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/final_state.png, replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/convergence_history.png, replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/density_step_history.png, replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/density_evolution.gif, replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/report.md`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/analysis/generate_parallel_full_report.py --asset_dir replications/2026-03-16_maintained_refresh/runs/topology/parallel_final --report_path replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/report.md
```

### topology_parallel_final_solver

- family: `topology`
- source: `src/problems/topology/jax/solve_topopt_parallel.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/parallel_full_run.json, replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/parallel_full_state.npz`

```bash
/usr/bin/mpiexec -n 32 /home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py --nx 768 --ny 384 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 32 --load_pad_cells 32 --volume_fraction_target 0.4 --theta_min 1e-6 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 10.0 --p_increment 0.2 --continuation_interval 1 --outer_maxit 2000 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --quiet --print_outer_iterations --save_outer_state_history --outer_snapshot_stride 2 --outer_snapshot_dir replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/frames --json_out replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/parallel_full_run.json --state_out replications/2026-03-16_maintained_refresh/runs/topology/parallel_final/parallel_full_state.npz
```

### topology_parallel_scaling

- family: `topology`
- source: `experiments/analysis/generate_parallel_scaling_stallstop_report.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/scaling_summary.csv, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/wall_scaling.png, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/phase_scaling.png, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/efficiency.png, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/quality_vs_ranks.png, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/report.md, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/run_r01.json, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/run_r02.json, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/run_r04.json, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/run_r08.json, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/run_r16.json, replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/run_r32.json`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/analysis/generate_parallel_scaling_stallstop_report.py --asset-dir replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling --report-path replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling/report.md
```

### topology_serial_report

- family: `topology`
- source: `experiments/analysis/generate_report_assets.py`
- outputs: `replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/report_run.json, replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/report_state.npz, replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/report_outer_history.csv, replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/mesh_preview.png, replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/final_state.png, replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/convergence_history.png, replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/density_evolution.gif, replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/report.md`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/analysis/generate_report_assets.py --asset-dir replications/2026-03-16_maintained_refresh/runs/topology/serial_reference --report-path replications/2026-03-16_maintained_refresh/runs/topology/serial_reference/report.md
```

## Reports

### generate_reports

- family: `campaign`
- source: `experiments/analysis/generate_replication_reports.py`
- outputs: `replications/2026-03-16_maintained_refresh/index.md, replications/2026-03-16_maintained_refresh/commands.md, replications/2026-03-16_maintained_refresh/model_cards/plaplace.md, replications/2026-03-16_maintained_refresh/model_cards/ginzburg_landau.md, replications/2026-03-16_maintained_refresh/model_cards/hyperelasticity.md, replications/2026-03-16_maintained_refresh/model_cards/topology.md`

```bash
/home/michal/repos/fenics_nonlinear_energies/.venv/bin/python -u experiments/analysis/generate_replication_reports.py --out-dir replications/2026-03-16_maintained_refresh
```
