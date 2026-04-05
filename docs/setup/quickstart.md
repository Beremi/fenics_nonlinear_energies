# Quickstart

This page is the shortest path from a fresh checkout to running the maintained
solvers and maintained benchmark runners.

## Recommended Starting Points

- repository index: [docs/README.md](../README.md)
- problem overviews:
  - [pLaplace](../problems/pLaplace.md)
  - [GinzburgLandau](../problems/GinzburgLandau.md)
  - [HyperElasticity](../problems/HyperElasticity.md)
  - [Plasticity](../problems/Plasticity.md)
  - [Plasticity3D](../problems/Plasticity3D.md)
  - [Topology](../problems/Topology.md)
- current maintained results:
  - [pLaplace results](../results/pLaplace.md)
  - [GinzburgLandau results](../results/GinzburgLandau.md)
  - [HyperElasticity results](../results/HyperElasticity.md)
  - [Plasticity results](../results/Plasticity.md)
  - [Topology results](../results/Topology.md)

## Environment

The maintained solver set expects:

- DOLFINx with PETSc for the FEniCS-based paths
- `jax`, `h5py`, and `pyamg` for the serial JAX paths
- `petsc4py` and `mpi4py` for the distributed JAX+PETSc paths

The checked-in `.venv` and the devcontainer are the easiest maintained
environments. For a local source build that mirrors the devcontainer, use
[local_build.md](local_build.md).

## Running In Docker

Serial run:

```bash
docker run --rm --entrypoint python3 -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/repo -w /repo fenics_test \
  src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py
```

Parallel run:

```bash
docker run --rm --shm-size=8g --entrypoint mpirun -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/repo -w /repo fenics_test \
  -n 4 python3 src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py
```

Use `--shm-size=8g` or larger for multi-rank MPI runs. The image uses MPICH,
and Docker's default shared-memory allocation is too small for larger parallel
jobs.

## Running In The Devcontainer

After reopening the repository in the devcontainer, run commands from the repo
root:

```bash
python3 src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py
mpirun -n 4 python3 src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py
```

## Direct Solver Commands

All examples below assume the current working directory is the repository root.
Example outputs are written under `artifacts/raw_results/example_runs/`.

### pLaplace

FEniCS custom:

```bash
./.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py \
  --levels 5 --quiet \
  --json artifacts/raw_results/example_runs/plaplace_fenics_custom_l5.json
```

FEniCS SNES:

```bash
./.venv/bin/python -u src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py \
  --levels 5 \
  --json artifacts/raw_results/example_runs/plaplace_fenics_snes_l5.json
```

Pure JAX serial:

```bash
./.venv/bin/python -u src/problems/plaplace/jax/solve_pLaplace_jax_newton.py \
  --levels 5 --quiet \
  --json artifacts/raw_results/example_runs/plaplace_jax_serial_l5.json
```

JAX+PETSc element:

```bash
./.venv/bin/python -u src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py \
  --level 5 --profile reference --assembly-mode element \
  --local-hessian-mode element --element-reorder-mode block_xyz \
  --local-coloring --nproc 1 --quiet \
  --json artifacts/raw_results/example_runs/plaplace_jax_petsc_element_l5.json
```

### GinzburgLandau

FEniCS custom:

```bash
./.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py \
  --levels 5 --quiet \
  --json artifacts/raw_results/example_runs/gl_fenics_custom_l5.json
```

FEniCS SNES:

```bash
./.venv/bin/python -u src/problems/ginzburg_landau/fenics/solve_GL_snes_newton.py \
  --levels 5 \
  --json artifacts/raw_results/example_runs/gl_fenics_snes_l5.json
```

JAX+PETSc element:

```bash
./.venv/bin/python -u src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py \
  --level 5 --profile reference --assembly_mode element \
  --local_hessian_mode element --element_reorder_mode block_xyz \
  --local_coloring --nproc 1 --quiet \
  --out artifacts/raw_results/example_runs/gl_jax_petsc_element_l5.json
```

### HyperElasticity

FEniCS custom:

```bash
mpiexec -n 1 ./.venv/bin/python -u experiments/runners/run_trust_region_case.py \
  --problem he --backend fenics --level 1 \
  --steps 24 --start-step 1 --total-steps 24 --profile performance \
  --ksp-type stcg --pc-type gamg --ksp-rtol 1e-1 --ksp-max-it 30 \
  --gamg-threshold 0.05 --gamg-agg-nsmooths 1 --gamg-set-coordinates \
  --use-near-nullspace --no-pc-setup-on-ksp-cap \
  --tolf 1e-4 --tolg 1e-3 --tolg-rel 1e-3 --tolx-rel 1e-3 --tolx-abs 1e-10 \
  --maxit 100 --linesearch-a -0.5 --linesearch-b 2.0 --linesearch-tol 1e-1 \
  --use-trust-region --trust-radius-init 0.5 \
  --trust-radius-min 1e-8 --trust-radius-max 1e6 \
  --trust-shrink 0.5 --trust-expand 1.5 \
  --trust-eta-shrink 0.05 --trust-eta-expand 0.75 --trust-max-reject 6 \
  --trust-subproblem-line-search --save-history --save-linear-timing --quiet \
  --out artifacts/raw_results/example_runs/he_fenics_custom_l1_s24.json
```

Pure JAX serial:

```bash
./.venv/bin/python -u src/problems/hyperelasticity/jax/solve_HE_jax_newton.py \
  --level 1 --steps 24 --total_steps 24 --quiet \
  --out artifacts/raw_results/example_runs/he_jax_serial_l1_s24.json
```

JAX+PETSc element:

```bash
./.venv/bin/python -u src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py \
  --level 1 --steps 24 --total_steps 24 --profile performance \
  --ksp_type stcg --pc_type gamg --ksp_rtol 1e-1 --ksp_max_it 30 \
  --gamg_threshold 0.05 --gamg_agg_nsmooths 1 --gamg_set_coordinates \
  --use_near_nullspace --assembly_mode element --element_reorder_mode block_xyz \
  --local_hessian_mode element --local_coloring --use_trust_region \
  --trust_subproblem_line_search --linesearch_tol 1e-1 --trust_radius_init 0.5 \
  --trust_shrink 0.5 --trust_expand 1.5 --trust_eta_shrink 0.05 \
  --trust_eta_expand 0.75 --trust_max_reject 6 --nproc 1 --quiet \
  --out artifacts/raw_results/example_runs/he_jax_petsc_element_l1_s24.json
```

### Topology

For timings comparable to the maintained topology tables, pin the JAX CPU
backend to a single thread before running the commands below:

```bash
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
```

Pure JAX serial reference:

```bash
./.venv/bin/python -u src/problems/topology/jax/solve_topopt_jax.py \
  --nx 192 --ny 96 --length 2.0 --height 1.0 \
  --traction 1.0 --load_fraction 0.2 \
  --fixed_pad_cells 16 --load_pad_cells 16 \
  --volume_fraction_target 0.4 --theta_min 0.001 \
  --solid_latent 10.0 --young 1.0 --poisson 0.3 \
  --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 \
  --beta_lambda 12.0 --volume_penalty 10.0 \
  --p_start 1.0 --p_max 4.0 --p_increment 0.5 \
  --continuation_interval 20 --outer_maxit 180 \
  --outer_tol 0.02 --volume_tol 0.001 \
  --mechanics_maxit 200 --design_maxit 400 \
  --tolf 1e-6 --tolg 1e-3 --ksp_rtol 1e-2 --ksp_max_it 80 \
  --quiet \
  --json_out artifacts/raw_results/example_runs/topology_serial_reference.json \
  --state_out artifacts/raw_results/example_runs/topology_serial_reference_state.npz
```

Parallel fine-grid benchmark:

```bash
mpiexec -n 32 ./.venv/bin/python -u src/problems/topology/jax/solve_topopt_parallel.py \
  --nx 768 --ny 384 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 \
  --fixed_pad_cells 32 --load_pad_cells 32 --volume_fraction_target 0.4 --theta_min 1e-6 \
  --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 \
  --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 \
  --p_start 1.0 --p_max 10.0 --p_increment 0.2 --continuation_interval 1 \
  --outer_maxit 2000 --outer_tol 0.02 --volume_tol 0.001 \
  --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 \
  --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --linesearch_relative_to_bound \
  --design_gd_line_search golden_adaptive --design_gd_adaptive_window_scale 2.0 \
  --mechanics_ksp_type fgmres --mechanics_pc_type gamg \
  --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 \
  --quiet --print_outer_iterations \
  --save_outer_state_history --outer_snapshot_stride 2 \
  --outer_snapshot_dir artifacts/raw_results/example_runs/topology_parallel_frames \
  --json_out artifacts/raw_results/example_runs/topology_parallel_final.json \
  --state_out artifacts/raw_results/example_runs/topology_parallel_final_state.npz
```

### Mohr-Coulomb Plasticity `P4` PMG Solve

The current featured parallel path lives under
`src/problems/slope_stability/`, but the official problem description is now
the plane-strain Mohr-Coulomb plasticity model. The showcased high-resolution
run uses the `L5` same-mesh `P4` space with a PMG hierarchy.

Problem page: [Plasticity](../problems/Plasticity.md)  
Current maintained results: [Plasticity results](../results/Plasticity.md)

```bash
./.venv/bin/python -u src/problems/slope_stability/jax_petsc/solve_slope_stability_dof.py \
  --level 5 --elem_degree 4 --lambda-target 1.0 \
  --profile performance --pc_type mg \
  --mg_strategy same_mesh_p4_p2_p1_lminus1_p1 --mg_variant legacy_pmg \
  --ksp_type fgmres --ksp_rtol 1e-2 --ksp_max_it 100 \
  --save-linear-timing --no-use_trust_region --quiet \
  --out artifacts/raw_results/example_runs/mc_plasticity_p4_l5/output.json \
  --state-out artifacts/raw_results/example_runs/mc_plasticity_p4_l5/state.npz
```

## Maintained Benchmark Runners

Convenience wrapper around the canonical maintained rerun sequence:

```bash
./.venv/bin/python experiments/runners/run_local_canonical_retest.py \
  --campaign <campaign>
```

Individually validated maintained runners:

```bash

./.venv/bin/python experiments/runners/run_plaplace_final_suite.py \
  --out-dir artifacts/reproduction/<campaign>/runs/plaplace/final_suite

./.venv/bin/python experiments/runners/run_gl_final_suite.py \
  --out-dir artifacts/reproduction/<campaign>/runs/ginzburg_landau/final_suite

./.venv/bin/python experiments/runners/run_he_final_suite_best.py \
  --out-dir artifacts/reproduction/<campaign>/runs/hyperelasticity/final_suite_best \
  --no-seed-known-results

./.venv/bin/python experiments/runners/run_he_pure_jax_suite_best.py \
  --out-dir artifacts/reproduction/<campaign>/runs/hyperelasticity/pure_jax_suite_best

./.venv/bin/python experiments/runners/run_topology_docs_suite.py \
  --out-dir artifacts/reproduction/<campaign>/runs/topology
```

Tracked docs-data refresh and figure rebuild:

```bash
./.venv/bin/python experiments/analysis/docs_assets/sync_tracked_docs_data.py \
  --campaign-root artifacts/reproduction/<campaign>

./.venv/bin/python experiments/analysis/docs_assets/build_all.py
```

## Output Conventions

- ad hoc runs:
  - `artifacts/raw_results/example_runs/`
- curated reproduction campaigns:
  - `artifacts/reproduction/<campaign>/`
- current published figures used by docs:
  - `docs/assets/`

Older sweeps, tuning campaigns, and refactor logs are preserved under
`archive/docs/` and `archive/`, but they are no longer the current runbook.
