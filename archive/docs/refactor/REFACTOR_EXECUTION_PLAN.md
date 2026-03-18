# Refactor Execution Plan

Date: 2026-03-15

Command inventories below are normalized to canonical `src/...`,
`experiments/...`, and `artifacts/...` paths. Historical wrapper-path parity is
tracked separately in the checkpoint log.

## Goal

Refactor the repository into a clearer canonical structure, remove duplicated
backend scaffolding where practical, preserve the current main solver CLIs via
compatibility wrappers or symlinks, and validate the work with staged baseline,
checkpoint, and final reproduction runs.

## Canonical Target Layout

The canonical structure after this refactor is:

```text
src/
  core/
    benchmark/
    cli/
    coloring/
    fenics/
    petsc/
    problem_data/
    serial/
  problems/
    plaplace/
      fenics/
      jax/
      jax_petsc/
      support/
    ginzburg_landau/
      fenics/
      jax/
      jax_petsc/
      support/
    hyperelasticity/
      fenics/
      jax/
      jax_petsc/
      support/
    topology/
      support/
experiments/
  runners/
  sweeps/
  diagnostics/
  analysis/
  legacy/
notebooks/
  demos/
  benchmarks/
docs/
  overview/
  setup/
  demos/
  problem_setup/
  implementation/
  benchmarks/
  assets/
data/
  meshes/
  curated_results/
artifacts/
  reproduction/
  raw_results/
  figures/
archive/
scratch/
```

## Current-To-Target Move Map

| Current path | Canonical target |
| --- | --- |
| `tools/` | `src/core/serial/` |
| `tools_petsc4py/` | `src/core/petsc/` |
| `graph_coloring/` | `src/core/coloring/` |
| `pLaplace2D_fenics/` | `src/problems/plaplace/fenics/` |
| `pLaplace2D_jax/` | `src/problems/plaplace/jax/` |
| `pLaplace2D_jax_petsc/` | `src/problems/plaplace/jax_petsc/` |
| `pLaplace2D_petsc_support/` | `src/problems/plaplace/support/` |
| `GinzburgLandau2D_fenics/` | `src/problems/ginzburg_landau/fenics/` |
| `GinzburgLandau2D_jax/` | `src/problems/ginzburg_landau/jax/` |
| `GinzburgLandau2D_jax_petsc/` | `src/problems/ginzburg_landau/jax_petsc/` |
| `HyperElasticity3D_fenics/` | `src/problems/hyperelasticity/fenics/` |
| `HyperElasticity3D_jax/` | `src/problems/hyperelasticity/jax/` |
| `HyperElasticity3D_jax_petsc/` | `src/problems/hyperelasticity/jax_petsc/` |
| `HyperElasticity3D_petsc_support/` | `src/problems/hyperelasticity/support/` |
| `topological_optimisation_jax/` | `src/problems/topology/` |
| root notebooks | `notebooks/demos/` or `notebooks/benchmarks/` |
| `mesh_data/` | `data/meshes/` |
| `results/`, `results_GL/`, `experiment_results_cache/` | `artifacts/raw_results/...` |
| `img/` | `artifacts/figures/` |
| `experiment_scripts/` | `experiments/...` |
| `tmp_scripts/`, `tmp_work/` | `scratch/` |

## Compatibility Policy

- Keep current main solver entry points working.
- Use thin wrappers or filesystem links for old locations.
- Update internal imports to canonical paths incrementally.
- Do not remove the old visible paths until parity checks pass.

## Shared-Code Extraction Scope

New shared modules to introduce:

- `src/core/problem_data/hdf5.py`
- `src/core/fenics/scalar_custom_newton.py`
- `src/core/fenics/nullspace.py`
- `src/core/petsc/gamg.py`
- `src/core/petsc/scalar_problem_driver.py`
- `src/core/petsc/load_step_driver.py`
- `src/core/petsc/reordered_element_base.py`
- `src/core/benchmark/results.py`
- `src/core/benchmark/repair.py`
- `src/core/cli/threading.py`
- `src/problems/topology/support/policy.py`

Keep problem-local:

- physics kernels in `jax_energy.py`
- FEniCS forms
- boundary conditions
- initial guesses
- HE rotating boundary logic
- topology mechanics/design coupling

## Reproduction Scope

In scope:

- `README.md` example entry points
- current final benchmark reports in `docs/`
- current topology report/state docs in `docs/`

Out of scope:

- `archive/` report reproduction
- historical root markdowns unless directly needed as reference

Reproduction policy:

- tolerance-based, not exact-bitwise
- same convergence/failure status required
- same qualitative conclusion required
- key metrics must stay within documented tolerances

Primary environment:

- current repository `.venv`

## Stage Gates

### Stage 0

- create canonical directories
- write this plan
- create `docs/REFACTOR_CHECKPOINTS.md`
- enumerate command inventory

Implemented on 2026-03-15:

- canonical top-level directories created
- core/problem/data/notebook paths moved into canonical homes with compatibility symlinks left at the old locations
- docs redistributed into `docs/overview`, `docs/setup`, `docs/problem_setup`, `docs/implementation`, and `docs/benchmarks`
- `experiment_scripts/` split into `experiments/runners`, `experiments/sweeps`, `experiments/diagnostics`, `experiments/analysis`, plus `artifacts/raw_results/experiment_scripts/` for non-script byproducts

### Stage 1

- baseline smoke runs on current code
- store outputs under `artifacts/reproduction/<campaign_id>/baseline/`

Required smoke coverage:

- pLaplace: FEniCS custom, pure JAX serial, JAX+PETSc
- GinzburgLandau: FEniCS custom, JAX+PETSc
- HyperElasticity: FEniCS custom, pure JAX serial, JAX+PETSc
- topology: maintained smoke-equivalent run(s)

### Stage 2

- representative checkpoint runs
- one report-representative case per problem family
- use these for fast parity checks during refactor

Frozen representative matrix:

- `.venv/bin/python src/problems/plaplace/jax/solve_pLaplace_jax_newton.py --levels 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/plaplace_jax_l5_np1.json`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/plaplace_fenics_custom_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/plaplace_jax_petsc_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/gl_fenics_custom_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/gl_jax_petsc_l5_np2.json`
- `.venv/bin/python src/problems/hyperelasticity/jax/solve_HE_jax_newton.py --level 1 --steps 24 --total_steps 24 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/he_jax_l1_steps24_np1.json`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total_steps 24 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/he_fenics_custom_l1_steps24_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 24 --total_steps 24 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/he_jax_petsc_l1_steps24_np2.json`
- `.venv/bin/python src/problems/topology/jax/solve_topopt_jax.py --quiet --json_out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/topology_serial_nx192_ny96_np1.json`
- `JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' mpiexec -n 2 .venv/bin/python src/problems/topology/jax/solve_topopt_parallel.py --nx 768 --ny 384 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 32 --load_pad_cells 32 --volume_fraction_target 0.4 --theta_min 1e-6 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 10.0 --p_increment 0.2 --continuation_interval 1 --outer_maxit 2000 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --quiet --json_out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/json/topology_parallel_nx768_ny384_np2.json`

### Stage 3

- structural moves
- compatibility links/wrappers
- docs/import updates
- rerun Stage 1 smoke matrix

### Stage 4

Shared extraction order:

1. problem-data loaders
2. scalar FEniCS custom driver
3. scalar JAX+PETSc driver
4. PETSc/FEniCS helpers
5. reordered element base
6. HE canonical data/step driver
7. topology policy helpers

After each extraction:

- rerun affected smoke cases
- rerun affected representative checkpoints

### Stage 5

- rerun the experiments needed for current final report tables and figures
- publish under `artifacts/reproduction/<campaign_id>/full/`
- curate accepted assets into `docs/assets/`

### Stage 6

- final comparison against docs and pre-refactor checkpoints
- update checkpoint log
- leave compatibility links/wrappers in place

Implemented closeout on 2026-03-16:

- maintained pLaplace / GL / HE figure-generation scripts moved from the legacy
  figure-generator location into `experiments/analysis/`
- generated benchmark figures now land under
  `artifacts/figures/benchmark_reports/`
- final campaign summary and comparison files live under
  `artifacts/reproduction/2026-03-15_refactor_stage2b_final/`
- validated curated topology scaling assets were promoted into the final
  campaign because the scaling solver/report path itself was unchanged in the
  cleanup closeout

## Tolerances

Default unless a report-specific tighter threshold is needed:

- status must match exactly
- iteration counts may drift if the report itself allows it, but should remain
  qualitatively consistent
- energy / compliance / volume style scalar metrics:
  relative tolerance `1e-3`
- wall-time metrics:
  informational only unless a report claim depends on ordering
- figure reproduction:
  accept regenerated curves/images when derived metrics match and visual trends
  are consistent

## Command Inventory

Campaign root currently in use:

- `artifacts/reproduction/2026-03-15_refactor_stage1/`
- `artifacts/reproduction/2026-03-15_refactor_stage2/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/`

Stage 1 baseline commands completed on 2026-03-15:

- `.venv/bin/python src/problems/plaplace/jax/solve_pLaplace_jax_newton.py --levels 5 --json artifacts/reproduction/2026-03-15_refactor_stage1/baseline/plaplace_jax_l5_np1.json --quiet`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --json artifacts/reproduction/2026-03-15_refactor_stage1/baseline/plaplace_fenics_custom_l5_np2.json --quiet`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --json artifacts/reproduction/2026-03-15_refactor_stage1/baseline/plaplace_jax_petsc_l5_np2.json --quiet --nproc 1`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --json artifacts/reproduction/2026-03-15_refactor_stage1/baseline/gl_fenics_custom_l5_np2.json --quiet`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --out artifacts/reproduction/2026-03-15_refactor_stage1/baseline/gl_jax_petsc_l5_np2.json --quiet --nproc 1`
- `.venv/bin/python src/problems/hyperelasticity/jax/solve_HE_jax_newton.py --level 1 --steps 1 --total_steps 96 --out artifacts/reproduction/2026-03-15_refactor_stage1/baseline/he_jax_l1_steps1_np1.json --quiet`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 1 --total_steps 96 --out artifacts/reproduction/2026-03-15_refactor_stage1/baseline/he_fenics_custom_l1_steps1_np2.json --quiet`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 1 --total_steps 96 --out artifacts/reproduction/2026-03-15_refactor_stage1/baseline/he_jax_petsc_l1_steps1_np2.json --quiet --nproc 1`
- `.venv/bin/python src/problems/topology/jax/solve_topopt_jax.py --nx 24 --ny 12 --fixed_pad_cells 4 --load_pad_cells 4 --outer_maxit 3 --design_maxit 60 --quiet --json_out artifacts/reproduction/2026-03-15_refactor_stage1/baseline/topology_serial_nx24_ny12_np1.json`
- `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' mpirun -n 2 .venv/bin/python src/problems/topology/jax/solve_topopt_parallel.py --nx 24 --ny 12 --fixed_pad_cells 4 --load_pad_cells 4 --outer_maxit 3 --design_maxit 20 --quiet --json_out artifacts/reproduction/2026-03-15_refactor_stage1/baseline/topology_parallel_nx24_ny12_np2.json`

Shared-helper extraction already completed in this pass:

- `src/core/problem_data/hdf5.py`
- `src/core/petsc/gamg.py`
- `src/core/fenics/nullspace.py`
- `src/core/cli/threading.py`
- `src/problems/topology/support/policy.py`

These commands, plus their logs and JSON outputs, are also recorded under the
campaign root and summarised in `artifacts/reproduction/2026-03-15_refactor_stage1/comparison.md`.

Stage 2 shared-scalar checkpoint commands completed on 2026-03-15:

- `python -m compileall -q src tests experiments/runners experiments/analysis`
- `.venv/bin/pytest -q tests/test_import_hygiene.py tests/test_jax_diff.py tests/test_problem_data_hdf5.py tests/test_shared_helpers.py tests/test_topology_policy.py`
- `.venv/bin/pytest -q tests/test_topology_gradient_descent_minimizers.py tests/test_topology_petsc_gradient_descent_minimizers.py`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/plaplace_fenics_custom_canonical_l5_np2.json`
- `mpirun -n 2 .venv/bin/python pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/plaplace_fenics_custom_wrapper_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/plaplace_jax_petsc_canonical_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/gl_fenics_custom_canonical_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/gl_jax_petsc_canonical_l5_np2.json`

Stage 2 checkpoint outputs:

- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/comparison.json`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/comparison.md`

Stage 2 reordered-base checkpoint commands completed on 2026-03-15:

- `python -m compileall -q src/core/petsc/reordered_element_base.py src/problems/plaplace/jax_petsc/reordered_element_assembler.py src/problems/ginzburg_landau/jax_petsc/reordered_element_assembler.py src/problems/hyperelasticity/jax_petsc/reordered_element_assembler.py`
- `.venv/bin/pytest -q tests/test_reordered_element_base.py`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/json/plaplace_jax_petsc_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/json/gl_jax_petsc_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 1 --total_steps 96 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/json/he_jax_petsc_l1_steps1_np2.json --nproc 1`

Stage 2 reordered-base checkpoint outputs:

- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/json/`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/comparison.json`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/comparison.md`

Stage 2 HE load-step checkpoint commands completed on 2026-03-15:

- `python -m compileall -q src/core/petsc/load_step_driver.py src/core/benchmark/repair.py src/problems/hyperelasticity/jax_petsc/solver.py src/problems/hyperelasticity/fenics/solver_custom_newton.py`
- `python -m compileall -q src tests`
- `.venv/bin/pytest -q tests/test_load_step_driver.py tests/test_reordered_element_base.py tests/test_import_hygiene.py`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/json/plaplace_jax_petsc_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/json/gl_jax_petsc_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 1 --total_steps 96 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/json/he_jax_petsc_l1_steps1_np2.json --nproc 1`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 1 --total_steps 96 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/json/he_fenics_custom_l1_steps1_np2.json`

Stage 2 HE load-step checkpoint outputs:

- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/json/`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/comparison.json`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/comparison.md`
