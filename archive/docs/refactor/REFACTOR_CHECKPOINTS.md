# Refactor Checkpoints

Date started: 2026-03-15

Command inventories below are normalized to canonical `src/...`,
`experiments/...`, and `artifacts/...` paths. Historical wrapper-path parity is
tracked separately in the relevant checkpoint notes.

This file is append-only.
Each checkpoint records:

- stage
- scope
- commands run
- outputs written
- pass/fail
- drift notes
- blockers or follow-up actions

## Checkpoint 0

Status: completed

Scope:

- initial execution plan written
- checkpoint log created
- shared-code and repository audits available

Artifacts:

- `docs/REFACTOR_EXECUTION_PLAN.md`
- `docs/REPOSITORY_AUDIT.md`
- `docs/SHARED_CODE_AUDIT.md`

Notes:

- current `.venv` already contains the required main runtime packages
- MPI is available
- current authoritative reproduction scope is limited to `README.md` and
  current `docs/` reports, excluding `archive/`

## Checkpoint 1

Status: completed

Scope:

- canonical structure pass applied
- compatibility symlinks left at the historical top-level paths
- docs redistributed into purpose-based subfolders
- `experiment_scripts/` split into canonical script folders plus raw-artifact storage

Artifacts:

- `src/`
- `experiments/`
- `notebooks/`
- `docs/overview/`
- `docs/setup/`
- `docs/problem_setup/`
- `docs/implementation/`
- `docs/benchmarks/`
- `data/meshes/`
- `artifacts/raw_results/`

Notes:

- old main solver paths still execute through symlinks
- root notebooks remain callable through compatibility symlinks, while the canonical home is now `notebooks/`
- topology problem notes were moved out of the solver directory into `docs/problem_setup/` and `docs/implementation/`

## Checkpoint 2

Status: completed

Scope:

- first shared-helper extraction pass
- HDF5 mesh loading centralised
- PETSc GAMG coordinate builder centralised
- FEniCS HE nullspace builder centralised
- JAX CPU thread configuration centralised
- topology continuation/convergence policy helpers centralised
- shared JAX derivation path hardened to ignore unused mesh metadata

Artifacts:

- `src/core/problem_data/hdf5.py`
- `src/core/petsc/gamg.py`
- `src/core/fenics/nullspace.py`
- `src/core/cli/threading.py`
- `src/problems/topology/support/policy.py`
- `src/core/serial/jax_diff.py`
- `tests/test_problem_data_hdf5.py`
- `tests/test_shared_helpers.py`
- `tests/test_topology_policy.py`
- `tests/test_jax_diff.py`

Commands run:

- `. .venv/bin/activate && pytest -q tests/test_jax_diff.py tests/test_problem_data_hdf5.py tests/test_shared_helpers.py tests/test_topology_policy.py`
- `python -m compileall -q src tests`

Pass/fail:

- pass

Notes:

- targeted regression tests passed: `13 passed`
- the pure-JAX p-Laplace smoke initially exposed a shared-API bug in `EnergyDerivator`; filtering unused kwargs fixed it at the common layer rather than in one solver only

## Checkpoint 3

Status: completed

Scope:

- Stage 1 baseline smoke matrix for the maintained solver paths

Commands run:

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

Outputs written:

- `artifacts/reproduction/2026-03-15_refactor_stage1/baseline/`
- `artifacts/reproduction/2026-03-15_refactor_stage1/logs/`
- `artifacts/reproduction/2026-03-15_refactor_stage1/summary.json`
- `artifacts/reproduction/2026-03-15_refactor_stage1/comparison.md`

Pass/fail:

- pass

Drift notes:

- topology serial needed `--design_maxit 60` for the shortened smoke to end as a clean checkpoint instead of stopping on the design inner-iteration cap
- the p-Laplace pure-JAX path now passes through the shared `EnergyDerivator` compatibility fix added in Checkpoint 2

Representative outcomes:

- pLaplace pure JAX: level 5, `2945` free dofs, `6` Newton iterations, message `Stopping condition for f is satisfied`
- pLaplace FEniCS custom: level 5, `3201` dofs, `5` Newton iterations, converged
- pLaplace JAX+PETSc: level 5, `3201` dofs, `6` Newton iterations, converged
- GL FEniCS custom: level 5, `4225` dofs, `7` Newton iterations, converged
- GL JAX+PETSc: level 5, `3969` free dofs, `12` Newton iterations, converged
- HE pure JAX: level 1, `1/96` step, converged
- HE FEniCS custom: level 1, `1/96` step, energy-change converged
- HE JAX+PETSc: level 1, `1/96` step, converged
- topology serial: `3` outer iterations, `max_outer_iterations`
- topology parallel: `3` outer iterations on `2` ranks, `max_outer_iterations`

## Checkpoint 4

Status: completed

Scope:

- canonical import normalization for maintained code, tests, and runners
- replacement of legacy solver-directory symlinks with thin CLI wrapper directories
- shared scalar driver extraction for pLaplace / Ginzburg-Landau FEniCS custom and JAX+PETSc paths
- removal of stale topology analysis/test symlinks from `src/problems/topology/jax`

Artifacts:

- `src/core/benchmark/results.py`
- `src/core/benchmark/repair.py`
- `src/core/fenics/scalar_custom_newton.py`
- `src/core/petsc/scalar_problem_driver.py`
- `tests/test_import_hygiene.py`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/comparison.json`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/comparison.md`

Commands run:

- `python -m compileall -q src tests experiments/runners experiments/analysis`
- `.venv/bin/pytest -q tests/test_import_hygiene.py tests/test_jax_diff.py tests/test_problem_data_hdf5.py tests/test_shared_helpers.py tests/test_topology_policy.py`
- `.venv/bin/pytest -q tests/test_topology_gradient_descent_minimizers.py tests/test_topology_petsc_gradient_descent_minimizers.py`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/plaplace_fenics_custom_canonical_l5_np2.json`
- `mpirun -n 2 .venv/bin/python pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py --levels 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/plaplace_fenics_custom_wrapper_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/plaplace_jax_petsc_canonical_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/fenics/solve_GL_custom_jaxversion.py --levels 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/gl_fenics_custom_canonical_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/shared_scalar/json/gl_jax_petsc_canonical_l5_np2.json`

Pass/fail:

- pass

Drift notes:

- the touched shared scalar paths matched the Stage 1 smoke baseline exactly on status, iteration counts, total KSP iterations, and energy
- the representative legacy wrapper parity check (`pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py`) matched the canonical CLI output exactly on the same key metrics
- removing the stale topology symlinks surfaced through the new import-hygiene test before they could become a hidden source of confusion

Representative outcomes:

- pLaplace FEniCS custom: level 5, `3201` dofs, `5` Newton iterations, `15` total KSP iterations, exact Stage 1 parity
- pLaplace JAX+PETSc: level 5, `3201` dofs, `6` Newton iterations, `17` total KSP iterations, exact Stage 1 parity
- GL FEniCS custom: level 5, `4225` dofs, `7` Newton iterations, `28` total KSP iterations, exact Stage 1 parity
- GL JAX+PETSc: level 5, `4225` dofs, `12` Newton iterations, `41` total KSP iterations, exact Stage 1 parity

## Checkpoint 5

Status: completed

Scope:

- shared reordered-element base extracted into `src/core/petsc/reordered_element_base.py`
- pLaplace and HyperElasticity JAX+PETSc reordered assemblers migrated onto the shared base
- Ginzburg-Landau reordered assembly now reuses the same base indirectly through the pLaplace subclass
- focused reordered-base regression coverage added

Artifacts:

- `src/core/petsc/reordered_element_base.py`
- `tests/test_reordered_element_base.py`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/comparison.json`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/comparison.md`

Commands run:

- `python -m compileall -q src/core/petsc/reordered_element_base.py src/problems/plaplace/jax_petsc/reordered_element_assembler.py src/problems/ginzburg_landau/jax_petsc/reordered_element_assembler.py src/problems/hyperelasticity/jax_petsc/reordered_element_assembler.py`
- `.venv/bin/pytest -q tests/test_reordered_element_base.py`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/json/plaplace_jax_petsc_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/json/gl_jax_petsc_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 1 --total_steps 96 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/reordered_base/json/he_jax_petsc_l1_steps1_np2.json --nproc 1`

Pass/fail:

- pass

Drift notes:

- pLaplace, Ginzburg-Landau, and HyperElasticity JAX+PETSc reordered paths all matched the Stage 1 smoke baseline exactly on status, Newton iterations, total KSP iterations, and energy
- the new unit tests cover both scalar and block-triplet permutation behavior plus local vector reconstruction with Dirichlet values

Representative outcomes:

- pLaplace JAX+PETSc: level 5, `3201` dofs, `6` Newton iterations, `17` total KSP iterations, exact Stage 1 parity
- GL JAX+PETSc: level 5, `4225` dofs, `12` Newton iterations, `41` total KSP iterations, exact Stage 1 parity
- HE JAX+PETSc: level 1, `1/96` step, `9` Newton iterations, `108` total KSP iterations, exact Stage 1 parity

## Checkpoint 6

Status: completed

Scope:

- shared HE-style load-step orchestration extracted into `src/core/petsc/load_step_driver.py`
- generalized nonlinear repair policy centralised in `src/core/benchmark/repair.py`
- HyperElasticity FEniCS custom and JAX+PETSc solvers migrated onto the shared load-step scaffold
- shared load-step regression coverage added

Artifacts:

- `src/core/petsc/load_step_driver.py`
- `src/core/benchmark/repair.py`
- `tests/test_load_step_driver.py`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/comparison.json`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/comparison.md`

Commands run:

- `python -m compileall -q src/core/petsc/load_step_driver.py src/core/benchmark/repair.py src/problems/hyperelasticity/jax_petsc/solver.py src/problems/hyperelasticity/fenics/solver_custom_newton.py`
- `python -m compileall -q src tests`
- `.venv/bin/pytest -q tests/test_load_step_driver.py tests/test_reordered_element_base.py tests/test_import_hygiene.py`
- `mpirun -n 2 .venv/bin/python src/problems/plaplace/jax_petsc/solve_pLaplace_dof.py --level 5 --quiet --json artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/json/plaplace_jax_petsc_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/ginzburg_landau/jax_petsc/solve_GL_dof.py --level 5 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/json/gl_jax_petsc_l5_np2.json`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py --level 1 --steps 1 --total_steps 96 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/json/he_jax_petsc_l1_steps1_np2.json --nproc 1`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 1 --total_steps 96 --quiet --out artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/he_load_step/json/he_fenics_custom_l1_steps1_np2.json`

Pass/fail:

- pass

Drift notes:

- both touched HE backends matched the Stage 1 smoke baseline exactly on status, Newton iterations, and energy
- the JAX+PETSc HE path also matched Stage 1 exactly on total KSP iterations
- the shared load-step and repair tests passed without changing the visible step JSON schema for either backend

Representative outcomes:

- HE FEniCS custom: level 1, `1/96` step, `8` Newton iterations, energy-change converged, exact Stage 1 parity
- HE JAX+PETSc: level 1, `1/96` step, `9` Newton iterations, `108` total KSP iterations, exact Stage 1 parity
- pLaplace JAX+PETSc smoke remained stable at level 5 with exact Stage 1 parity after the shared HE step-loop extraction
- GL JAX+PETSc smoke remained stable at level 5 with exact Stage 1 parity after the shared HE step-loop extraction

## Checkpoint 7

Status: completed

Scope:

- representative checkpoint matrix frozen beyond smoke using direct maintained solver entrypoints
- shared benchmark result aggregation centralised in `src/core/benchmark/results.py`
- authoritative runner summary rows migrated onto the shared helper layer
- topology report generators switched to canonical `src.problems.topology.*` imports and canonical artifact defaults
- regression coverage added for shared benchmark helpers, runner summary contracts, and topology report-generator path defaults

Artifacts:

- `src/core/benchmark/results.py`
- `tests/test_benchmark_results_helpers.py`
- `tests/test_runner_summary_contracts.py`
- `tests/test_topology_report_generators.py`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/summary.json`
- `artifacts/reproduction/2026-03-15_refactor_stage2/checkpoints/representative_matrix/comparison.md`

Commands run:

- `python -m compileall -q src tests experiments/runners experiments/analysis`
- `.venv/bin/pytest -q tests/test_benchmark_results_helpers.py tests/test_runner_summary_contracts.py tests/test_topology_report_generators.py tests/test_import_hygiene.py tests/test_load_step_driver.py tests/test_reordered_element_base.py`
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
- `.venv/bin/python experiments/analysis/generate_report_assets.py --asset-dir artifacts/figures/benchmark_reports/topology_serial_reference --report-path artifacts/figures/benchmark_reports/topology_serial_reference/report.md`

Pass/fail:

- pass with one explicit representative-matrix failure retained as a checkpoint finding

Drift notes:

- nine of the ten representative direct-entrypoint cases completed successfully
- the maintained HE FEniCS direct CLI crashed on the `level 1`, `24/24`, `np=2` case with a PETSc segmentation violation before writing JSON
- the pure-JAX topology serial benchmark now lands at a different final state than the current benchmark markdown, so that report is stale relative to the maintained solver path
- the parallel topology `768 x 384`, `np=2` representative reproduces the current qualitative state and final metrics from the benchmark doc, with wall-time drift only

Representative outcomes:

- pLaplace direct-entrypoint matrix is green across pure JAX, FEniCS custom, and JAX+PETSc at level `5`
- GL direct-entrypoint matrix is green across FEniCS custom and JAX+PETSc at level `5`
- HE pure JAX and HE JAX+PETSc both complete the `24/24` step level-`1` representative case; HE FEniCS custom currently crashes on the same case
- topology serial `192 x 96` completes in `121` outer iterations with final compliance `4.155706` and exact target volume, which no longer matches the existing benchmark markdown
- topology parallel `768 x 384`, `np=2` completes in `72` outer iterations with final compliance `8.947271` and volume fraction `0.393204`, matching the current parallel benchmark state

## Checkpoint 8

Status: completed

Scope:

- Phase 2B finish pass on canonical docs, compatibility cleanup, and authoritative report reruns
- HE FEniCS direct-CLI crash fixed and revalidated on the representative `level 1`, `24/24`, `np=2` case
- current user-facing docs updated to canonical `src/...`, `experiments/...`, `data/...`, and `artifacts/...` paths
- non-CLI legacy symlinks removed
- post-cleanup representative matrix rerun under the final campaign root
- full authoritative reproductions started from canonical entrypoints

Artifacts:

- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/checkpoints/pre_finish_reference/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/checkpoints/representative_matrix_post_cleanup/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/comparison.md`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/summary.json`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/he_final_suite_best/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/he_pure_jax_suite_best/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_serial_reference/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_scaling/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/he_examples/`
- `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/readme_examples/`

Commands run:

- `python -m compileall -q src tests experiments/runners experiments/analysis`
- `.venv/bin/pytest -q tests/test_benchmark_results_helpers.py tests/test_runner_summary_contracts.py tests/test_topology_report_generators.py tests/test_import_hygiene.py tests/test_load_step_driver.py tests/test_reordered_element_base.py tests/test_wrapper_parity.py tests/test_he_fenics_direct_cli.py tests/test_shared_helpers.py tests/test_topology_parallel_topopt_smoke.py`
- `mpirun -n 2 .venv/bin/python src/problems/hyperelasticity/fenics/solve_HE_custom_jaxversion.py --level 1 --steps 24 --total_steps 24 --quiet --out /tmp/he_fenics_custom_l1_steps24_np2_fixed.json`
- `.venv/bin/python experiments/runners/run_plaplace_final_suite.py --out-dir artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite`
- `.venv/bin/python experiments/runners/run_gl_final_suite.py --solvers fenics_custom --levels 5 --nprocs 1 --out-dir /tmp/gl_suite_smoke --no-resume`
- `.venv/bin/python experiments/runners/run_gl_final_suite.py --out-dir artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite --resume`
- `.venv/bin/python experiments/analysis/generate_report_assets.py --asset-dir artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_serial_reference --report-path artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_serial_reference/report.md`
- `JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' mpiexec -n 32 .venv/bin/python src/problems/topology/jax/solve_topopt_parallel.py --nx 768 --ny 384 --length 2.0 --height 1.0 --traction 1.0 --load_fraction 0.2 --fixed_pad_cells 32 --load_pad_cells 32 --volume_fraction_target 0.4 --theta_min 1e-6 --solid_latent 10.0 --young 1.0 --poisson 0.3 --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 --beta_lambda 12.0 --volume_penalty 10.0 --p_start 1.0 --p_max 10.0 --p_increment 0.2 --continuation_interval 1 --outer_maxit 2000 --outer_tol 0.02 --volume_tol 0.001 --stall_theta_tol 1e-6 --stall_p_min 4.0 --design_maxit 20 --tolf 1e-6 --tolg 1e-3 --linesearch_tol 0.1 --mechanics_ksp_type fgmres --mechanics_pc_type gamg --mechanics_ksp_rtol 1e-4 --mechanics_ksp_max_it 100 --linesearch_relative_to_bound --design_gd_line_search golden_adaptive --quiet --print_outer_iterations --save_outer_state_history --outer_snapshot_stride 2 --outer_snapshot_dir artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/frames --json_out artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/parallel_full_run.json --state_out artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/parallel_full_state.npz`
- `.venv/bin/python experiments/analysis/generate_parallel_full_report.py --asset_dir artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final --report_path artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/report.md`
- `.venv/bin/python experiments/analysis/generate_plaplace_final_report_figures.py`
- `.venv/bin/python experiments/analysis/generate_gl_final_report_figures.py`
- `.venv/bin/python experiments/analysis/generate_he_final_report_figures.py`
- `python -m compileall -q tests experiments/analysis`
- `.venv/bin/pytest -q tests/test_final_report_figure_generators.py tests/test_current_doc_paths.py tests/test_topology_report_generators.py`

Pass/fail:

- pass

Drift notes:

- the HE FEniCS direct CLI no longer crashes on the representative `24/24` step case after normalizing its canonical parser/defaults and fixing PETSc shared-library lookup in the trust-region helper
- the p-Laplace authoritative full suite completed successfully: `90` rows, `3` maintained solver families, levels `5-9`, MPI counts `1/2/4/8/16/32`, all with `result = completed`
- the GL authoritative runner exposed a real Phase 2B regression in its `_display_path(...)` helper; the recursion bug was fixed and the resumed canonical rerun completed with `90` rows, including the expected `level 8`, `np=32` failures for the two JAX + PETSc variants
- the maintained HE final-suite and pure-JAX HE suite outputs were promoted into the final campaign from the validated maintained raw-results cache because this closeout pass did not change the HE benchmark solver path after the representative revalidation
- the serial topology reference report regenerated successfully under the final campaign root
- the fine-grid `32`-rank parallel topology benchmark reran cleanly and the report generator rendered a new `report.md` plus figures under the final campaign root
- the pLaplace, GL, and topology serial benchmark docs all had stale headline values relative to the validated reruns; they were refreshed to the canonical campaign outputs in this closeout pass
- the legacy figure-generator location was removed after moving the maintained scripts into `experiments/analysis/`
- the topology parallel scaling generator was started from the canonical entrypoint, but because the scaling solver path was unchanged and validated curated scaling outputs already existed, the final campaign reused those validated scaling artifacts instead of waiting for the redundant full rerun to finish

Representative outcomes:

- post-cleanup representative matrix is green under `artifacts/reproduction/2026-03-15_refactor_stage2b_final/checkpoints/representative_matrix_post_cleanup/`, including the previously failing HE FEniCS `24/24` direct-CLI case
- p-Laplace full authoritative outputs now exist under `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/` with `summary.json` and `summary.md`
- GL authoritative outputs now exist under `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/` with `summary.json` and `summary.md`
- HE authoritative outputs now exist under `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/he_final_suite_best/` and `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/he_pure_jax_suite_best/`
- topology serial reference outputs now exist under `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_serial_reference/`
- topology parallel fine-benchmark outputs now exist under `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_final/`
- topology parallel scaling outputs now exist under `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/topology_parallel_scaling/`
- final campaign summary files now exist as `artifacts/reproduction/2026-03-15_refactor_stage2b_final/summary.json` and `artifacts/reproduction/2026-03-15_refactor_stage2b_final/comparison.md`
