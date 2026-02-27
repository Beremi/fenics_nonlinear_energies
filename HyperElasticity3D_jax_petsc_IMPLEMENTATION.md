# HyperElasticity3D JAX+PETSc Implementation Notes

This document describes the current `HyperElasticity3D_jax_petsc/` implementation, how it mirrors the `pLaplace2D_jax_petsc/` design, and what is currently limiting performance.

## 1. Scope and Goal

The solver is a thin MPI-parallel JAX+PETSc implementation intended to reproduce the **custom FEniCS Newton behavior** (not SNES behavior) for HyperElasticity 3D.

Main targets:
- Same nonlinear solver policy as `HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py`
- Same load stepping and retry policy
- Same linear settings/profiles (HYPRE reference profile, GAMG performance profile)
- DOF-partitioned local Hessian assembly in the same style as `pLaplace2D_jax_petsc`

## 2. File Map

Implemented package:
- `HyperElasticity3D_jax_petsc/__init__.py`
- `HyperElasticity3D_jax_petsc/mesh.py`
- `HyperElasticity3D_jax_petsc/rotate_boundary.py`
- `HyperElasticity3D_jax_petsc/parallel_hessian_dof.py`
- `HyperElasticity3D_jax_petsc/solve_HE_dof.py`

Reused shared infrastructure:
- `tools_petsc4py/dof_partition.py`
- `tools_petsc4py/parallel_assembler.py`
- `tools_petsc4py/minimizers.py`

Reference implementation mirrored:
- `pLaplace2D_jax_petsc/parallel_hessian_dof.py`
- `pLaplace2D_jax_petsc/solve_pLaplace_dof.py`

## 3. Data Model and Indexing

HyperElasticity mesh is node-based, but `DOFPartition` and assemblers operate in flat DOF space.

In `mesh.py`, each tet connectivity is expanded from scalar-node to DOF connectivity:
- scalar: `[n0, n1, n2, n3]`
- dof: `[3*n0, 3*n0+1, 3*n0+2, ..., 3*n3+2]` (12 entries)

This expansion is required so partitioning, local overlap construction, coloring, and COO extraction are all consistent with PETSc vector/matrix DOF indexing.

## 4. Partitioning and Local Coloring

The local-coloring path (`LocalColoringAssembler`) follows the same design as pLaplace:
- Per-rank local domain from owned DOFs + overlap
- Per-rank local coloring on `A^2|_J` (no global coloring broadcast)
- P2P ghost exchange for local vector reconstruction
- Local HVP evaluations for all colors
- Each rank inserts only its owned matrix rows via PETSc COO

No Hessian value Allreduce is used in local-coloring mode.

Additional vector-problem handling for HE:
- `ownership_block_size=3` in partition/distribution
- matrix block size set to 3 (`A.setBlockSize(3)`)
- optional rigid-body near-nullspace (6 modes)
- optional GAMG coordinates from owned xyz triplets
- default `reorder=False` (preserve xyz triplets for block-aware handling)

## 5. Energy, Gradient, Hessian

Energy model is Neo-Hookean:
- `W = C1 * (I1 - 3 - 2*log(detF)) + D1 * (detF - 1)^2`
- optional debug switch `use_abs_det`

Local JAX functions:
- weighted local energy for global energy reduction
- full local energy for exact local grad/HVP at owned rows

Hessian assembly:
- sparse finite differences via coloring/HVP extraction
- COO insertion through `setValuesCOO`
- explicit `A.assemble()` after COO insertion (to avoid hidden matrix-finalization cost inside KSP solve)

## 6. Nonlinear Solver Parity (Custom FEniCS)

`solve_HE_dof.py` uses `tools_petsc4py.minimizers.newton()` with:
- `tolf=1e-4`
- `tolg=1e-3`
- `tolg_rel=1e-3`
- `tolx_rel=1e-3`
- `tolx_abs=1e-10`
- `maxit=100`
- `require_all_convergence=True`
- `fail_on_nonfinite=True`
- line search interval `(-0.5, 2.0)`, tolerance `1e-3`

Load-step retry policy:
- on non-finite or max-iteration stall, retry once with:
  - `ksp_rtol *= 0.1`
  - `ksp_max_it *= 2`
  - line-search upper bound clamped to `<= 1.0`

Load stepping:
- supports `--steps`, `--total_steps`, `--start_step`
- `rotation_per_iter = 4 * 2*pi / total_steps`
- right-face Dirichlet BC is recomputed each step from reference coordinates

## 7. Linear Profiles

Profiles exposed via CLI:

Reference profile:
- `ksp_type=gmres`
- `pc_type=hypre`
- `ksp_rtol=1e-1`
- `ksp_max_it=30`
- `pc_setup_on_ksp_cap=True`

Performance profile:
- `ksp_type=gmres`
- `pc_type=gamg`
- `gamg_threshold=0.05`
- `gamg_agg_nsmooths=1`
- `ksp_rtol=1e-1`
- `ksp_max_it=30`
- `pc_setup_on_ksp_cap=True`
- block size 3 + near-nullspace + `pc.setCoordinates(...)`

## 8. Timing Instrumentation

`--save_history` stores per-Newton iteration:
- `t_grad`, `t_hess`, `t_ls`, `t_update`, `ksp_its`, convergence diagnostics

`--save_linear_timing` stores per linear callback:
- assembly total + assembly subparts (`p2p`, `hvp_compute`, `extraction`, `coo_assembly`)
- `setop_time`, `set_tolerances_time`, `pc_setup_time`, `solve_time`
- `linear_total_time`

## 9. Current Performance Findings

Step-1 (level 3, np=16, GAMG) from recent profiling:
- Sequential local HVP mode (`--hvp_eval_mode sequential`) is significantly faster than batched vmap for HE in this environment.
- Explicit `A.assemble()` moved hidden matrix-finalization work out of `ksp.solve` into assembly accounting.
- Resulting total step time improved substantially versus prior batched baseline, but remains much slower than custom FEniCS.

Observed remaining bottlenecks:
- Hessian assembly still dominates (`hvp_compute` + COO/finalization)
- Linear phase time distribution differs from FEniCS despite similar iteration counts
- Shared-memory CPU utilization does not scale linearly with MPI rank count on this workload

## 10. Important References

Project context and run setup:
- `README.md`
- `instructions.md`
- `HyperElasticity3D_jax_petsc_prompt.md`

Design background:
- `jax_parallel_partitioning.md`
- `graph_coloring_implementation.md`

HE benchmarking and solver behavior:
- `results_HyperElasticity3D.md`
- `experiment_scripts/he_l3_24_comparison_jax_fenics_jaxpetsc.md`
- `experiment_scripts/he_step1_timing_breakdown_fenics_vs_jaxpetsc.md`
- `experiment_scripts/he_step1_timing_investigation_fix1.md`

