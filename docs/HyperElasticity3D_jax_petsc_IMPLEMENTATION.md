# HyperElasticity3D JAX+PETSc Implementation Notes

This document describes the current `HyperElasticity3D_jax_petsc/` implementation, how it mirrors the `pLaplace2D_jax_petsc/` design, and what is currently limiting performance.

## Current Layout Note

The repo has since been refactored away from the original single-file layout.

Current structure:

- CLI wrapper: `HyperElasticity3D_jax_petsc/solve_HE_dof.py`
- solver logic: `HyperElasticity3D_jax_petsc/solver.py`
- problem-specific assembler glue: `HyperElasticity3D_jax_petsc/parallel_hessian_dof.py`
- shared mesh / boundary helpers: `HyperElasticity3D_petsc_support/`
- shared JAX+PETSc assembler infrastructure: `tools_petsc4py/jax_tools/parallel_assembler.py`

Where this note refers to the older `mesh.py` / `rotate_boundary.py` layout, read that as the
current support-package split above.

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
- `HyperElasticity3D_jax_petsc/parallel_hessian_dof.py`
- `HyperElasticity3D_jax_petsc/reordered_element_assembler.py`
- `HyperElasticity3D_jax_petsc/solve_HE_dof.py`
- `HyperElasticity3D_jax_petsc/solver.py`
- `HyperElasticity3D_petsc_support/mesh.py`
- `HyperElasticity3D_petsc_support/rotate_boundary.py`
- `HyperElasticity3D_petsc_support/__init__.py`

Reused shared infrastructure:
- `tools_petsc4py/dof_partition.py`
- `tools_petsc4py/jax_tools/parallel_assembler.py`
- `tools_petsc4py/minimizers.py`
- `tools_petsc4py/jax_tools/__init__.py`

Reference implementation mirrored:
- `pLaplace2D_jax_petsc/parallel_hessian_dof.py`
- `pLaplace2D_jax_petsc/solve_pLaplace_dof.py`

## 3. Data Model and Indexing

HyperElasticity mesh is node-based, but `DOFPartition` and assemblers operate in flat DOF space.

In `HyperElasticity3D_petsc_support/mesh.py`, each tet connectivity is expanded from scalar-node
to DOF connectivity:
- scalar: `[n0, n1, n2, n3]`
- dof: `[3*n0, 3*n0+1, 3*n0+2, ..., 3*n3+2]` (12 entries)

This expansion is required so partitioning, local overlap construction, coloring, and COO extraction are all consistent with PETSc vector/matrix DOF indexing.

## 4. Partitioning and Local Coloring

The SFD path still uses the shared local-coloring infrastructure
(`LocalColoringAssembler`) and follows the same design as pLaplace:
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

HE 3D vector P1 elements produce **63 graph colors** (vs 8 for 2D scalar P1
pLaplace), making the SFD approach expensive in assembly.

The production **element** path no longer goes through the generic local-coloring
assembler. It uses `HEReorderedElementAssembler` in
`reordered_element_assembler.py`:

- reorder free DOFs first (`block_xyz` default for element mode)
- derive PETSc row ownership from that reordered free-DOF layout
- build a larger overlap domain per rank so each owned row can be assembled
  locally
- rebuild the current free vector by `Allgatherv`
- compute the local gradient from one overlap-subdomain energy functional and
  exact per-element Hessians by vmapped JAX kernels
- scatter directly into the owned COO row pattern

This is a HE-specific production path because the whole point is to fix the
matrix ownership/layout regression observed in the older reduced-operator
element implementation.

## 5. Energy, Gradient, Hessian

Energy model is Neo-Hookean:
- `W = C1 * (I1 - 3 - 2*log(detF)) + D1 * (detF - 1)^2`
- optional debug switch `use_abs_det`

Local JAX functions:
- weighted local energy for global energy reduction
- full local energy for exact local grad/HVP at owned rows

### Hessian assembly modes (selected via `--assembly_mode`)

**SFD** (default, `--assembly_mode sfd`):
- Sparse finite differences via graph coloring + HVP extraction
- 63 sequential `jax.jvp(grad, v, indicator)` calls per Newton step
- COO insertion into `self.A` via `setValuesCOO(INSERT_VALUES)`
- Explicit `A.assemble()` after COO insertion

**Element** (`--assembly_mode element`):
- dedicated `HEReorderedElementAssembler` selected directly in `solver.py`
- analytical element Hessians via `jax.hessian(element_energy)` + `jax.vmap`
- whole-local gradient via `jax.grad(local_full_energy)` on the overlap
  subdomain, then direct extraction of owned rows
- default distribution strategy: overlap domain + `Allgatherv`
- default reorder for production element mode: `block_xyz`
- alternative reorder modes: `none`, `block_rcm`, `block_metis`

**Key implementation notes for element mode:**

- The large performance gap on HE was traced mainly to the PETSc row
  distribution / operator layout, not just to Hessian values.
- The production element path therefore couples the exact element Hessian with
  a different ownership/subdomain build than the SFD path.
- PETSc's `MatSetPreallocationCOO` modifies the input column index array
  in-place for MPIAIJ matrices (remapping off-process columns to local
  indices). The element→COO position lookup must therefore be built from the
  original adjacency-derived indices before this remapping occurs.

## 6. Nonlinear Solver Parity (Custom FEniCS)

`HyperElasticity3D_jax_petsc/solver.py` uses `tools_petsc4py.minimizers.newton()`.

The completed HE PETSc benchmark campaign used the following default nonlinear
policy:

- `tolf=1e-4`
- `tolg=1e-3`
- `tolg_rel=1e-3`
- `tolx_rel=1e-3`
- `tolx_abs=1e-10`
- `maxit=100`
- `require_all_convergence=True`
- `fail_on_nonfinite=True`
- `use_trust_region=True`
- PETSc trust-subproblem solve via `ksp_type=stcg`
- post trust-subproblem line search ON
- line-search interval `(-0.5, 2.0)`
- `linesearch_tol=1e-1`
- `trust_radius_init=0.5`
- `trust_shrink=0.5`, `trust_expand=1.5`
- `trust_eta_shrink=0.05`, `trust_eta_expand=0.75`
- `trust_max_reject=6`

The best backend-specific radii from the STCG sweep were:

- `fenics_custom`: `1.0`
- `jax_petsc_element`: `0.5`

The shared final default stays at `0.5` to preserve a single like-for-like
nonlinear policy across the PETSc backends.

See `TRUST_REGION_LINESEARCH_TUNING.md` and `final_HE_results.md` for the sweep
results and the finished benchmark report.

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

Historical profiles exposed via CLI:

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

Final completed benchmark default:

- `ksp_type=stcg`
- `pc_type=gamg`
- `gamg_threshold=0.05`
- `gamg_agg_nsmooths=1`
- `ksp_rtol=1e-1`
- `ksp_max_it=30`
- `pc_setup_on_ksp_cap=False`
- near-nullspace ON
- GAMG coordinates ON
- trust-region ON with post line search

## 8. Timing Instrumentation

`--save_history` stores per-Newton iteration:
- `t_grad`, `t_hess`, `t_ls`, `t_update`, `ksp_its`, convergence diagnostics

`--save_linear_timing` stores per linear callback:
- assembly total + assembly subparts (`p2p`, `hvp_compute`, `extraction`, `coo_assembly`)
- `setop_time`, `set_tolerances_time`, `pc_setup_time`, `solve_time`
- `linear_total_time`

## 9. Current Performance Findings

The historical progression is:

1. `sfd` was much too expensive because HE needs 63 colors.
2. the first analytical element implementation removed most assembly cost but
   still kept the old PETSc ownership/layout penalty.
3. the reordered overlap element path removed most of that layout penalty and
   is now the production `--assembly_mode element` implementation.

Current fine-case reference point (`level 4`, `step 1 / 96`, `np=32`,
`GMRES + GAMG`):

| Variant | Setup [s] | Step [s] | Assembly [s] | PC setup [s] | KSP solve [s] | Newton | KSP iters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FEniCS custom | 0.192 | 5.142 | 1.141 | 0.299 | 2.774 | 18 | 102 |
| Old JAX element path | 6.722 | 16.397 | 2.163 | 1.267 | 9.550 | 12 | 75 |
| Production reordered element path | 7.556 | 5.473 | 1.937 | 0.333 | 1.653 | 13 | 53 |

Key findings:
- `sfd` remains mainly a comparison/debug path for HE.
- The production element path has largely removed the old solve-side gap on the
  fine `step 1 / 96` case.
- Switching the production overlap gradient from per-element scatter to one
  whole-local gradient provided another small step-time win on top of the
  reordered ownership change.
- The remaining difference to FEniCS is no longer dominated by `pc_setup` or
  `ksp.solve`; it is now mostly in end-to-end setup/JIT cost.
- `block_xyz` is the default production reorder because it was the best fine
  `np=32` candidate in the distribution study, with `block_rcm` as the next
  alternative to recheck.

Full analysis:
- [investigation_jaxpetsc_performance_gap.md](../archive/investigation_jaxpetsc_performance_gap.md)
- [HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md](../archive/HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md)

## 10. Important References

Project context and run setup:
- `README.md`
- `instructions.md`
- `../archive/HyperElasticity3D_jax_petsc_prompt.md`

Design background:
- `jax_parallel_partitioning.md`
- `graph_coloring_implementation.md`

HE benchmarking and solver behavior:
- `../archive/results_HyperElasticity3D.md`
- `../archive/HE_ELEMENT_DISTRIBUTION_INVESTIGATION.md`
- `experiment_scripts/he_l3_24_comparison_jax_fenics_jaxpetsc.md`
- `experiment_scripts/he_step1_timing_breakdown_fenics_vs_jaxpetsc.md`
- `experiment_scripts/he_step1_timing_investigation_fix1.md`


### Advanced Command Line Options (Optimizations)

The `solve_HE_dof.py` entrypoint includes advanced flags geared toward peak high-deformation performance:

- `--assembly_mode element`: (Recommended). Configures JAX to evaluate exact analytical block Hessians evaluating via `@jax.jit(jax.vmap(compute_elem_hessians))`. Cuts matrix assembly latency by >50% compared to typical `sfd` computation.
- `--retry_on_failure`: Wraps the Newton solver in a robust fallback loop. Highly recommended for severe twisting boundaries. If gradients or line searches explode, dynamically expands line search alpha boundaries `[-0.5, 2.0]` and tightens KSP rules safely recovering from what would otherwise be a mathematical termination.
- `--pc_setup_on_ksp_cap`: Retains the same GAMG setup until the previous KSP
  hit `ksp_max_it`. This is **not** part of the final best-choice HE PETSc
  setting, which rebuilds the PC every Newton iteration by leaving this flag
  off.
