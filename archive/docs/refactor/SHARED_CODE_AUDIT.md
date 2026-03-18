# Shared Code Audit

Date: 2026-03-15

## Goal

This note looks past the folder layout and focuses on the concrete solver
implementations:

- what is genuinely problem-specific,
- what is repeated backend scaffolding,
- what is already partly shared,
- and what can realistically be extracted into shared modules.

## Executive Summary

The code is more shared in structure than it looks, but that shared structure is
not consistently extracted yet.

My high-level read is:

- `pLaplace` and `GinzburgLandau` are heavily duplicated within the same backend
  family
- `HyperElasticity` uses many of the same backend patterns, but adds genuine
  complexity: vector DOFs, nullspaces, load stepping, and rotating boundary data
- topology is architecturally separate, but still contains internal duplication
  between its serial and parallel drivers

The biggest opportunity is not “make one solver for every problem”.
The biggest opportunity is:

- extract backend scaffolding,
- keep physics/problem definitions local,
- and let each problem provide only the small problem-specific hooks.

## What Is Already Shared Well

### 1. Serial and PETSc minimizers

Already shared:

- `tools/minimizers.py`
- `tools_petsc4py/minimizers.py`

This is the right pattern:

- common nonlinear algorithm in one place
- problem code passes energy/gradient/Hessian callbacks

### 2. Generic JAX+PETSc global-coloring / local-coloring assembler layer

Already shared:

- `tools_petsc4py/jax_tools/parallel_assembler.py`
- `tools_petsc4py/dof_partition.py`

This is the strongest existing abstraction in the repo.

`pLaplace`, `GinzburgLandau`, and `HyperElasticity` already use the same idea:

- a generic parallel assembly engine
- plus a small `JaxProblemSpec`

That is exactly the pattern the rest of the repo should move toward.

### 3. GL already reuses the pLaplace reordered element assembler structure

Concrete sign:

- `GinzburgLandau2D_jax_petsc/reordered_element_assembler.py`
  subclasses
- `pLaplace2D_jax_petsc/reordered_element_assembler.py`

That means the scalar overlap assembler is already acting like a shared base,
even though it still lives in a problem-specific file with a problem-specific
name.

## Where The Concrete Duplication Is Strongest

## A. Scalar FEniCS custom solvers

Files:

- `pLaplace2D_fenics/solver_custom_newton.py`
- `GinzburgLandau2D_fenics/solver_custom_newton.py`

These two are extremely close.

Observed:

- `git diff --no-index --stat` reports only `44` changed lines between them
- both files are about `500` lines long
- they share the same overall control flow almost line-for-line

Shared structure:

- mesh loading and function-space creation
- Newton callback wiring
- PETSc KSP setup
- optional trust-region KSP handling
- linear timing collection
- repair/retry logic
- result flattening
- `run_level(...)` wrapper pattern

Problem-specific part:

- the mesh path and initial guess
- the energy form `J_energy`
- default linear-solver policy
- GAMG-coordinate details

Conclusion:

These should not remain as two separate full-size implementations.

Recommended extraction:

- a shared scalar FEniCS custom-Newton driver base
- problem-specific modules should provide:
  - `load_mesh(...)`
  - `make_forms(...)`
  - `make_initial_guess(...)`
  - backend defaults / metadata

Suggested target:

- `src/core/fenics/scalar_custom_newton.py`
- `src/core/fenics/result_utils.py`

## B. Scalar JAX+PETSc solver drivers

Files:

- `pLaplace2D_jax_petsc/solver.py`
- `GinzburgLandau2D_jax_petsc/solver.py`

Observed:

- `git diff --no-index --stat` reports only `64` changed lines
- both files are about `600` lines long

Shared structure:

- `PROFILE_DEFAULTS`
- `_resolve_linear_settings`
- `_pc_options`
- `_needs_repair`
- `_sum_step_linear`
- `_assemble_time`
- `_build_gamg_coordinates`
- `run(args)` skeleton
- repair/retry path
- result flattening
- `run_level(...)` adapter

Problem-specific part:

- which mesh loader is used
- which assembler class is chosen
- scalar tolerances and profile defaults
- scalar energy semantics

Conclusion:

These two should be backed by one shared scalar JAX+PETSc driver.

Recommended extraction:

- a generic scalar JAX+PETSc nonlinear driver
- problem modules provide:
  - mesh/problem loader
  - assembler class
  - backend defaults
  - result metadata

Suggested target:

- `src/core/petsc/scalar_problem_driver.py`

## C. Scalar mesh/problem-data loaders

Files:

- `pLaplace2D_jax/mesh.py`
- `GinzburgLandau2D_jax/mesh.py`
- `pLaplace2D_petsc_support/mesh.py`

Observed:

- `pLaplace2D_jax/mesh.py` vs `GinzburgLandau2D_jax/mesh.py`
  differ by only `14` changed lines
- `pLaplace2D_jax/mesh.py` vs `pLaplace2D_petsc_support/mesh.py`
  differ by only `8` changed lines

Shared structure:

- HDF5 loading
- adjacency reconstruction
- parameter dictionary construction
- return shape of `(params, adjacency, u_init)`

Problem-specific part:

- file path
- initial guess
- parameter keys present in the HDF5 file

Conclusion:

These should use a shared HDF5 problem-data loader with thin problem adapters.

Suggested target:

- `src/core/problem_data/hdf5.py`
- `src/problems/plaplace/support/data.py`
- `src/problems/ginzburg_landau/support/data.py`

## D. HyperElasticity mesh loaders are also duplicated

Files:

- `HyperElasticity3D_jax/mesh.py`
- `HyperElasticity3D_petsc_support/mesh.py`

Observed:

- both load the same HDF5 source
- both reconstruct adjacency
- both compute free-DOF data
- both compute the elasticity near-nullspace / elastic kernel

The PETSc-support version is more complete, but the JAX version clearly
duplicates the same ideas.

Conclusion:

`HyperElasticity` should have one canonical mesh/problem-data loader, with
conversion helpers for JAX and PETSc consumers.

Suggested target:

- `src/problems/hyperelasticity/support/data.py`

## E. HE nullspace and GAMG coordinate helpers are repeated

Concrete duplication:

- `build_nullspace(...)` appears in both
  - `HyperElasticity3D_fenics/solver_custom_newton.py`
  - `HyperElasticity3D_fenics/solver_snes.py`
- `_build_gamg_coordinates(...)` exists in
  - `pLaplace2D_jax_petsc/solver.py`
  - `GinzburgLandau2D_jax_petsc/solver.py`
  - `HyperElasticity3D_jax_petsc/solver.py`
- similar coordinate/nullspace logic also appears in experiment scripts

Conclusion:

These helpers should live in shared PETSc/FEniCS utility modules, not inside
individual solver files.

Suggested target:

- `src/core/petsc/gamg.py`
- `src/core/fenics/nullspace.py`

## F. CLI wrappers are repeated

Files:

- `pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py`
- `GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py`
- `pLaplace2D_jax_petsc/solve_pLaplace_dof.py`
- `GinzburgLandau2D_jax_petsc/solve_GL_dof.py`
- `HyperElasticity3D_jax_petsc/solve_HE_dof.py`

Observed:

- same parser patterns
- same thread-environment setup
- same `run(args)` dispatch structure

Conclusion:

Parser construction and environment setup should be shared at the backend
family level.

Suggested target:

- `src/core/cli/threading.py`
- `src/core/cli/petsc_profiles.py`
- backend-specific parser factories

## G. Result flattening and repair logic is repeated

Repeated helper patterns:

- `_needs_repair`
- `_sum_step_linear`
- `_assemble_time`
- `_flatten_result`
- retry-on-failure logic
- timing record shaping

This appears in:

- scalar FEniCS custom solvers
- scalar JAX+PETSc solvers
- analogous but slightly richer form in HE drivers

Conclusion:

A shared result-shaping / retry-policy layer would remove a lot of repeated
driver code.

Suggested target:

- `src/core/benchmark/results.py`
- `src/core/benchmark/repair.py`

## H. Topology has its own internal duplication

Files:

- `topological_optimisation_jax/solve_topopt_jax.py`
- `topological_optimisation_jax/solve_topopt_parallel.py`

Shared helpers currently duplicated:

- `_constitutive_plane_stress`
- `staircase_p_step`
- `_relative_state_change`
- `_message_is_converged`

The serial and parallel topology drivers are still structurally different, so I
would not force them into one solver file.
But these shared policy helpers should absolutely move to a common topology
support module.

Suggested target:

- `src/problems/topology/support/policy.py`

## What Should Stay Problem-Specific

Not everything should be shared.

The following should remain local to each problem family:

- `jax_energy.py` physics kernels
- FEniCS form definitions
- problem-specific boundary conditions
- problem-specific initial-guess logic
- hyperelastic rotating-boundary handling
- topology design/mechanics coupling logic

These are the places where the mathematical problem actually differs.

## What Should Stay Backend-Specific

Also worth keeping separate:

- FEniCS SNES drivers versus custom PETSc-Newton drivers
- pure-JAX serial solver logic versus JAX+PETSc distributed logic
- topology support, which is a two-field staggered optimisation workflow rather
  than a direct copy of the scalar/vector energy benchmarks

The goal should be shared scaffolding, not one giant universal solver.

## Recommended Shared Extraction Layers

Here is the concrete layering I would aim for.

### 1. Problem-data layer

Shared responsibilities:

- HDF5 loading
- adjacency reconstruction
- base parameter normalization

Target:

- `src/core/problem_data/hdf5.py`

Problem adapters:

- `src/problems/*/support/data.py`

### 2. Backend utility layer

Shared responsibilities:

- ghost update helpers
- nullspace builders
- GAMG coordinate helpers
- PETSc profile handling

Target:

- `src/core/fenics/`
- `src/core/petsc/`

### 3. Backend driver layer

Shared responsibilities:

- scalar FEniCS custom-Newton control flow
- scalar JAX+PETSc driver control flow
- load-step driver pattern for HE-like trajectories
- result flattening and retry policy

Targets:

- `src/core/fenics/scalar_custom_newton.py`
- `src/core/petsc/scalar_problem_driver.py`
- `src/core/petsc/load_step_driver.py`
- `src/core/benchmark/results.py`

### 4. Problem hook layer

Each problem should provide small hook modules:

- forms / integrands
- initial guess
- mesh path / data adapter
- backend defaults

This is the same pattern already used successfully by
`tools_petsc4py/jax_tools/parallel_assembler.py`.

## Recommended Refactor Priority

If this is done incrementally, I would do it in this order.

### Priority 1. Extract scalar shared scaffolding

Why first:

- highest duplication
- lowest conceptual risk
- already proven by the similarity of `pLaplace` and `GL`

Do first:

- scalar FEniCS custom driver
- scalar JAX+PETSc solver driver
- scalar HDF5 mesh/problem loader

### Priority 2. Extract generic reordered element assembler base

Why second:

- the code is already halfway there
- GL already treats the pLaplace assembler as a de facto base class
- HE duplicates the same layout ideas in a different file

Do next:

- move the overlap/reordering/scatter infrastructure to a shared base
- keep only local energy kernels and block-size specifics in problem modules

### Priority 3. Extract PETSc/FEniCS support helpers

Do next:

- nullspace utilities
- GAMG coordinate helpers
- result/timing helpers
- parser/thread setup helpers

### Priority 4. Clean topology internal sharing

Do later:

- move policy helpers shared by serial and parallel topology drivers

This is worthwhile, but less urgent than the scalar/backend duplication.

## Concrete “Shared Place” Candidates

If I had to name the most useful shared modules right now, they would be:

- `src/core/problem_data/hdf5.py`
- `src/core/fenics/scalar_custom_newton.py`
- `src/core/fenics/nullspace.py`
- `src/core/petsc/scalar_problem_driver.py`
- `src/core/petsc/load_step_driver.py`
- `src/core/petsc/gamg.py`
- `src/core/petsc/reordered_element_base.py`
- `src/core/benchmark/results.py`
- `src/core/cli/threading.py`
- `src/problems/topology/support/policy.py`

## Final Assessment

How much is actually shared?

- High sharedness:
  - scalar `pLaplace` and `GinzburgLandau` within the same backend
- Medium sharedness:
  - `HyperElasticity` with the same backend families
- Low sharedness:
  - topology versus the other problem families

So the best refactor direction is:

- share by backend skeleton first,
- not by forcing all problems into one abstraction at once.

That will give you real cleanup without making the code artificially generic.
