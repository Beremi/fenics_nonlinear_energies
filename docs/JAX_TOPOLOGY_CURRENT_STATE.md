# JAX Topology Current State

Date: 2026-03-15

This note records the topology configuration we are carrying forward after the
JAX and JAX+PETSc parallel implementation work.

## Recommended Active Path

- Historical/reference path:
  - `topological_optimisation_jax/solve_topopt_jax.py`
  - kept as the serial pure-JAX reference and for comparison with older reports
- Recommended active path:
  - `topological_optimisation_jax/solve_topopt_parallel.py`
  - distributed JAX+PETSc topology solve
  - this is the topology implementation to treat as current

## Chosen Default Solver Policy

The code defaults for the parallel path are normalized to the stable solver
policy, not to the full fine-grid benchmark recipe.

- Mechanics:
  - PETSc `fgmres + gamg`
  - rigid-body near-nullspace enabled
  - `mechanics_ksp_rtol = 1e-4`
  - `mechanics_ksp_max_it = 100`
  - fallback mechanics retries remain enabled
- Design:
  - distributed gradient descent
  - supported default line search: `golden_adaptive`
  - relative line-search tolerance against the active bracket bound enabled
  - `design_maxit = 20`
  - `tolg = 1e-3`
  - line-search fail-safe retained: a failed golden search is accepted only when
    the last gradient norm is already small enough
- Continuation / outer logic:
  - `theta_min = 1e-6`
  - staircase continuation with `p_increment = 0.2`
  - continuation interval `= 1`
  - `p_max = 10.0`
  - max-it gate over the recent outer history retained
  - graceful stall stop retained with:
    - `stall_theta_tol = 1e-6`
    - `stall_p_min = 4.0`

## What Is Intentionally Not The Default State

These are documented benchmark choices, not generic CLI defaults:

- fine benchmark mesh `768 x 384`
- `32` MPI ranks
- very large outer iteration caps used for long benchmark campaigns
- exploratory line-search variants:
  - `golden_linf`
  - `golden_gamma_beta`
- abandoned mechanics-cutout / floating-DOF experiments

## Current Reporting Convention

- Implementation details:
  - [`JAX_TOPOLOGY_jax_petsc_IMPLEMENTATION.md`](JAX_TOPOLOGY_jax_petsc_IMPLEMENTATION.md)
- Final benchmark and scaling report:
  - [`final_JAX_TOPOLOGY_parallel_results.md`](final_JAX_TOPOLOGY_parallel_results.md)

## Current Recommendation

Use the parallel JAX+PETSc solver as the maintained topology path, keep the
pure-JAX solver as a reference implementation, and treat the final benchmark
report as the authoritative summary of the current stable behavior and scaling.
