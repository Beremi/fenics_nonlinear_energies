# Topology JAX+PETSc Implementation

This note documents the current retained parallel topology implementation after
the repository cleanup.

## Current Layout

- CLI: `src/problems/topology/jax/solve_topopt_parallel.py`
- shared topology support:
  - `src/problems/topology/support/`
- distributed design/mechanics helpers:
  - `src/problems/topology/jax/parallel_support.py`
- shared PETSc minimizers and policies:
  - `src/core/petsc/minimizers.py`
  - `src/core/benchmark/repair.py`

The public report layer lives under `experiments/analysis/`. The canonical
current results are summarised in [docs/results/Topology.md](../results/Topology.md).

## Scope

The retained topology workflow is a distributed 2D topology optimisation path
with:

- PETSc-parallel mechanics
- distributed design energy and gradient evaluation
- staircase SIMP continuation in the outer loop

The goal of the maintained path is an operationally stable large-scale workflow,
not a perfectly rank-invariant topology design algorithm.

## Distributed Data Model

The implementation uses structured rectangular partitions:

- displacement unknowns are partitioned in PETSc-owned ranges
- the latent design field is also distributed
- both fields use owned-plus-ghost layouts for local kernels

The retained design path does not rebuild the full design vector on every rank
inside the hot loop. Instead it works from local-plus-ghost data and uses MPI
reductions only for scalar outputs.

## Mechanics Path

Current maintained mechanics policy:

- `fgmres + gamg`
- rigid-body near-nullspace enabled
- GAMG coordinates enabled
- `mechanics_ksp_rtol = 1e-4`
- `mechanics_ksp_max_it = 100`

The current implementation keeps the MPI shutdown fix that synchronized final
snapshot bookkeeping across ranks.

## Design Path

Current maintained design policy:

- latent-variable gradient descent
- `golden_adaptive` line search
- `tolg = 1e-3`
- `tolf = 1e-6`
- `design_maxit = 20`

The retained implementation also keeps the practical fail-safe that allows the
outer loop to continue when line search fails but the final gradient norm is
already sufficiently small.

## Continuation And Stop Logic

Current maintained defaults:

- `theta_min = 1e-6`
- `p_start = 1.0`
- `p_increment = 0.2`
- `continuation_interval = 1`
- `p_max = 10.0`

The fine-grid benchmark also uses a graceful stall stop once:

- `p >= 4.0`
- `dtheta <= 1e-6`
- `dtheta_state <= 1e-6`

This is operationally useful, but it is not exact target convergence; the
current results pages call that out explicitly.

## Known Limits

- final fine-grid states are not perfectly rank-invariant
- the strong-scaling report is end-to-end, not strict fixed-work scaling
- the smaller `192 x 96` direct-comparison case still does not complete for the
  maintained parallel path

## Related Docs

- [Topology problem overview](../problems/Topology.md)
- [Topology results](../results/Topology.md)
- archived design/reference notes:
  `archive/docs/reference/topology_description.md`
  `archive/docs/reference/topology_preconditioning.md`
