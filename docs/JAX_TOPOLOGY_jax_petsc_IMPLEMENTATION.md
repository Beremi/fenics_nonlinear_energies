# JAX Topology JAX+PETSc Implementation Notes

This document describes the retained parallel topology implementation, the
solver policy we consider current, and the design choices that survived the
cleanup pass.

## Current Layout

The retained topology-parallel stack is intentionally small.

- CLI / driver:
  - `topological_optimisation_jax/solve_topopt_parallel.py`
- distributed mechanics and design support:
  - `topological_optimisation_jax/parallel_support.py`
- shared PETSc minimizers:
  - `tools_petsc4py/minimizers.py`
- retained report tooling:
  - `topological_optimisation_jax/generate_parallel_full_report.py`
  - `topological_optimisation_jax/generate_parallel_scaling_stallstop_report.py`
- retained tests:
  - `topological_optimisation_jax/test_parallel_topopt_smoke.py`
  - `topological_optimisation_jax/test_petsc_gradient_descent_minimizers.py`

Exploratory report generators, mechanics benchmark helpers, and abandoned
mechanics cutout/floating-DOF experiments were intentionally dropped.

## Scope And Goal

The solver is a distributed 2D topology optimisation path where:

- mechanics is the PETSc-parallel block
- design uses distributed energy and gradient evaluation
- the outer loop performs volume-controlled continuation in SIMP `p`

The retained objective is a stable end-to-end parallel workflow, not a fully
rank-invariant or theoretically optimal design algorithm.

## Distributed Data Model

The implementation uses a structured partition of the rectangle:

- displacement unknowns are partitioned in PETSc-owned ranges
- the latent design field is also distributed
- both fields use owned-plus-ghost layouts for local kernels

Key pieces in `parallel_support.py`:

- `StructuredTopologyPartition`
  - geometric/domain partition metadata
  - local element/node views
  - force data and element metrics
- `OwnedGhostLayout`
  - MPI ownership range for the field
  - local owned/ghost reconstruction
  - local exchange for owned-plus-ghost evaluation
- `TopologyMechanicsAssembler`
  - distributed mechanics matrix assembly and KSP solve
- `TopologyDesignEvaluator`
  - distributed design energy / gradient / volume evaluation

The retained design path does **not** rebuild the full global design vector on
every rank for the hot energy/gradient loop. It works from local+ghost data and
uses reductions for scalar outputs.

## Mechanics Path

Current mechanics policy:

- PETSc KSP: `fgmres`
- PETSc PC: `gamg`
- near-nullspace:
  - rigid-body near-nullspace enabled by default
- coordinates:
  - GAMG coordinates enabled by default
- default tolerances:
  - `mechanics_ksp_rtol = 1e-4`
  - `mechanics_ksp_max_it = 100`

Retained behavior:

- mechanics fallback retries remain available in the assembler path
- the shutdown-hang fix from the final snapshot bookkeeping is retained

Deliberately dropped:

- projector/shell-based masked mechanics operators
- reduced active-DOF cutout solves
- floating-DOF elimination experiments

Those branches were useful diagnostically, but they were not kept because they
did not produce a usable MPI production path.

## Design Path

Current design policy:

- nonlinear method:
  - gradient descent on the latent design vector
- supported default line search:
  - `golden_adaptive`
- line-search tolerance:
  - interpreted relative to the active bracket bound
- default tolerances:
  - `tolg = 1e-3`
  - `tolf = 1e-6`
  - `design_maxit = 20`

Important retained logic:

- accepted design-step fail-safe:
  - if golden-section search fails but the last gradient norm is already small,
    the outer loop is allowed to continue
- recent max-it gate:
  - continuation in `p` is blocked when recent outer iterations show repeated
    mechanics/design max-it hits

Still present in `tools_petsc4py/minimizers.py`, but not the recommended
default:

- `golden_linf`
- `golden_gamma_beta`

These remain because they are still exercised by retained minimizer tests, but
the topology docs and scaling campaign treat `golden_adaptive` as the supported
default mode.

## Continuation And Convergence Policy

Current retained continuation policy:

- `theta_min = 1e-6`
- `p_start = 1.0`
- `p_increment = 0.2`
- `continuation_interval = 1`
- `p_max = 10.0`

Outer stopping rules:

- ordinary completion when:
  - `p >= p_max`
  - volume residual is small enough
  - design/compliance changes are small enough
- graceful stall completion when:
  - `p >= 4.0`
  - `dtheta <= 1e-6`
  - `dtheta_state <= 1e-6`

The graceful stall stop is operationally useful, but it is not the same as true
target convergence. On fine meshes it can stop in a visibly under-filled state.

## Reporting And Reproduction

The retained report generators now default to ignored scratch output under:

- `topological_optimisation_jax/report_runs/`

Curated final benchmark/scaling assets live in:

- `docs/assets/jax_topology_parallel/final_benchmark/`
- `docs/assets/jax_topology_parallel/scaling/`

This keeps future experiments from polluting the repo root or reintroducing the
old `report_assets_parallel*` sprawl.

## Known Limits

- final states are not perfectly rank-invariant under the current stall-stop and
  continuation logic
- the scaling report is an end-to-end strong-scaling study, not a pure
  fixed-work strong-scaling measurement, because termination points vary by rank
- the current fine-grid benchmark still stops on a graceful stall criterion well
  before exact target volume is recovered

## Non-Goals After Cleanup

The following are intentionally not part of the retained topology codebase:

- mechanics cutout / floating-DOF masking
- projected-shell mechanics operators
- reduced active-DOF mechanics solves
- exploratory benchmark markdowns in the repo root
- large intermediate `report_assets_parallel*` directories
