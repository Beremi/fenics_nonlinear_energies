# JAX Topology Current State

Date: 2026-03-12

This note records the working configuration we are carrying forward for the
next implementation phase. It supersedes the exploratory comparison reports
that were used to get here.

## Chosen Solver Path

- Mechanics subproblem:
  - nonlinear method: existing Newton / trust-region mechanics solve
  - linear/preconditioning path: PETSc `GAMG`
  - elasticity metadata: 2D rigid-body near-nullspace enabled
- Design subproblem:
  - nonlinear method: gradient descent
  - line search: adaptive golden-section search
  - carried state: the absolute value of the last accepted `alpha` is reused as
    the horizon scale `a` in the next outer design solve
  - adaptive bracket: `[0, 2a]`
- Outer continuation:
  - staircase SIMP schedule
  - `p_start = 1.0`
  - `p_increment = 0.5`
  - `continuation_interval = 20`
  - `p_max = 4.0`

## Default Working Settings In Code

These are now the intended defaults in
`topological_optimisation_jax/solve_topopt_jax.py`:

- `mechanics_solver_type = "petsc_gamg"`
- `mechanics_use_near_nullspace = True`
- `design_nonlinear_method = "gd_golden_adaptive"`
- `design_gd_adaptive_nonnegative = True`
- `linesearch_tol = 1e-1`

The older `generate_report_assets.py` benchmark is intentionally pinned to the
historical pure-JAX / Newton reference settings so that report remains
reproducible.

## Final Benchmark Snapshot Used For Direction

Fine benchmark: `192 x 96`

Reference Newton design solve:

- result: `completed`
- outer iterations: `172`
- wall time: `360.029 s`
- final compliance: `4.193275`
- final volume fraction: `0.400892`

Adaptive GD, symmetric bracket `[-2a, 2a]`:

- result: `completed`
- outer iterations: `160`
- wall time: `298.185 s`
- total design line-search evaluations: `9182`
- final compliance: `4.153689`
- final volume fraction: `0.399862`

Adaptive GD, positive bracket `[0, 2a]`:

- result: `completed`
- outer iterations: `148`
- wall time: `298.974 s`
- total design line-search evaluations: `8436`
- final compliance: `4.138879`
- final volume fraction: `0.400590`

The positive-only bracket is the selected design-side setting.

## Why This Is The Next-Step Configuration

- Mechanics is the naturally parallel block and already has a viable PETSc
  multigrid path.
- Design does not currently benefit from forcing a Hessian-based linear solve.
  The adaptive GD path is simpler, robust on the full benchmark, and parallel
  enough for the next stage because it only needs energy/gradient evaluations.
- This keeps the implementation compact while still matching the long-term
  direction toward distributed execution.

## Next Step

Focus new work on the parallel mechanics/GAMG path and keep the design block on
the GD formulation unless a clearly better distributed design solve appears.
