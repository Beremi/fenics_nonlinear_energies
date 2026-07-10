# EXP-DIST-001: Distributed Equivalence

## Research Question

At an identical canonical state and discrete functional, do replicated and
rank-local problem construction, serial and distributed assembly, and one-,
two-, and four-rank ownership layouts produce equivalent algebraic objects?

## Claim-Aligned Design

The selected paper retains two controlled questions:

1. with procedural rank-local construction, point-to-point overlap, owned-row
   COO assembly, and the element route held fixed, does changing the MPI rank
   count from one to two to four preserve the canonical hyperelastic algebraic
   objects; and
2. with the Plasticity3D mesh, prescribed state, ownership contract, and rank
   count held fixed, do the element, colored-recovery, and constitutive routes
   preserve the canonical gradient and tangent actions at one, two, and four
   ranks?

Mesh-source, replicated/rank-local construction, all-gather/point-to-point
distribution, and global/owned-row COO ablations are not required because the
manuscript makes no causal, timing, or memory claim about those factors. A
future claim about any such factor requires a new one-factor-at-a-time protocol
and cannot reuse the current rank-count result as evidence.

## Fixed-State Algebraic Gate

Store one canonical state and deterministic direction. Before any nonlinear
solve, compare:

- coordinates, element connectivity, boundary masks, affine lift, constrained
  and free-DOF maps, and canonical permutation;
- scalar energy, canonical residual, full small assembled matrix, and HVP;
- structural sparsity, ownership, ghost dependencies, and branch map where
  applicable;
- Riesz primal/dual norms and the true residual of any norm solve.

Use exact equality for integer topology and identifiers, scale-aware FP64
tolerances for floating geometry, and relative derivative tolerances initially
targeted at `1e-8`. Freeze final tolerances before the publication run.

## Solved-Endpoint Gate

Only after fixed-state equality passes, the separate route Tier-B and stopping
campaigns may compare solved endpoints under the same stopping and solver
policy. They compare independent final residuals, weighted full state, energy,
and declared observables. Partition-dependent nonlinear trajectories are
allowed; the endpoint tolerance must be calibrated against a tighter solve.

## Required Outputs

- canonical states and directions;
- per-rank construction/ownership manifests and hashes;
- fixed-state energies, residuals, matrices, HVPs, and branch maps;
- successful, failed, capped, and timed-out run records;
- endpoint state/residual/observable comparisons and memory by rank/node.

No timing or memory advantage may be interpreted until both algebraic gates
pass. A fixed-work endpoint with maximum-iteration termination is a diagnostic,
not a convergence result.

## Controlled Local Pilot (2026-07-10)

The implementation pilot in
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-DIST-001/`
isolates the rank-count factor for the canonical hyperelasticity case. Both
runs use the same procedural (P_1) mesh definition, rank-local construction,
block-XYZ canonical ordering, point-to-point overlap exchange, owned-row local
COO assembly, stored twist state, deterministic direction, and MUMPS LU
correction solve. Only the ownership partition changes between one and two MPI
ranks.

The fixed-state gate passed at the prespecified pilot tolerances:

- coordinates, connectivity, free-DOF map, right-boundary mask, affine lift,
  canonical state, canonical direction, and CSR structure matched exactly;
- the relative energy discrepancy was $8.67\times10^{-19}$;
- the residual and assembled-matrix discrepancies were zero at stored FP64
  precision;
- the relative matrix-action discrepancy was $2.24\times10^{-16}$;
- the relative Newton-correction discrepancy was $2.33\times10^{-16}$; and
- independently recomputed linear-system residuals were $1.78\times10^{-15}$
  and $1.84\times10^{-15}$ relative on one and two ranks, respectively.

This result resolves only the controlled one-versus-two-rank fixed-state pilot.
The required four-rank extension and the separately gated nonlinear endpoints
remain outstanding. The worktree was dirty and
both rank counts shared one workstation. Consequently, phase timings in the
pilot report are diagnostic only and must not support a performance or memory
claim. Publication evidence requires a clean-commit rerun with pinned placement
and prespecified repetitions.

## Managed Rank-Count Publication Run

The managed local producer is prepared to execute the rank-count factor at
exactly one, two, and four ranks:

```bash
./.venv/bin/python experiments/runners/run_hyperelasticity_distribution_equivalence.py \
  --run-kind publication \
  --output-dir artifacts/reproduction/<clean-campaign>/_publication_staging/EXP-DIST-001
```

Publication mode fails unless the Git worktree is clean, the output directory
is fresh, and the frozen level, state angle, repetition count, solver
tolerances, and comparison tolerances retain their preregistered values. It
writes three strict run records and requires both the one-versus-two and
one-versus-four comparisons to pass. The aggregate comparison is the
conjunction of all gates and the maximum error across the two comparisons; an
omitted or failed rank cannot be hidden by aggregation.

This command has been prepared but not executed as part of the implementation
change. A successful clean run resolves only the required rank-count question
at the fixed state. The calibrated nonlinear solved-endpoint gate remains a
separate route/stopping task, and the local timings remain inadmissible for a
performance or scaling claim.

## Separate Distributed Colored-Recovery Gate

The hyperelastic rank-count producer above deliberately uses the element-
Hessian route, so it cannot establish distributed colored-recovery
correctness. A second local correctness campaign is prepared in
`experiments/runners/run_local_distributed_route_verification.py`. Its frozen
matrix contains 12 blocks:

- element degrees (P_1) and (P_2);
- prescribed elastic and mixed-branch states;
- one, two, and four ranks; and
- element AD, colored sparse finite differences, and constitutive AD in a
  balanced route order.

Every route stores the canonical state, gradient, four deterministic tangent
actions, branch diagnostics, and per-rank ownership summary. Feasible
one-rank cases additionally store direct CSR matrices; the distributed cases
compare the four actions and ownership partitions across ranks. The strict
adjudicator fails on a missing route/rank, incomplete ownership, changed
branch diagnostics, nonidentical canonical state, or an action/matrix error
outside the frozen absolute-and-relative gate.

The scheduler-free review command writes an immutable 36-process plan without
executing route processes:

```bash
HEAD=$(git rev-parse HEAD)
./.venv/bin/python experiments/runners/run_local_distributed_route_verification.py \
  --run-kind publication \
  --expected-commit "$HEAD" \
  --out-root artifacts/reproduction/<clean-campaign>/EXP-DIST-001-colored-review
```

The review root is preparation-only and cannot be reused for execution. A
publication run creates a second fresh root and repeats the frozen plan before
launching it:

```bash
HEAD=$(git rev-parse HEAD)
LOCAL_DISTRIBUTED_RUN_CONFIRMED=YES \
./.venv/bin/python experiments/runners/run_local_distributed_route_verification.py \
  --run-kind publication \
  --expected-commit "$HEAD" \
  --out-root artifacts/reproduction/<clean-campaign>/EXP-DIST-001-colored \
  --execute
```

The driver rejects a dirty worktree, a nonmatching full commit, or an output
root outside `artifacts/reproduction`. Its measurements are correctness
diagnostics and cannot support a timing or scaling claim.
