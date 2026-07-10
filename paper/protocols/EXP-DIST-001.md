# EXP-DIST-001: Distributed Equivalence

## Research Question

At an identical canonical state and discrete functional, do replicated and
rank-local problem construction, serial and distributed assembly, and one-,
two-, and four-rank ownership layouts produce equivalent algebraic objects?

## Factorized Design

The experiment must vary one factor at a time:

1. mesh source: HDF5 versus procedural construction;
2. construction: replicated versus rank-local;
3. distribution: all-gather versus point-to-point overlap;
4. assembly: global COO versus owned-row local COO;
5. ranks: 1, 2, and 4.

The current reviewer-gap pair changes mesh source, construction, distribution,
and assembly together. It is a bundle smoke test, not an isolated correctness
experiment.

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

Only after fixed-state equality passes, solve with the same stopping and solver
policy. Compare independent final residuals, weighted full state, energy, and
physical observables. Partition-dependent nonlinear trajectories are allowed;
the endpoint tolerance must be calibrated against a tighter solve.

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
It does not complete the full factorized matrix above: four ranks, independent
mesh-source/construction/distribution/assembly changes, and the calibrated
nonlinear solved-endpoint gate remain outstanding. The worktree was dirty and
both rank counts shared one workstation. Consequently, phase timings in the
pilot report are diagnostic only and must not support a performance or memory
claim. Publication evidence requires a clean-commit rerun with pinned placement
and prespecified repetitions.
