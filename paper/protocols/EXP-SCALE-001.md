# EXP-SCALE-001: Fixed-Policy Distributed Viability on Karolina

## Status and decision

**Prepared, not submitted.** This is a bounded distributed-viability campaign,
not an authorization to use the expired/expiring allocation record. The required
series is Hyperelasticity level 5. Plasticity3D is optional and must remain a
separate result block.

## Scientific question

With the discrete problem, derivative route, PMG hierarchy, coarse policy,
globalization, stopping contract, and output scope fixed, how do setup, first
nonlinear-step solve time, iteration counts, endpoint, and memory change from
one to eight Karolina CPU nodes?

## Required Hyperelasticity series

- Fixed work: Hyperelasticity `L5`, first load step of a declared 24-step path.
- Derivatives: element AD with rank-local COO assembly.
- Construction: procedural rank-local mesh, point-to-point overlap exchange.
- Linear policy: STCG and a fixed level-3 PMG hierarchy; Chebyshev/Jacobi
  smoothing (two steps) and Hypre coarse solve with the same options at every
  rank. No rank-dependent redundant groups or automatic coarse-level changes.
- Globalization: Armijo plus trust-region acceptance and post-subproblem line
  search with identical parameters at all ranks.
- Rank points: 128, 256, 512, and 1024 ranks on 1, 2, 4, and 8 nodes,
  respectively.
- Repetitions: five independent processes per rank point.
- Resource ceiling: 22.50 node-hours.

This fixed first-step series supports a distributed-kernel/viability statement,
not a completed nonlinear load-path scaling claim. A completed-path claim would
require a separately approved campaign.

## Optional Plasticity3D series

- `P4(L1)`, 24-point rule, glued bottom, strength-reduction factor 1.55.
- Constitutive AD, rank-local assembly, fixed `same_mesh_p4_p2_p1` Hypre PMG,
  and identical stopping/globalization at every rank.
- Numerically checked reference-elastic-energy stopping with metric-current-state
  correction normalization, GMRES/Hypre Riesz solves at relative tolerance
  $10^{-10}$ and a recomputed $10^{-8}$ true-residual gate. MUMPS inertia must
  numerically check positive definiteness of the glued free-space metric operator at every rank point.
- 1, 2, and 4 nodes at 128 ranks per node.
- Five repetitions, each with a 30-minute process ceiling.
- Separate optional ceiling: 17.50 node-hours.

Enable these rows only after `EXP-DERIV-001`, `EXP-STOP-001`, and the relevant
`EXP-DISC-001` gate pass. Do not merge them with Hyperelasticity scaling.

## Measurements and checks

For every repetition record:

- setup, mesh/distribution, compilation, assembly, communication, PMG/coarse,
  Krylov, globalization, and total wall times;
- nonlinear and Krylov iteration counts, accepted/rejected steps, KSP reasons,
  and every terminal stopping metric;
- full exported endpoint state, energy, and problem observables;
- rank placement, local elements, owned/overlap DOFs, and imbalance;
- per-rank `ru_maxrss`, Slurm MaxRSS/MaxVMSize, and node-level memory where
  available. Summed RSS is labeled only as an upper-bound diagnostic, never
  exact aggregate memory.

Before timing analysis, compare every multi-node state to the one-node state.
The state-norm, energy, observable, residual, and iteration-count tolerances must
be declared in the analysis manifest. A row that reaches different accuracy or
an inequivalent endpoint is not used for speedup.

Report median time, an interval, speedup `T_1/T_p`, and efficiency
`T_1/(p T_p)` with the rank/node basis explicit. Use the adjective `scalable`
only if a threshold was declared before analysis and the measured efficiency
passes it. Otherwise report numerical efficiency without the adjective.

The machine-readable gates are frozen in
[`EXP-SCALE-001-analysis-contract.json`](EXP-SCALE-001-analysis-contract.json).
Analysis uses `experiments/analysis/analyze_exp_scale_001.py`; settled Slurm
accounting is collected after job completion with
`experiments/analysis/collect_slurm_accounting.py`. The collector is offline by
default and contacts `sacct` only with its explicit live-query flag.

## Scheduling and provenance

One- and two-node rows use `qcpu_exp`; four- and eight-node rows use `qcpu`.
Every node carries 128 one-thread MPI ranks with `block:block`, explicit
`map_cpu:0..127`, and local NUMA memory binding. No job requests exclusive or
explicit memory resources. Required and optional totals remain below the
100-node-hour agent guard.

Every publication row requires the reviewed matrix hash, a clean commit,
revalidated account/QoS/allocation, environment capture, per-run command,
Slurm stdout/stderr, and `sacct` output. Failure, timeout, and memory-censoring
records are retained.

## Terminal decisions

- **FIXED-POLICY VIABILITY:** endpoint/accuracy gates pass and all policy fields
  are identical; report measured scaling without extrapolation.
- **TUNED DEPLOYMENT ONLY:** any rank-dependent policy change is necessary;
  move those results to a separately labeled tuned series.
- **CENSORED:** a rank point fails time, memory, or solver gates; retain the row
  and bound the claim to successful points.
- **INVALID:** hidden policy, output, stopping, or endpoint changes are present.
