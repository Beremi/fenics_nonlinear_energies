# Publication Revision Execution Report and Remaining Action Plan

Last updated: 2026-07-10.

## 1. Purpose and decision

This report records what was implemented, what was actually executed, what the
current evidence establishes, and the exact order of the remaining work.  It is
the operational companion to the full dependency plan in
[`publication_action_plan.md`](publication_action_plan.md).  The latter remains
the authoritative scientific checklist; this report is the shorter handoff for
finishing the revision without promoting diagnostic results into publication
claims.

The recommended primary venue is the *SIAM Journal on Scientific Computing* or
a comparable scientific-computing journal.  The current work does not yet
contain a new optimization algorithm or the KKT/convergence theory expected for
a central *SIAM Journal on Optimization* contribution.  The venue rationale and
current novelty boundary are recorded in
[`venue_and_contribution_decision.md`](venue_and_contribution_decision.md).

The working contribution is deliberately narrow:

> For one fixed constrained discrete finite-element functional, identify the
> assumptions under which element AD, constitutive AD, and colored sparse
> Hessian recovery are equivalent; verify those assumptions and derivatives;
> and determine their CPU/MPI crossover from paired, equal-accuracy,
> factorized-cost experiments.

This contribution is not yet empirically complete.  The mathematical and local
verification infrastructure is substantially complete, but the clean
publication reruns and the Karolina crossover campaign have not been executed.

## 2. Evidence status at handoff

The following distinctions are mandatory throughout the paper and archive.

1. **Implemented and tested** means that code paths and focused tests pass in
   the current development tree.
2. **Diagnostic pilot evidence** means that a numerical calculation was run
   from the current dirty worktree.  It may reveal defects, calibrate a future
   protocol, or support explicitly provisional discussion.  It is not final
   publication evidence.
3. **Prepared, not submitted** means that the cluster matrix, executor, safety
   checks, and dry-run commands exist, but no scheduler submission or cluster
   result exists.
4. **Publication evidence** requires a clean immutable commit, prespecified
   protocol, complete run record, archived state and hashes, admitted terminal
   status, and all problem-specific accuracy gates.  No new revision result has
   reached this class yet.
5. **Historical evidence** remains under the 2026-07-08 submission campaign.
   It cannot be silently relabeled or mixed with the revision campaign.

Current hard facts:

- the Git worktree is dirty and contains pre-existing user changes as well as
  the revision;
- no Karolina job was submitted or run;
- no DOLFINx/ADIOS2 dependency was rebuilt;
- the manuscript makes no matched DOLFINx claim, so that comparison is removed
  from required scope. An optional future comparison remains blocked by an
  ADIOS2 shared-library ABI mismatch and requires explicit user approval before
  dependency repair;
- the generated working PDF compiles and passes syntax, log, float,
  hygiene, literature-index, and visual checks;
- the table-evidence checker correctly refuses the 0/14 diagnostic source set,
  and the submission-bundle manifest separately refuses promotion while its
  source paths are uncommitted.

## 3. Work completed in the current revision

### 3.1 Scientific scope and mathematical claims

1. The paper has been refocused from a broad solver inventory to derivative
   placement in distributed nonlinear finite elements.
2. The main text now defines the fixed constrained discrete scalar functional
   before comparing its derivatives.
3. The element-versus-constitutive identity is stated as a conditional
   proposition.  Its assumptions include the same scalar density, quadrature,
   material and history data, affine lift, fixed strain map, selected smooth
   branch, principal-value ordering, and shear convention.
4. Owned-row colored recovery is stated separately with structural-pattern,
   color-interference, ghost-state, and PETSc-ownership assumptions.
5. Branch interfaces and unresolved repeated spectra are explicitly excluded
   from ordinary-Hessian and Newton-convergence claims.
6. Plasticity3D is described as a synthetic endpoint potential, not an
   incremental path-consistent constitutive model or physical slope-stability
   validation.
7. Topology optimization is supplementary and is not called an optimal design,
   converged optimization, or KKT point.
8. Riesz-scaled residuals and corrections are separated from coefficient norms
   and from the Euclidean trust-region norm.

The exact claim vocabulary and family-by-family restrictions are in
[`mathematical_status_and_claim_dictionary.md`](mathematical_status_and_claim_dictionary.md).

### 3.2 Common numerical and provenance contract

The revision implements:

1. strict versioned terminal JSON records with atomic replacement;
2. explicit failed, capped, fixed-work, converged, and nonfinite terminal
   states;
3. JSON-safe nonfinite handling using `null`, never nonstandard `NaN` or
   infinity tokens;
4. canonical state exports with global ordering, mesh/topology metadata,
   constraints, metric provenance, and content hashes;
5. scalar P1 lumped-mass Riesz maps;
6. a Plasticity3D glued-free-space reference-elastic Riesz map with symmetry,
   MUMPS inertia, and independently recomputed inverse-solve residual checks;
7. a hyperelastic two-end-constrained reference-elastic Riesz map assembled at
   the verified identity deformation, with the same numerical inertia check and residual
   checks;
8. per-phase and per-rank timing fields suitable for collective-maximum timing;
9. campaign, case, route, repetition, state, direction, code, input, and output
   identities needed for later claim-to-artifact lineage.
10. a managed 14-source publication finalizer that executes only from one clean
    immutable experiment commit, preserves untouched raw payloads and receipts,
    emits hash-bound table-facing copies, supports a verified descendant release
    commit, and rejects dirty, pilot, altered, cross-experiment, or path-escaping
    inputs;
11. semantic table admission that recomputes every displayed family gate,
    requires the exact 14 inputs and four manuscript tables, and regenerates the
    tables independently for byte comparison.

Coefficient Euclidean stopping remains available only as a legacy/default
diagnostic where the new metric has not been explicitly selected.  Publication
campaigns must select and numerically check the declared problem-specific metric.

### 3.3 Experiments and implementation checks already executed

All rows in this subsection are dirty-worktree diagnostics unless explicitly
stated otherwise.

| ID | Completed work | Principal observation | Admissible conclusion | Still open |
| --- | --- | --- | --- | --- |
| `EXP-DERIV-001` | Smooth and branch-interior gradient, Hessian, action, finite-difference, and symmetry checks | Fixed-element P1/P2/P4 derivative routes pass their frozen tolerances | Local fixed-functional derivative consistency at tested states | Clean rerun and complete assembled/distributed matrix gates |
| `EXP-MC-001` | Five strict branch interiors, all five adjacent interfaces from both sides, rotations, and repeated-spectrum probes | 5/5 interiors, 15/15 interface pairs, 15/15 rotation checks, and 7/7 repeated-spectrum cases were finite; maximum gate HVP error was $1.54\times10^{-9}$ | The selected branch program is finite and internally consistent at the tested states | No differentiability, generalized derivative, physical, or nonlinear-convergence claim |
| `EXP-VAL-001` p-Laplace | Independent NumPy/SciPy P1 manufactured solve | Last successive rates approach 1.999 in $L^2$ and 0.999 in $H^1$ | Expected smooth P1 spatial consistency | Clean rerun; production L-shaped case is not validated |
| `EXP-VAL-001` Ginzburg--Landau | Controlled positive-branch manufactured solve | Last rates are 2.001 in $L^2$ and 1.000 in $H^1$ | Smooth selected-branch spatial consistency | No multi-basin robustness or minimum claim |
| `EXP-VAL-001` affine hyperelasticity | Independent analytic energy, Piola, tangent, traction, rigid-mode, and rotation checks | Relative energy/residual/Hessian errors are $4.58\times10^{-15}$, $2.67\times10^{-15}$, and $4.38\times10^{-16}$ | Exact affine-patch formulation/derivative check | Does not test nonaffine convergence or rotating-beam validation |
| `EXP-VAL-001` nonaffine hyperelasticity | Independent P1 manufactured solves on 4/8/16/24 subdivisions with order-4/6/8 load checks | Last rates are 1.887 for displacement $L^2$, 1.006 for deformation gradient, and 0.983 for first Piola; minimum $J_h=0.844$; maximum response/load-quadrature fraction is $8.29\times10^{-6}$ of FE error | Nontrivial spatial/formulation consistency with positive determinant and negligible tested load-quadrature contamination | Clean rerun; an optional matched backend is outside required scope |
| `EXP-DIST-001` | Canonical one-versus-two-rank hyperelastic fixed-state comparison | Topology/maps/CSR match exactly; relative energy $8.67\times10^{-19}$; residual and matrix differences zero; action/correction errors $2.24\times10^{-16}$ and $2.33\times10^{-16}$ | Controlled rank-count algebraic equivalence for this construction | Four ranks, factorized construction changes, nonlinear endpoints, clean placement/repetitions |
| `EXP-STOP-001` scalar | Deterministic zero-source GL baseline versus tenfold-tighter lumped-$L^2$ stopping | Relative final-state difference $2.22\times10^{-7}$ | One local GL tolerance-sensitivity observation | Cross-mesh calibration and clean rerun; the p-Laplace pair is excluded because its random initial vectors were neither seeded nor stored |
| `EXP-STOP-001` Plasticity3D | Loose versus tight one-step route diagnostic | Tight KSP reduces element/constitutive state difference to $5.28\times10^{-11}$ relative | Historical route discrepancy is consistent with inexact linear solves | Full Riesz-stopped nonlinear endpoints; no timing claim |
| `EXP-STOP-001` hyperelasticity | One-/two-rank reference-map setup and terminal evaluation | Both have inertia $(0,0,2133)$ and identical state scale; inverse residuals $1.20\times10^{-10}$ and $9.35\times10^{-11}$ | Reference metric construction and serialization work at this level | Nonlinear solve, mesh sweep, and preconditioner calibration |
| `EXP-GLOB-001` | Controlled Newton--Armijo versus reduced-subspace trust--Armijo | On one GL instance Newton fails after 40 rejected trials; trust reaches stored criteria in 12 iterations.  Hyperelastic endpoints fail the equality gate | Exact failure semantics and one qualitative robustness observation | Distinct instances, Riesz endpoints, clustering, repetitions; no timing probability claim |
| `EXP-DISC-001` | Independent named rules and fixed-state P1/P2/P4 energy/residual/action evaluation | P2 action differs by 2.08% and P4 by 4.35% under the enriched reference, despite small energy changes | Energy alone is an invalid quadrature-adequacy test | Solve the 24- and 125-point P4 problems separately with admitted stops |
| `EXP-TOPO-001` | Corrected material-measure/fraction semantics and one-/two-rank smoke | Unit metadata is now explicit; the capped diagnostic remains infeasible and non-KKT | Software/unit diagnostic only | Core optimization campaign intentionally excluded |
| `EXP-ROUTE-001` local records | Twelve local records with frozen features and action checks | 0/12 enter the strict route map because their legacy timing records do not prove the MPI collective maximum; 0 rows are eligible for model fitting | The tightened analyzer fails closed and preserves the rejection reasons | Clean paired cluster blocks and held-out evaluation |

The full protocols, commands, gates, and raw diagnostic paths are indexed by
[`protocols/README.md`](protocols/README.md).

### 3.4 Prepared Karolina work

The bundle under
[`experiments/runners/paper_revision_karolina`](../experiments/runners/paper_revision_karolina/README.md)
contains the row matrix, executor, Slurm script, dry-run-only preparation
wrapper, handoff, and safety manifest for `EXP-ROUTE-001`, `EXP-DISC-001`, and
`EXP-SCALE-001`.

The frozen matrix SHA-256 is
`0010453084157f5ccf0ba307ed26b377c4aff80ad9f095fa359270e90ad5b1a5`.
The route-analysis contract is version 2 with SHA-256
`275f08e03221aed13a1b3cdbd1649a99274cb26b2c6d949f62bb4c3e6012fb00`;
the route-cost analyzer SHA-256 is
`abf34fed7715173e3d8c8c4b896c728a05b97d5bf12887f870eb1d4b1b240d38`.
The reviewed static campaign manifest SHA-256 is
`26da0b6790307baa14830e1cbb30f45e301de82f1acca5ec3289a1b90aa5d5b3`.
It contains 115 required Slurm rows (299 in-allocation process executions,
99.95 node-hours) and 33 optional rows (75 process executions, 62.50
node-hours). The full 148-row, 162.45-node-hour inventory is explicitly
forbidden as a single submission tranche.

The final local preparations are archived as explicit non-execution records:

- `artifacts/reproduction/paper_revision_karolina/paper_revision_karolina_prepared_v8/`
  contains the 115 required commands;
- `artifacts/reproduction/paper_revision_karolina/paper_revision_karolina_route_optional_v8/`
  contains the 30 optional Tier-B commands;
- `artifacts/reproduction/paper_revision_karolina/paper_revision_karolina_p3d_scaling_optional_v8/`
  contains the three optional Plasticity3D scaling commands;
- `artifacts/reproduction/paper_revision_workstation/exp_route_001_prepared_v8/`
  contains the 12-block, 36-route-process workstation plan.

Every preparation manifest records `source_dirty: true` and status
`prepared_not_submitted` or `prepared_not_executed`; a future clean operator
must regenerate rather than promote these plans.

The route design has been revised so one allocation executes a randomized,
balanced all-route block.  It records every rank-local phase time and defines
the comparison time as the MPI collective maximum.  Independent blocks supply
replication.  Separate microbenchmarks measure kernel, contraction, coloring,
insertion, communication, cache, memory, and imbalance factors. Three
independent blocks per rank record raw-rank times and a fixed-total-work,
realized imbalance. The synthetic kernels are descriptive mechanism
diagnostics, not route-faithful selector features; the production selector uses
only measured route covariates and remains contingent on its frozen holdout
gates. Multiple
deterministic actions, saved gradient/residual, branch counts, direct feasible
CSR checks, and a strict endpoint analyzer precede any release of timing.
The single route-work feature is now aligned with the collective-max response:
it uses the busiest rank's overlap-element workload.  The constitutive count
includes the dense tangent and both contractions in $B^\top C B$.  These are
prespecified structural counts, not exact FLOP claims.

Colored sparse recovery at P4 points is retained as a visible prespecified
non-attempt motivated by pilot memory risk; no failure threshold is claimed and
the slot is never silently omitted or imputed. The
high/low-order confirmation uses the numerically checked reference-elastic stopping map
and a separate Tier-B endpoint analyzer.

No `sbatch`, admission-test, or scheduler query was invoked in this revision.
The allocation date recorded in the private reference has expired, so a future
authorized session must revalidate the allocation, account, QoS, partitions,
environment, clean commit, and matrix hash before even a Slurm test-only call.

### 3.5 Manuscript and generated evidence surface

The paper has been rewritten as a compact working manuscript titled
*Derivative Placement in Distributed Nonlinear Finite Elements: Equivalence,
Verification, and Evidence*.  The abstract, introduction, related work,
methodology, implementation, benchmark design, verification, results,
discussion, conclusion, and appendix now follow the repository style guide.

The main changes are:

1. definitions and propositions precede empirical claims;
2. the three route assumptions and owned-row recovery conditions are explicit;
3. manufactured, analytic, and fixed-state verification precede performance;
4. all historical speedup, route-ranking, scaling, convergence, and topology
   claims that fail the revised evidence contract have been removed;
5. current dirty pilots are labeled diagnostic and are not used for timing or
   universal ranking;
6. Plasticity3D branch formulas and elastic-first selection are complete in the
   appendix;
7. four compact revision tables are regenerated directly from strict
   artifacts; their manifest records `publication_evidence: false`;
8. recent closest-work records narrow the novelty claim to conditional
   three-route equivalence, distributed ownership, and an eventual controlled
   CPU/MPI selection result;
9. the clean publication workflow now has an executable 14-source plan,
   managed receipts, semantic admission, exact table binding, and independent
   regeneration rather than a manually asserted admission flag;
10. the route pipeline admits either a successful selector or the prespecified
    finite-map negative result, but both require all 96 active rows, six exact
    censors, 74 training rows, 22 holdout rows, and empty invalid input;
11. the negative branch strips coefficients, predictions, rankings, winners,
    recommendations, and crossover claims, while the synthetic factor study
    remains reportable but cannot gate or enter the selector;
12. the one/two-rank MPI pilot is now identified as an element-Hessian assembly
    check; distributed colored recovery remains an explicit open gate;
13. the stopping interface now states its absolute and relative quantities,
    Boolean terminal predicate, zero-initial-residual rule, and exact retained
    diagnostic settings.  The unseeded, unstored p-Laplace comparison has been
    removed rather than presented as a controlled result; and
14. mechanics geometry and parameters now use one explicit nondimensional
    convention, with elastic-modulus conversions and the distinct strength-
    reduction factors recorded.

The section-by-section removal/replacement record is
[`manuscript_evidence_rewrite_ledger.md`](manuscript_evidence_rewrite_ledger.md).

### 3.6 Verification already performed

The focused scientific suites exercise publication records, metric solvers,
all new verification runners, distribution, globalization, route analyzers,
factor microbenchmarks, topology semantics, table generation, campaign safety,
and manuscript hygiene.  The final integrated selection contains 250 tests and
passes in 107.73 seconds.  Within that selection, the outcome-independent
finalizer/admission/table/route subset passes 92 tests and the final campaign
preparation/hash subset passes 17 tests.  The final commands and expected
release failures are recorded in Section 7 rather than preserving superseded
transient failures from matrix development.

The PDF checks completed successfully for:

- LaTeX compilation with resolved references;
- absence of overfull/underfull boxes and LaTeX/package warnings;
- `qpdf --check`;
- figure/table auxiliary ordering;
- prohibition of unapproved hard `[H]` floats;
- manuscript hygiene;
- literature source-index consistency;
- page-by-page rendered visual inspection at final size.

`make submission-check` is expected to remain nonzero in the development tree:
the revision tables are diagnostic, the source admission audit admits 0/14
inputs, and the submission bundle contains uncommitted or stale source paths.
These are independent fail-closed release blockers.  Do not bypass them;
finalize clean sources, regenerate tables, and rebuild the bundle from the
release candidate.

## 4. Ordered remaining work

The order below is mandatory.  Later analysis and prose must not be used to
retroactively change an earlier experiment gate after results are visible.

### 4.1 Freeze a clean experiment commit and immutable protocol set

**Priority:** P0.  **Compute:** workstation only.  **Owner:** authors/release
operator.  **Prerequisite:** final review of the current implementation.

Tasks:

1. Separate unrelated user changes from the paper revision without destructive
   Git operations.
2. Resolve every independent red-team critical or major finding.
3. Rerun the complete focused and publication test suites.
4. Commit the exact runner, solver, protocol, analyzer, table generator, and
   manuscript sources used for the campaign.
5. Require `git status --porcelain` to be empty.
6. Initialize the managed source plan with
   `finalize_revision_publication_campaign.py init-plan`; review and commit that
   plan before executing any source command.
7. Record the experiment commit, environment lock, compiler/MPI/PETSc/JAX versions, input
   hashes, protocol hashes, matrix hash, and generated baseline patch in the
   revision manifest.
8. Create the publication evidence-source manifest through the versioned
   finalizer and admission tools
   that independently checks each input's own run kind, clean commit, command,
   environment, terminal gate result, analyzer hash, and artifact hashes. A
   reviewed signature/approval may aggregate those checks; a manually written
   `admitted=true` flag is not sufficient by itself.
9. Rerun every retained local verification through its managed plan command,
   without a dirty-pilot override.
10. Permit later manuscript/table commits only as clean descendants of the
    immutable experiment commit.  Revalidate ancestry and every bound source
    hash; do not require the experiment and release commits to be identical.

Gate: all local correctness rows pass unchanged prespecified tolerances and
write managed receipts or strict records whose experiment commit is the same
clean immutable commit.  The table and release commits are clean descendants
whose bound producer files remain byte-identical.

Failure action: fix the scientific or serialization defect, create a new
protocol version before rerunning, and never overwrite the failed record.

### 4.2 Obtain independent protocol sign-off before expensive execution

**Priority:** P0.  **Compute:** none.  **Owner:** independent numerical and HPC
reviewers.

Review and sign:

1. `EXP-ROUTE-001` train/holdout split, paired-block randomization, timing
   reduction, censoring, error model, and endpoint gates;
2. `EXP-STOP-001` Riesz operators, units, scales, inverse-solve tolerances, and
   cross-mesh calibration;
3. `EXP-DISC-001` factor separation and the 24/125-point common evaluator;
4. `EXP-SCALE-001` fixed policy and node/rank placement;
5. resource ceilings, array dimensions, Slurm placement, account/QoS safety,
   and output/archive estimates.

Gate: written sign-off contains no unresolved major item and all hashes match
the clean commit.

Failure action: revise and version the protocol before any cluster result is
generated.

### 4.3 Complete the clean local correctness and stopping matrix

**Priority:** P0.  **Compute:** S.  **Prerequisites:** 4.1--4.2.

Execute, in this order:

1. scalar manufactured p-Laplace and Ginzburg--Landau verification;
2. affine and nonaffine hyperelastic verification;
3. all Plasticity3D material-point and fixed-element branch checks;
4. one-/two-/four-rank canonical fixed-state distribution checks;
5. scalar, Plasticity3D, and hyperelastic Riesz-map mesh/tolerance sweeps;
6. full global matrix/action comparisons for every route that will be timed.

Required outputs include canonical states and directions, full error sweeps,
symmetry and numerical inertia checks, true inverse-solve residuals, branch margins, memory,
run records, commands, and hashes.

Gate: every route planned for timing passes gradient, residual, matrix, and at
least four deterministic action checks at the frozen tolerances.  Every final
endpoint passes its Riesz residual and correction gate.

Failure action: censor the failed route/case and report it.  Do not time a
failed derivative or substitute a looser norm.

### 4.4 Execute the required Karolina route and factorized-cost tranche

**Priority:** P0 for the central contribution.  **Compute:** L.  **Owner:** an
authorized Karolina operator.  **Prerequisites:** 4.1--4.3, an active
allocation, and explicit user authorization.

The current bundle is execution-prepared but not yet scientifically released.
The pre-scheduler implementation review is complete with these dispositions:

1. the headline case is consistently $P_4(L_1)$ at strength-reduction factor
   $\lambda_{\mathrm{sr}}=1.55$, with
   element and constitutive AD active and colored SFD retained as a
   prespecified non-attempt without a threshold claim;
2. the replicated synthetic factor study is explicitly descriptive and
   non-route-faithful; it is neither a predictor feature nor a selector gate;
3. production fixed-state route blocks, quadrature blocks, and synthetic
   mechanism diagnostics have distinct labels and admission paths;
4. every admitted timing retains raw rank values and reconstructs its declared
   MPI collective maximum;
5. independent route/factor blocks are balanced, and the selector propagates
   paired allocation-block uncertainty through coefficients, predictions, and
   route ordering;
6. the 12-block, 36-process workstation training plan is frozen and passes
   prepare-only validation, while execution still requires a clean commit;
7. end-to-end synthetic selector tests cover full-rank fitting, conditioning,
   holdout blindness, pass/failure thresholds, uncertainty-resolved ordering,
   and endpoint-gate failure;
8. memory is an observed capacity diagnostic only; no memory predictor or
   measured colored-SFD failure threshold is claimed;
9. analysis admission hashes commands, environments, direct matrices, route
   arrays, factor records, settled accounting, semantic submission ledgers,
   release authorizations, copied review artifacts, and the Tier-B endpoint
   analysis; all record paths are archive-relative and relocation-tested;
10. the fixed Tier-B matrix confirms ordering only at prespecified points.  A
    crossover-location claim requires a separate post-fit, hash-bound matrix
    and a new release decision.

The remaining blockers are external or result-dependent: create one clean
immutable experiment commit; execute the workstation blocks from that commit;
revalidate the Karolina allocation, account, QoS, partitions, and software
environment; obtain the explicit human tranche releases; submit and collect
the canonical route rows with settled accounting; run the hash-bound Tier-B
endpoint analysis; and expose the rank-32 holdout only after the fitted model is
frozen.  No scheduler command has been invoked during this preparation.

Safety sequence:

1. revalidate the allocation end date, account `fta-26-40`, QoS `3571_6328`,
   and `qcpu_exp`/`qcpu` partitions;
2. verify PETSc 3.24.x, petsc4py, `Mat.setPreallocationCOO`, Hypre, and MUMPS;
3. verify the clean commit and campaign-matrix hash;
4. regenerate the dry-run plan and inspect every exact `sbatch` command;
5. run one-node Slurm test-only admission with the required revalidation
   environment variables;
6. inspect `test_only_results.jsonl`;
7. submit only the required route/cost tranche, in staged groups, after a
   second explicit confirmation;
8. monitor failures without changing policy or silently resubmitting under a
   different configuration.

Gate: every required paired block is present; route position is balanced;
rank-local timings reconstruct the declared collective maximum; correctness and
endpoint admission pass before timing is visible; structural memory censors are
retained.

Failure action: report the finite empirical map and censor reasons.  Do not fit
or advertise a crossover selector when minimum train/holdout coverage fails.

### 4.5 Run the strict route and endpoint analyses

**Priority:** P0.  **Compute:** S.  **Prerequisite:** admitted outputs from
4.4.

Tasks:

1. run the finite-map/cost-model analyzer on the clean submitted campaign;
2. run the Tier-B endpoint analyzer on all high/low-order confirmation blocks;
3. fit only the prespecified training rows;
4. evaluate rank-32 and other frozen holdouts once;
5. report uncertainty, prediction error, censored cases, memory limits, and
   factor contributions;
6. perform sensitivity checks that were declared before unblinding;
7. retain raw paired block differences rather than only aggregate speedups.

Gate: all minimum coverage, endpoint, interval, holdout, and prediction gates
in the analysis contract pass.

Failure action: publish a bounded descriptive map or a negative result.  Remove
the predictive-selection claim and all resolved route rankings.

### 4.6 Complete the separated quadrature/discretization campaign

**Priority:** P1; P0 only if Plasticity3D remains a main benchmark.
**Compute:** M.  **Prerequisites:** admitted P3D stopping and derivative gates.

Run the six `EXP-DISC-001` rows in their frozen order: P4(L1) smoke, P4(L1)
24-point solve, P4(L1) 125-point solve, P4(L2) 24-point solve, P4(L2) 125-point
solve, and tight P4(L1) tolerance row.  Re-evaluate each successful endpoint
with both independently constructed rules.

Gate: own-rule residuals pass; endpoints are compared under the same 125-point
evaluator; tolerance error is below the estimated quadrature/mesh effect; and
branch changes are explicitly classified.

Failure action: call the result endpoint sensitivity or remove the study.  Do
not restore the historical energy-only degree trend.

### 4.7 Complete the controlled globalization study only if retained

**Priority:** P1.  **Compute:** M.  **Prerequisites:** common metric and
canonical endpoint checks.

Tasks:

1. freeze exact Newton--Armijo, reduced-trust--Armijo, and any separate
   production-bundle algorithms;
2. use distinct loads or initial states as robustness units;
3. record all accepted/rejected steps, function/gradient/HVP calls,
   negative-curvature events, Krylov work, and Riesz metrics;
4. cluster nonconvex endpoints before comparing methods;
5. use at least five machine-noise repetitions only within one admitted
   endpoint class.

Gate: the controlled tier differs only in globalization and meets the same
endpoint accuracy.  Production bundles are labeled separately.

Failure action: retain only the verified failure semantics and one-case
qualitative observation; make no success-probability or timing claim.

### 4.8 Run fixed-policy scaling only after correctness and equal accuracy

**Priority:** conditional P1.  **Compute:** M/L.  **Prerequisites:** a retained
scaling claim, verified case, frozen solver/preconditioner policy, and admitted
endpoint metrics. This step does not block the selected route/cost paper if the
scaling claim is removed.

Run the HyperElasticity 1/2/4/8-node series with five independent repetitions
per point.  Keep the mesh, rank density, PMG hierarchy, Hypre coarse policy,
tolerances, and output scope fixed.  Keep any Plasticity3D viability series
optional and separate.

Gate: all scale points solve the same problem to the same accuracy, timings are
collective maxima with uncertainty, and no tuned point is mixed into the fixed
series.

Failure action: report deployment behavior or numerical efficiency without
calling the result conventional strong scaling.

### 4.9 Keep the matched-backend comparison outside required scope

**Status:** removed from required manuscript scope; optional P2 companion.
**Compute:** none for the selected paper.

The manuscript retains independent manufactured and analytic verification and
makes no matched-backend claim. No dependency repair or user decision is
required to complete that scope. If the authors later broaden the paper, the
optional path is:

1. obtain explicit authorization to repair or rebuild the local ADIOS2/DOLFINx
   environment;
2. run an identical mesh/quadrature/load/constraint/state/stopping comparison;
3. require state, residual, HVP, reaction, energy, and observable errors to pass
   prespecified conditioning-aware tolerances before adding any claim.

The optional path currently remains blocked by the ADIOS2 ABI mismatch. Never
call a different functional or stopping rule a matched reference, and do not
restore `validated against` wording without a passing comparison.

### 4.10 Adjudicate evidence and regenerate the final manuscript

**Priority:** P0.  **Compute:** S.  **Prerequisite:** terminal decisions for all
retained experiments.

Tasks:

1. finalize and verify the managed 14-source campaign without overwriting raw
   staging evidence;
2. run the independent source admission audit and require 14/14 admitted;
3. update the atomic claim--evidence ledger with one terminal status per claim;
4. replace provisional table rows only from admitted clean artifacts;
5. add a route-order selector only if its holdout gate passes; add a crossover
   location only after the separately released post-fit confirmation passes;
6. add quadrature/globalization/scaling results only under their terminal
   labels;
7. state all censoring, failures, and negative results;
8. regenerate tables/figures from archived raw data and manifests;
9. update abstract and conclusion last;
10. rerun the primary-source novelty search and narrow claims if necessary.

Gate: every abstract and conclusion statement has one exact proof, clean
artifact, or explicit scope decision.

Failure action: delete the unsupported statement.  A green build cannot
override a failed scientific gate.

### 4.11 Build the clean release and reproduction archive

**Priority:** P0.  **Compute:** S.  **Prerequisite:** 4.10.

Tasks:

1. build the submission bundle from a clean descendant of the immutable
   experiment commit;
2. validate every table/figure input and manifest hash;
3. run `make -C paper submission-check` and all scientific tests;
4. verify license and citation metadata;
5. create the versioned release tag and require the tag and checked-out release
   commit to agree, while every evidence manifest names the reachable immutable
   experiment commit and verifies unchanged producer hashes;
6. deposit the archive, obtain a DOI, download it, and rerun Tier-1 checks;
7. confirm the paper availability statement, DOI, release, and PDF identify the
   same source state.

Gate: the archive recreates every central table and figure and contains no
restricted inputs.

Failure action: do not submit the manuscript until the mismatch is repaired.

### 4.12 Conduct final independent red-team and visual review

**Priority:** P0.  **Compute:** none/S.  **Prerequisite:** release candidate.

Obtain independent reviews of:

1. theorem assumptions and proofs;
2. numerical accuracy, conditioning, and failure semantics;
3. timing statistics and crossover inference;
4. claim-to-artifact provenance;
5. journal fit and novelty;
6. every rendered PDF page at final physical size.

Gate: no unresolved critical or major finding; all minor corrections are
resolved or explicitly dispositioned.

## 5. Publication acceptance matrix

| Question a skeptical reviewer will ask | Required evidence | Current state |
| --- | --- | --- |
| Are the routes derivatives of the same scalar? | Proposition assumptions plus fixed-state gradient/matrix/action checks | Theory and local pilots pass; clean global matrix campaign open |
| Is colored recovery exact under distribution? | Pattern/color proof, owned-row/ghost checks, canonical MPI comparison | Conditional proof and serial colored regression tests exist; the reported MPI pilot uses local element Hessians, so distributed colored recovery is open |
| Are endpoints equally accurate? | Numerically checked Riesz residual and correction, tighter-solve calibration | Metrics implemented; cross-mesh clean calibration open |
| Is the branch model mathematically honest? | Complete scalar program, strict-interior diagnostics, explicit exclusions | Scoped pass for a synthetic surrogate |
| Is the method physically validated? | Independent or matched physical evidence | Manufactured/analytic verification only; no production physical validation |
| Which route is fastest, and where? | Paired blocks, collective-max timing, factorized costs, held-out prediction | Prepared but not executed; no current timing conclusion |
| Is quadrature adequate? | Own-rule solved endpoints and common enriched evaluation | Fixed-state diagnostic rejects energy-only inference; solve campaign open |
| Is globalization comparison causal? | Controlled algorithms, common endpoint accuracy, repeated instances | One diagnostic observation; publication comparison open |
| Does scaling preserve the algorithm? | Fixed policy, equal accuracy, repetitions, uncertainty | Conditional P1 evidence only; prepared but not run and removable without changing the central claim |
| Is topology an optimization result? | Feasibility, KKT, baseline, fixed problem | No; supplementary diagnostic only |
| Can every result be reproduced? | Clean commit, strict manifests, archive, DOI | Infrastructure exists; clean release and DOI open |

## 6. Final go/no-go rule

The paper may be submitted as a scientific-computing paper only after the clean
derivative, stopping, route-crossover, and release gates pass, or after the
manuscript is narrowed so that every failed/open gate is no longer part of the
claimed contribution.  It should not be submitted to SIOPT as a new
optimization-method paper without a genuinely new optimization algorithm or
analysis and the corresponding KKT/convergence evidence.

At this handoff, the correct status is:

> **Scientifically strengthened working draft; implementation and local
> diagnostics largely complete; central distributed performance evidence,
> clean promotion, and archival release still open.**

## 7. Final local integration record

The final working-tree QA on 2026-07-10 produced the following results.

1. The integrated scientific, evidence, campaign, documentation, and paper
   suite passed **250 tests** in **107.73 seconds**.
2. The exact 14-source diagnostic audit completed successfully as an audit and
   admitted **0/14** inputs, as required.  Its global checks pass except for the
   intentionally dirty worktree; the route source retains 11 grouped blockers
   covering pilot/stale provenance, incomplete route rows, absent Tier-B
   evidence, and stale publication decoration.
3. The diagnostic evidence-manifest checker returned success in diagnostic
   mode and correctly reported that the tables are not submission-admissible.
4. The final PDF has **19 A4 pages**.  LaTeX reports no overfull or underfull
   boxes, package warnings, undefined references, or rerun requests.  `qpdf`,
   auxiliary ordering, float placement, manuscript hygiene, the literature
   index, and archive-neutral asset validation all pass.  The asset validator
   binds the four included tables and scans 18 provenance files.
5. All 19 PDF pages were rendered at 150 dpi and inspected at final page size.
   No clipping, overlap, broken equation, unreadable table, or inconsistent
   heading/footing was found.
6. `make -C paper submission-check` reaches the revision-evidence gate and
   exits nonzero only because the tables are diagnostic, the worktree is dirty,
   and no clean source-evidence manifest exists.  The bundle checker separately
   rejects uncommitted refresh paths.
7. The release-blocker audit also retains the repository license, durable
   archive/DOI, and publication-evidence blockers.  Its generic journal-template
   item is outside the requested no-template scope and is not treated here as a
   scientific defect.
8. `git diff --check` passes.  No scheduler command, Slurm admission test, or
   cluster experiment was executed.  The canonical v8 plans are dry-run records
   with `source_dirty: true`; they must be regenerated from the eventual clean
   experiment commit rather than promoted.
