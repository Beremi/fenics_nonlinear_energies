# Publication Action Plan

Last updated: 2026-07-10.

## Purpose And Authority

This document is the active, dependency-ordered plan for turning the current
manuscript into a scientifically rigorous and reproducible journal paper. It
incorporates the detailed SIOPT-style referee audit of the historical broad
manuscript. The plan prioritizes mathematical validity, direct derivative
verification, equal-accuracy experiments, optimization evidence, and
reproducibility before prose revision.

The historical checklist in [`todo.md`](todo.md) remains useful as an audit
ledger. Its earlier conclusion that only licensing, archival DOI, and editorial
metadata block submission is superseded by the scientific blockers in this
plan. Similarly, a successful `make -C paper publish-check` establishes asset
and provenance consistency; it does not establish that the mathematical or
numerical claims meet a journal's acceptance standard.

Journal-template accommodation is deliberately outside this plan, in
accordance with the requested scope. The plan covers scientific content,
experiments, exposition, reproducibility, and release evidence.

## Status And Priority Vocabulary

Use one status for every task:

- `BLOCKED`: a named prerequisite or decision is unresolved;
- `READY`: all prerequisites are satisfied;
- `IN PROGRESS`: work has started;
- `EVIDENCE COMPLETE`: the outputs exist and the acceptance criteria pass;
- `REMOVED FROM SCOPE`: the material was deliberately excluded from the paper;
- `DONE`: all scientific, manuscript, and archival work is complete.

Use the following priorities:

- `P0`: publication-blocking;
- `P1`: required for a strong paper but not a prerequisite for every other task;
- `P2`: valuable only after the central contribution is secure.

Compute estimates use these classes:

- `S`: workstation work or at most 2 node-hours;
- `M`: approximately 2--20 node-hours;
- `L`: approximately 20--100 node-hours;
- `XL`: more than 100 node-hours or a multi-day allocation.

These are planning classes, not allocation requests. Before an `L` or `XL`
campaign, prepare the experiment card, estimate the ranks and wall time, inspect
the applicable IT4I guidance under `external/HPC_cluster_config/`, and obtain
user approval.

All numerical tolerances below are recommended starting points unless a norm,
denominator, and scale are explicitly defined. Final gates must be dimensionless
or physically scaled, conditioning-aware, and prespecified and frozen before the
publication run. A near-zero quantity requires a declared safe scale rather
than division by the raw reference value.

## Global Completion Rule

The paper is scientifically ready for journal formatting and submission only
when all of the following are true:

1. The central contribution is appropriate for the selected journal and can be
   stated without relying on software integration alone.
2. Every mathematical identity used by a central solver is proved under stated
   assumptions or explicitly restricted to a synthetic benchmark.
3. Every central derivative route passes direct gradient, Hessian-action,
   assembly, and branch-consistency checks.
4. Compared solvers terminate under one defined, mesh-aware accuracy contract.
5. Every performance claim includes repeated measurements, uncertainty, full
   timing scope, and complete environment metadata.
6. Every optimization result reports feasibility and a justified stationarity
   or KKT measure.
7. Every figure and table is generated from archived raw evidence with a
   complete command-to-claim lineage.
8. The licensed, versioned release and its durable archive reproduce the final
   manuscript assets from a clean environment.
9. Independent mathematical, numerical, and journal-fit reviews contain no
   unresolved critical or major findings.

## Mandatory Execution Order

Do not rewrite the manuscript around anticipated results. Complete the theory,
verification, validation, and principal experiments first, adjudicate the
evidence, and only then restructure the text.

| Step | Work package | Type | Priority | Depends on | Completion gate |
| ---: | --- | --- | --- | --- | --- |
| 1 | Freeze the paper's scientific identity | Decision | P0 | none | one defensible central claim |
| 2 | Freeze the baseline and evidence protocol | Governance | P0 | 1 | versioned claim and experiment ledger |
| 3 | Establish the mathematical foundations | Theory | P0 | 1--2 | proofs or explicit scope reductions |
| 4 | Implement the common numerical contract | Infrastructure | P0 | 3 | tested norms, timings, and result schema |
| 5 | Run direct derivative and branch verification | Experiment | P0 | 3--4 | all derivative correctness gates pass |
| 6 | Run independent verification; add a matched backend only if claimed | Experiment | P0 | 4--5 | central implementations independently verified; optional comparator resolved or removed |
| 7 | Rerun the headline derivative-route result, if retained | Experiment | P0 when retained | 5--6 | replicated equal-accuracy evidence |
| 8 | Establish the route-crossover and cost model | Experiment | P0 | 5--6 and a retained route case | bounded, generalizable selection evidence |
| 9 | Redesign the globalization comparison | Theory/experiment | P1 | 4--5 | exact algorithms and common-accuracy results |
| 10 | Replace the confounded discretization study | Experiment | P1 | 4--6 | separated quadrature, mesh, and degree evidence |
| 11 | Resolve topology optimization | Decision/theory/experiment | P0 for SIOPT | 1, 3--6 | KKT-quality result or removal from core claims |
| 12 | Run controlled scaling and memory studies only if retained | Experiment | conditional P1 | verified retained cases only | fixed-policy, repeated scaling evidence or a recorded scope removal |
| 13 | Adjudicate all claims against the evidence | Scientific gate | P0 | every required or removed Step 3--12 item | atomic claim--evidence matrix |
| 14 | Refocus and shorten the manuscript | Manuscript | P0 | 13 | one coherent evidence-driven narrative |
| 15 | Rewrite each manuscript section | Manuscript | P0 | 14 | section-specific acceptance checks pass |
| 16 | Regenerate displays and supplement | Artifact | P1 | 14--15 | minimal, reproducible evidence surface |
| 17 | Prepare the clean release candidate and reproduction bundle | Archive | P0 | 13--16 | licensed candidate and clean reproduction |
| 18 | Run independent red-team reviews | Review | P0 | 14--17 | no unresolved major findings |
| 19 | Complete final QA, tag, archive, and DOI deposit | Final QA | P0 | all prior steps | release and PDF are mutually consistent |

A dependency is satisfied by `DONE`, `EVIDENCE COMPLETE`, or a documented
`REMOVED FROM SCOPE` decision approved in Step 1. Do not force an irrelevant
Plasticity3D, topology, degree, or scaling campaign to block a coherent paper
that deliberately excludes it.

### Route-To-Required-Step Matrix

| Route selected in Step 1 | Required central work | Conditional or removable work |
| --- | --- | --- |
| A: derivative-placement optimization practice | Steps 2--6, 8, 13--19; Step 7 for the retained headline case | Steps 9--12 according to the research questions; topology may be removed |
| B: topology algorithm | topology-relevant parts of Steps 2--6, Step 11, and Steps 13--19 | Plasticity Steps 7--8 and unrelated scaling may be removed |
| C: nonsmooth mechanics optimization | retained-case parts of Steps 2--10 and Steps 13--19 | topology and noncentral benchmark families may be removed |
| D: SISC derivative-placement fallback | retained-case parts of Steps 2--8 and Steps 13--19 | Steps 9--12, topology, and any unproved plasticity interpretation may be removed |

---

## Step 1. Freeze The Scientific Identity Of The Paper

**Type:** `DECISION GATE`  
**Priority:** `P0`  
**Initial status:** `READY`  
**Compute:** none

### Objective

Choose the contribution before changing the experiment matrix. The current
manuscript demonstrates substantial integration, but a SIOPT submission needs
an identifiable optimization advance or a broadly generalizable result about
optimization practice.

### Tasks

1. Create `paper/venue_and_contribution_decision.md` containing:

   - the target journal;
   - one sentence stating the central contribution;
   - no more than three research questions;
   - the mathematical result or experiment answering each question;
   - the two or three benchmark families needed in the main text;
   - claims explicitly excluded from the paper;
   - material assigned to supplementary information.

2. Build a one-page novelty matrix with one atomic row per proposed
   contribution:

   | Proposed claim | Closest method | What is new | Required proof | Required experiment | Limitation |
   | --- | --- | --- | --- | --- | --- |

   Refresh the closest-method and state-of-the-art search from primary sources
   before completing this matrix. Search the optimization, automatic
   differentiation, consistent-tangent, distributed sparse-recovery, and
   topology literature relevant to the selected route. Repeat the search during
   the final rewrite only as a freshness check; novelty cannot be decided after
   the experiments have already been designed.

3. Choose one primary route:

   - **Route A, optimization practice:** derive and validate a predictive rule
     for selecting element AD, constitutive AD, or colored sparse recovery in
     distributed Newton methods. This route requires a cost model and crossover
     evidence beyond one favorable case.
   - **Route B, optimization algorithm:** turn the frozen reciprocal topology
     model into a precisely defined sequential approximation method with
     first-order consistency, KKT termination, and convergence analysis.
   - **Route C, nonsmooth mechanics optimization:** derive the Mohr--Coulomb
     endpoint potential from a recognized variational principle and analyze a
     solver appropriate to its established regularity.
   - **Route D, SISC fallback:** focus on correctness, cost, memory, and parallel
     behavior of derivative placement in nonlinear finite-element Newton
     methods. This is the lower-risk route if no SIOPT-level advance survives.

4. State the contribution in at most three sentences without using repository
   structure, implementation breadth, or "toolset integration" as the claimed
   novelty.

5. Do not revise the title or abstract until this gate passes.

### Deliverables

- Venue and contribution decision record.
- Novelty matrix.
- Final research questions and explicit exclusions.
- Main-text versus supplement benchmark allocation.

### Acceptance Criteria

- At least one central result is a new optimization method or analysis, or a
  generalizable optimization-practice result supported across controlled
  regimes.
- Every abstract-level claim maps to a planned proof or prespecified
  experiment.
- The distinction from the closest literature is affirmative, precise, and
  source-verifiable.
- No experiment remains in the main paper solely because it has already been
  performed.

### Failure Action

If no defensible optimization contribution is selected, adopt Route D and
target SISC or a scientific-software venue. Do not compensate by overstating
standard software integration.

---

## Step 2. Freeze The Baseline And Create The Evidence Protocol

**Type:** `SCIENTIFIC GOVERNANCE`  
**Priority:** `P0`  
**Depends on:** Step 1  
**Compute:** `S`

### Objective

Separate historical evidence, pilot diagnostics, and final publication data.
Prevent changes made after viewing results from silently altering the
scientific question.

### Tasks

1. Preserve the current manuscript and evidence state:

   - record the exact Git commit;
   - record whether the tree is dirty and save a patch for relevant changes;
   - hash `paper/build/main.pdf` and the current submission bundle;
   - preserve `artifacts/reproduction/paper_submission_2026_07_08/` as a
     historical snapshot.

2. Create a new campaign root, for example
   `artifacts/reproduction/paper_revision_<date>/`. Never overwrite an earlier
   run or historical campaign.

3. Parameterize or update the existing paper pipeline so the new campaign root
   is an explicit input. Audit `paper/scripts/build_submission_bundle.py`, the
   figure and table generators, asset validators, release blocker, reproducibility
   note, and related tests for hard-coded references to
   `paper_submission_2026_07_08`. Preserve that bundle as historical evidence;
   do not silently repoint or overwrite it.

   Apply the same rule to experiment runners. In particular, parameterize the
   hard-coded raw/report roots in
   `run_paper_reviewer_gap_experiments.py` and require explicit campaign,
   run-UUID, route, and repetition identifiers. Audit runners that read existing
   documentation CSV files instead of executing cases; such imported rows are
   historical evidence and cannot become final revision measurements.

4. Create a provisional experiment card per final campaign using
   `docs/codex/templates/experiment_card.md`. At this stage record the research
   question, intended metric, comparison set, resource ceiling, and claim. Do
   not freeze mathematical tolerances or final solver parameters until Steps 3
   and 4 define the valid model, norms, algorithms, and KKT metrics.

5. Use `docs/codex/templates/run_manifest.md` for every run. Record:

   - commit and dirty-tree patch hash;
   - exact command and working directory;
   - code, configuration, mesh, and input hashes;
   - Python and package versions;
   - JAX/XLA, PETSc, MPI, compiler, BLAS, and preconditioner versions;
   - CPU/node/memory model, scheduler job, rank layout, affinity, and threads;
   - JAX 64-bit state and synchronization policy;
   - seed or deterministic policy;
   - raw outputs, states, logs, tables, and figures.

6. Create an atomic claim ledger using
   `docs/codex/templates/claim_evidence_table.md`. Include all claims currently
   appearing in the abstract, contributions, results synthesis, discussion, and
   conclusion.

7. Classify existing evidence correctly:

   - `run_paper_reviewer_gap_experiments.py` is reusable pilot infrastructure;
     its mostly single-run and fixed-work outputs are not final publication
     evidence.
   - Historical 222/288/372-second timings remain historical until Step 7.
   - Existing 5--10% comparator gates are regression thresholds, not validation
     tolerances.
   - Existing green paper checks demonstrate asset consistency only.

8. Audit comparator source availability and licenses before scheduling any
   comparator experiment. A central comparator must be inspectable and legally
   archivable; otherwise replace it or classify it as noncentral before compute
   is spent.

9. Define a negative-result record. Retain failures, timeouts, out-of-memory
   cases, and statistically inconclusive comparisons rather than filtering them
   from the final campaign.

10. Assign both an owner and an evidence reviewer to every `P0` step. The owner
    produces the deliverable; the reviewer signs the acceptance gate before a
    dependent step starts.

11. Run every publication-grade experiment in Steps 5--12 from a clean, frozen
    experiment commit. Record an empty `git status --porcelain` and the commit
    identifier before execution. A deliberately patched experimental build must
    be versioned as a new commit rather than represented only by a dirty-tree
    patch.

### Deliverables

- Frozen-baseline manifest.
- New campaign directory and protocol index.
- Parameterized paper-pipeline configuration for the new campaign.
- Provisional experiment cards and run-manifest schema. Final protocol
  prespecification and version freeze occur at the Step 4 protocol gate.
- Initial claim--evidence table.
- Negative-result policy.
- Comparator license/source audit and P0 responsibility matrix.

### Acceptance Criteria

- Every planned central claim has a named proof, experiment, or primary source.
- Every publication run can be distinguished from pilots and historical data.
- The governance and provisional cards are complete enough for Steps 3--4 to
  define the mathematical and numerical contracts without predetermining their
  tolerances.
- A clean table row can be traced to its raw run, command, code version, and
  input hashes.
- Every publication run is planned to start from a clean recorded commit.

---

## Step 3. Establish The Mathematical Foundations

**Type:** `SCIENTIFIC BLOCKER`  
**Priority:** `P0`  
**Depends on:** Steps 1--2  
**Compute:** `S--M`

### 3.1 Classify Every Benchmark Mathematically

Create a status table for all six problem families with these fields. Complete
the full proof program only for cases retained by Step 1. Excluded families
need enough classification to justify their removal or limited supplementary
status, but they do not block unrelated publication experiments.

- continuous problem;
- exact discrete functional or residual;
- variables, affine lifts, and constraints;
- quadrature rule;
- differentiability class;
- convexity or nonconvexity;
- derivative used by the solver;
- endpoint meaning;
- stopping contract;
- strongest permissible conclusion.

Use only the following status vocabulary unless a stronger result is proved:

- unique minimizer;
- approximate first-order stationary point;
- approximate local minimizer with a justified second-order condition;
- certified global minimizer under convexity or another global certificate;
- best observed point, with no optimality implication;
- approximate equilibrium;
- approximate generalized stationary point;
- regularized stationary point;
- approximate branchwise stationary endpoint;
- endpoint surrogate;
- fixed-work diagnostic;
- constrained design point satisfying stated KKT tolerances.

Add tests or generation checks that prevent `converged`, `minimizer`, or
`optimization solution` from appearing when the required metrics are absent.

### 3.2 Prove Or Reject The 2D Mohr--Coulomb Potential

For the implemented expression

\[
  \Phi(\varepsilon)
  = \sigma^\star(\varepsilon)^\top\varepsilon^e
  - \frac{1}{2}\sigma^\star(\varepsilon)^\top
    S\sigma^\star(\varepsilon),
\]

direct differentiation contains

\[
  \mathrm{d}\sigma^\star{}^\top
  \bigl(\varepsilon^e-S\sigma^\star\bigr).
\]

Complete the following tasks:

1. Define the admissible stress set, principal-stress ordering, sign convention,
   line and apex regions, and Davis-B parameter assumptions.
2. Determine whether the implemented stress is the unique solution of a
   complementary-energy projection problem.
3. If so, prove existence, uniqueness, equivalence of branch formulas with the
   projection KKT conditions, and the envelope identity
   \(\nabla_\varepsilon\Phi=\sigma^\star\).
4. Classify continuity, Lipschitz continuity, differentiability, and tangent
   definiteness on and across every interface.
5. State parameter assumptions that exclude singular denominators. Treat
   zero-friction or other singular cases separately or exclude them explicitly.
6. Verify that any principal-radius regularization and the closed-form return
   correspond to the same regularized variational problem.

### 3.3 Establish The 3D Branch Structure

1. Define the three-dimensional admissible stress set and complementary metric.
2. Determine whether the elastic, face, edge, and apex energies are value
   functions of one variational problem.
3. Prove that branch predicates cover the intended domain, implement consistent
   active sets, and agree in value and gradient wherever continuity is claimed.
4. State sufficient assumptions for all denominators, including the apex term
   proportional to \(K\sin^2\phi\).
5. Classify the global potential as one of: \(C^1\) piecewise \(C^2\), locally
   Lipschitz, semismooth, or discontinuous.
6. Treat the \(10^{-15}\operatorname{diag}(0,1,2)\) eigenvalue perturbation as
   an implementation detail, not part of the mathematical model. Quantify its
   scale and rotational effect if retained.
7. Handle repeated eigenvalues through spectral-function theory or an
   appropriate generalized derivative instead of assuming stable eigenvector
   ordering.

### 3.4 Select A Solver Compatible With The Established Regularity

- If the potential is \(C^1\), piecewise \(C^2\), and strongly semismooth,
  formulate a semismooth stationarity equation and show that the active-branch
  tangent belongs to an appropriate generalized Jacobian.
- If it is only locally Lipschitz, define stationarity through a generalized
  subdifferential and do not write an ordinary gradient at switching states.
- If a smooth regularization is adopted, define it explicitly, prove smoothness,
  and plan a regularization-sensitivity experiment.
- State which convergence assumptions apply to each benchmark and which runs
  remain heuristic.

### 3.5 Formalize Derivative Equivalence And Distributed Recovery

State and prove a fixed-branch proposition giving

\[
  \nabla_{u_e}\Phi_e
  = \sum_q w_q B_q^\top\sigma_q,
  \qquad
  \nabla^2_{u_e}\Phi_e
  = \sum_q w_q B_q^\top C_q B_q,
\]

under identical quadrature, potential, branch, strain convention, affine lift,
and constrained free-variable map.

For colored recovery, state the assumptions for exact recovery from AD-HVPs:

- the structural pattern contains every active nonzero;
- colors do not interfere in an owned row;
- ghost coverage contains all owned-row dependencies;
- row ownership is unique;
- branch switching does not silently invalidate the derivative being recovered.

Distinguish exact AD-HVP recovery from finite-difference recovery, whose error
depends on the perturbation parameter.

### 3.6 Formalize The Topology Problem And Frozen Model

If topology may remain central, define one fixed constrained problem with:

- one physical material target;
- one final \(p_{\mathrm{SIMP}}\);
- fixed regularization, loads, pads, mesh, and filter/phase-field parameters;
- bounds and design-variable representation;
- exact reduced compliance gradient and volume derivative;
- feasibility, projected-gradient, complementarity, and mechanics residuals.

Prove the value and first-order consistency of the frozen reciprocal model at
the current design. Determine whether it is a global majorizer, a local upper
model, a tangent model, or only a heuristic. If it is claimed as a new method,
replace the empirical controller with a fully defined sequential, trust-region,
or augmented-Lagrangian algorithm and supply convergence analysis under stated
assumptions. Otherwise use a standard optimizer and treat the reciprocal model
as a software experiment.

### Deliverables

- Benchmark mathematical-status table and claim dictionary.
- 2D and 3D potential derivation notes.
- Branch-interface and denominator assumption table.
- Proposition or theorem classifying the potential and its stress map.
- Solver-regularity decision.
- Derivative-equivalence and distributed-recovery propositions.
- Fixed topology problem, KKT system, and frozen-model result if topology is
  retained.
- Independent proof/regularity review record for every retained central model.

### Acceptance Criteria

- Every stress-gradient identity follows from a displayed derivation.
- Every reachable branch is defined, and every continuity claim is proved or
  removed.
- The nonlinear algorithm's assumptions match the established regularity.
- Every use of `minimizer`, `stationary`, `equivalent`, `stress`, `tangent`, and
  `optimization solution` matches the proved object.
- If a potential identity or topology method cannot be justified, the affected
  material is corrected, explicitly made synthetic, demoted to the supplement,
  or removed before expensive experiments.
- An independent mathematical reviewer has checked the central derivations,
  regularity classification, and parameter assumptions before Step 4 is
  approved.

### Hard Failure Actions

- If \(\nabla_\varepsilon\Phi\ne\sigma^\star\), correct the formulation or
  remove that potential from the scientific evidence.
- If the 3D potential lacks the regularity required by the solver, change the
  solver or reduce the claim to a heuristic algorithmic benchmark.
- If no analyzed topology method is provided, topology cannot serve as the
  SIOPT contribution.

---

## Step 4. Implement The Common Numerical And Measurement Contract

**Type:** `INFRASTRUCTURE`  
**Priority:** `P0`  
**Depends on:** Step 3  
**Compute:** `S`

### 4.1 Accuracy Contract

Define a discrete Riesz map \(W_h\) appropriate to each problem:

- a mass or reference \(H^1\) map for scalar fields;
- a reference elastic-energy map for mechanics;
- a mass-weighted design-space map for topology.

Require \(W_h\) to be symmetric positive definite on the constrained free
space. Eliminate or constrain rigid modes before using an elastic-energy map.
State whether \(W_h^{-1}g\) is evaluated exactly or by an iterative solve; if it
is approximate, set and verify a norm-solve tolerance small enough that the
stopping decision cannot change within the reported margin.

Implement and document the dual residual

\[
  \lVert g_k\rVert_{V_h^\ast}
  := \sqrt{g_k^\top W_h^{-1}g_k},
\]

a dimensionless relative residual, and the relative correction

\[
  s_k^{\mathrm{rel}}
  :=
  \frac{\lVert\delta u_k\rVert_{V_h}}
       {\max\{\lVert u_k\rVert_{V_h},u_{\mathrm{scale}}\}}.
\]

The exact normalization and physical scales must be stated. The unscaled
coefficient norm may remain as a diagnostic but cannot be the cross-mesh or
cross-degree convergence measure.

Require:

1. a residual gate for every result labeled completed;
2. correction size only as a secondary stagnation safeguard;
3. explicit true and preconditioned KSP residuals;
4. an inexact-Newton forcing condition or a documented fixed forcing rule;
5. identical final-accuracy targets for methods compared by time;
6. a tenfold-tighter sensitivity run for each headline case.

### 4.2 Canonical State And Derivative Comparisons

Implement:

- global canonical state ordering independent of MPI partition;
- mass- or energy-weighted full-state differences;
- route-to-route gradient and HVP differences;
- full matrix differences on small cases;
- Hessian symmetry defect;
- active-branch map differences;
- minimum branch margin and principal-value gap;
- topology feasibility, projected KKT, complementarity, and mechanics metrics.

Save initial, representative intermediate, and final states for central cases.

### 4.3 Timing Contract

For every performance run:

1. use MPI barriers around the declared timing region;
2. force JAX completion before stopping the timer;
3. state whether the reported value is rank maximum, mean, or launcher wall
   time;
4. separate process startup, JIT compilation, coloring, derivative evaluation,
   constitutive contraction, assembly, communication, preconditioner setup,
   Krylov solve, globalization evaluations, state output, and total time;
5. report cold-process and warm in-process measurements separately;
6. randomize route order within balanced replicate blocks;
7. retain all failed, timed-out, and censored observations;
8. predefine any outlier policy; default to deleting none.

Add periodic atomic progress/checkpoint records or signal-aware finalization in
the solver process. The current outer runners may terminate a process group at
the wall cap before the child writes counters or a final state; without this
instrumentation, the required timeout evidence cannot be collected.

### 4.4 Result Schema

Every final run record must contain at least:

- campaign, experiment, case, method, route, and repetition identifiers;
- problem, mesh, degree, quadrature, degrees of freedom, ranks, and threads;
- exact solver and preconditioner parameters;
- termination status and reason;
- absolute, relative, scaled residual, correction, and energy change;
- nonlinear, Krylov, function, gradient, Hessian/HVP, and preconditioner counts;
- timing decomposition;
- peak per-rank and per-node memory plus tracked allocations;
- state, branch, feasibility, and KKT diagnostics where applicable;
- environment and provenance fields from Step 2.

### 4.5 Implementation Surface

Extend established runners and result conventions rather than creating
disconnected scripts. Relevant starting points include:

- `experiments/runners/run_paper_reviewer_gap_experiments.py`;
- `experiments/runners/run_derivative_route_compare.py`;
- `experiments/runners/run_plasticity3d_derivative_ablation.py`;
- `experiments/runners/run_globalization_method_compare.py`;
- `experiments/runners/run_plasticity3d_validation.py`;
- `experiments/runners/run_topology_docs_suite.py`;
- `experiments/runners/barbora_he_first_step_scaling/`.

Add focused schema and numerical-contract tests under `tests/`. Do not add a
runtime dependency unless it is necessary and documented.

### 4.6 Exact Solver Contract

Before any headline rerun, freeze exact algorithms for every retained solver.
For line search, trust region, hybrid, continuation, or residual-bisection
methods, specify:

- merit function and model;
- trial-step and inner Krylov rule;
- forcing parameter;
- acceptance and rejection inequalities;
- trust-radius or line-search updates;
- negative-curvature and failed-solve handling;
- retry limits;
- residual, correction, stagnation, and cap exits;
- meaning of all reported counters.

Add boundary-condition tests for acceptance thresholds, negative curvature,
failed linear solves, and roundoff termination. Step 9 will compare these
already frozen algorithms; it must not define a solver only after Step 7 has
used it.

### 4.7 Protocol Review Gate

Before finalizing the experiment cards from Step 2, obtain an independent
review of the norms, tolerances, exact algorithms, timing regions, statistical
contrasts, output schema, and failure rules. Resolve every major protocol
finding before publication-grade runs begin.

After that review, finalize and freeze each experiment card with the exact
research question, hypotheses, primary and secondary metrics, problem contract,
controls, repetitions, route order, dimensionless acceptance/failure criteria,
resource ceiling, statistical analysis, and intended manuscript claim. Any
later change affecting scientific interpretation creates a new protocol
version.

### Acceptance Criteria

- At one identical stored state, one-rank and two-rank smoke cases produce
  equivalent canonical algebraic objects within declared scaled tolerances.
  Complete solved endpoints use the separate calibrated consistency contract
  from EXP-DIST-001.
- All required fields appear in successful, failed, capped, and timed-out JSON
  records.
- Phase timings have a documented relation to the reported total and cannot
  silently overlap.
- A smoke run produces a complete manifest and regenerates its table row
  without manual editing.
- Exact retained solver decisions are reproducible from the algorithm,
  parameter table, and stored history.
- The independent protocol review has no unresolved major finding.
- No full publication campaign begins before this step passes.

---

## Step 5. Run Direct Derivative, Assembly, Branch, And Stopping Verification

**Type:** `EXPERIMENT`  
**Priority:** `P0` for all retained central derivative routes  
**Depends on:** Steps 3--4  
**Compute:** `S--M`

Create focused experiment cards for the following gates. Use FP64 throughout
and deterministic directions or seeds.

### EXP-DERIV-001: Smooth Derivative Correctness

**Cases:** fixed-element \(p\)-Laplace regular,
Ginzburg--Landau regular/indefinite, and hyperelasticity near-identity and
intermediate states; Plasticity3D branch-interior fixed-element checks and one
separate serial assembled state at each retained degree. Distributed
Plasticity3D route checks are a distinct EXP-DIST-001 block. No full-mesh or
MPI conclusion is inferred from the smooth fixed-element cases.

**Controls:** identical discrete functional, quadrature, constrained space,
state, and direction. Use an independently evaluated centered finite difference
as an external numerical check and compare sparse matrix products with HVPs.
For the constitutive route, independently assemble
\(r_e=\sum_q w_qB_q^\top\sigma_q\) and compare it with the element-AD
residual. The production constitutive-tangent route currently shares an
element-AD residual, so comparing that shared residual alone would be
tautological.

**Metrics:**

- first-order energy Taylor remainder;
- gradient-difference and Hessian-action Taylor remainders;
- pairwise gradient and HVP errors;
- assembled-matrix/HVP error;
- Hessian symmetry defect;
- constrained/free-DOF consistency;
- serial versus distributed difference only in the separately declared
  distributed blocks.

**Recommended prespecified gates:**

- second-order first-derivative Taylor remainder over at least three
  non-roundoff step sizes;
- expected asymptotic order for the Hessian-action remainder;
- FP64 AD-route differences at most \(10^{-8}\), unless a conditioning analysis
  justifies a looser threshold in advance;
- symmetry defect at most \(10^{-10}\) for analytically symmetric cases.

### EXP-MC-001: Mohr--Coulomb Branch And Eigenvalue Verification

**State matrix:** interiors of every branch; both sides of each interface over
a prespecified range of dimensionless, normalized branch margins; apex and
near-apex states; repeated and nearly repeated principal stresses; random
rotations; and states exported from actual initial, intermediate, and final
solves. Define the stress, strain, and branch scales before generating these
states; absolute interface distances are not comparable across parameter
regimes.

**Controls:** use the complementary-energy projection or KKT problem, or
symbolic differentiation of the stated potential, as the primary independently
derived material reference. A separate NumPy transcription of the same branch
formulas is useful only as a cross-implementation regression check. Also use
finite differences of the scalar potential and all available FE derivative
routes at identical stored states.

**Metrics:**

- error between \(\nabla_\varepsilon\Phi\) and reconstructed stress;
- value and stress jumps across interfaces;
- tangent/HVP consistency and symmetry;
- rotation invariance of energy and covariance of stress;
- branch coverage and route-to-route branch identity;
- denominator margins, branch margins, and principal-value gaps;
- effect of the eigenvalue tie-break.

**Recommended prespecified gates:**

- smooth-state scaled stress and tangent errors initially targeted near
  \(10^{-8}\), with the final threshold derived from conditioning, state scale,
  and MPI reduction behavior before the run;
- rotation invariance and stress covariance near FP64 tolerance away from
  degenerate spectra;
- every claimed-continuous interface has a numerically vanishing jump;
- nonsmooth interfaces are classified and handled by a compatible algorithm;
- no reported trajectory approaches an excluded singular denominator without
  a documented safeguard.

Require exact route-to-route branch labels only outside a prespecified
normalized switch margin. Inside that margin, compare one-sided value/stress
limits, directional derivatives, and inclusion in the stated generalized
Jacobian. If the eigenvalue perturbation is retained, bound its expected
rotation error relative to the state scale and verify that bound over a scale
sweep rather than demanding unqualified machine-precision invariance.

### EXP-STOP-001: Stopping And Inexact-Solve Calibration

**Cases:** small and medium smooth cases; Plasticity3D \(P_1(L_1)\),
\(P_2(L_1)\), and \(P_4(L_1)\); one moderate topology mechanics problem.

**Matrix:** several nonlinear scaled-residual targets, for example
\(10^{-2}\) through \(10^{-6}\), crossed with KSP relative targets from
\(10^{-1}\) through \(10^{-4}\), where feasible. Hold all other model and
solver choices fixed.

**Metrics:** scaled and unscaled residuals, correction, energy, full state,
branch fractions, observables, iterations, and difference from the tightest
successful reference.

**Two-stage selection rule:** first choose a provisional algebraic tolerance by
comparison with a substantially tighter solve on the same discretization. The
provisional tolerance must change every primary observable by less than its
intended reported precision and the scaled state by less than a prespecified
same-discretization bound. After Step 10 estimates discretization error, confirm
that algebraic error is smaller. If not, tighten the contract and rerun every
affected endpoint and timing experiment.

### EXP-DIST-001: Distributed Equivalence

The selected paper requires two deliberately narrow controls. First, at an
identical canonical hyperelastic state, hold the procedural rank-local
construction, point-to-point overlap, owned-row COO assembly, and element
Hessian route fixed while changing only the rank count over one, two, and four
ranks. Second, at prescribed Plasticity3D states, hold the mesh and ownership
contract fixed while comparing the element, colored-recovery, and constitutive
routes over the same rank counts. Exact topology/input identity is part of both
gates.

HDF5 versus procedural mesh source, replicated versus rank-local construction,
all-gather versus point-to-point distribution, and global versus owned-row COO
remain useful deployment ablations, but the selected manuscript makes no
causal or performance claim about those factors. They are therefore not
required publication experiments. Any future statement attributing an effect
to one of them reactivates a one-factor-at-a-time experiment before release.

Use two gates:

1. **Fixed-state algebraic equivalence:** require tight scaled agreement for
   coordinates/connectivity, residuals, matrices, HVPs, and branch maps at one
   identical stored state.
2. **Solved-endpoint consistency:** use the separately prepared route Tier-B
   and stopping campaigns, with calibrated residual, observable, and
   weighted-state tolerances because partition-dependent preconditioners and
   reduction orders can alter nonlinear trajectories, especially for
   nonconvex problems.

### Implementation And Smoke Commands

Reuse the existing pilot matrix where useful:

```bash
./.venv/bin/python -m pytest tests/test_paper_reviewer_gap_experiments.py
./.venv/bin/python experiments/runners/run_paper_reviewer_gap_experiments.py \
  --mode smoke --sections p3d_derivative_degree --no-resume
```

The smoke campaign does not replace the derivative-specific experiment cards
or the new direct checks.

### Deliverables

- Raw Taylor curves and state-level derivative comparisons.
- Branch/interface identity table and rotation tests.
- Stopping-tolerance sensitivity report.
- Distributed-equivalence report.
- Focused regression tests for every discovered defect.

### Acceptance Criteria

- All smooth-route correctness gates pass before any route is timed for a paper
  claim.
- Every headline run uses a residual gate calibrated here.
- Branch-switch behavior is reported rather than hidden inside endpoint
  agreement.
- A derivative disagreement blocks Steps 7--12 until corrected and rerun.

---

## Step 6. Run EXP-VAL-001: Independent Verification And Optional Matched Comparison

**Type:** `EXPERIMENT`  
**Priority:** `P0`  
**Depends on:** Steps 4--5  
**Compute:** `M--L`

### Verification Ladder

1. **Patch and element tests.** Verify constitutive response, element residual,
   tangent action, constraint elimination, and reaction forces.
2. **Manufactured solutions.** Add smooth \(p\)-Laplace and
   Ginzburg--Landau examples and a hyperelastic patch or manufactured
   deformation test. Measure expected spatial convergence rates. For nonlinear
   manufactured problems, verify that the imposed source makes the manufactured
   state a solution, define the intended basin or initialization, preserve
   admissibility such as \(J>0\) in hyperelasticity, and confirm that the solver
   reaches the intended stationary branch rather than another solution.
3. **Optional identical-functional backend checks.** Add a JAX+PETSc, pure-JAX,
   and FEniCS comparison only if the final paper makes a cross-backend claim.
   Such a comparison must use an identical mesh, quadrature, energy,
   constraints, initial state, and stopping rule.
4. **Optional matched hyperelastic external comparison.** If retained, use the
   same constitutive law, quadrature, boundary data, and a nontrivial free
   response. Do not use the boundary-controlled maximum displacement as the
   primary validation metric.
5. **Conditional independent plasticity reference.** This block is required
   only if the paper introduces a mechanics-validation claim for Plasticity3D.
   In that case, release or recreate an independently derived and inspectable
   quadrature/material implementation and assembled-formula comparator using
   the identical constrained space.

   The selected paper instead declares Plasticity3D to be a synthetic
   branch-structured discrete optimization functional. EXP-MC-001,
   EXP-DERIV-001, and EXP-DIST-001 test its internal formula, derivative,
   assembly, and distribution consistency; none is relabeled as independent
   physical validation. Introducing a physical constitutive claim reactivates
   this block and additionally requires a matched incremental return-map or an
   analytical physical benchmark.

6. **Physical relevance, only if claimed.** Add single-material-point
   projection tests and at least one recognized boundary-value or limit-analysis
   benchmark with matched constitutive assumptions. If this is not done,
   classify the endpoint potential explicitly as a synthetic branch-structured
   benchmark rather than mechanics validation.

### Required Metrics

- observed manufactured-solution convergence order;
- weighted \(L^2\), \(H^1\), displacement, strain, and stress errors as
  appropriate;
- energy, gradient, HVP, reaction, and full-state differences;
- residuals evaluated independently by both implementations when a comparator
  is retained;
- comparator code version, license, command, and raw outputs when applicable.

### Acceptance Criteria

- Manufactured cases achieve the expected asymptotic rate within a
  prespecified margin, such as 0.25 in observed order.
- Identical discrete implementations agree under scaled normwise tolerances
  derived from conditioning, problem scale, solver tolerance, and deterministic
  or nondeterministic MPI reduction order. Near-machine-epsilon equality is not
  assumed across different assembly orders.
- Every retained external comparison uses identical definitions. Unmatched
  comparisons remain explicitly qualitative and cannot support validation;
  omitting all cross-backend claims is an admissible scope decision.
- Regression gates are not described as validation tolerances.
- Every retained comparator is legally distributable and included in the final
  archive. If the existing endpoint comparator cannot be archived, replace it
  or remove it as a central baseline.

### Manuscript Consequences

- A different-quadrature Ginzburg--Landau result remains a formulation check,
  not identical-functional validation.
- The current JAX-FEM result remains a companion-model check unless the law and
  discrete problem are matched.
- The required manuscript scope relies on independent manufactured and analytic
  verification, not on a DOLFINx comparison. The optional DOLFINx path remains
  blocked by the local ADIOS2 ABI mismatch and requires explicit authorization
  before any dependency repair.
- Replace "agreement at plotted scale" with numerical norms.
- Do not use `validated against` or physical-validation wording unless the
  corresponding optional evidence passes.

---

## Step 7. Run EXP-ROUTE-001 Tier B: The Headline Derivative-Route Confirmation

**Type:** `EXPERIMENT`  
**Priority:** `P0` when this Plasticity3D case is retained; otherwise
`REMOVED FROM SCOPE`  
**Depends on:** Steps 5--6  
**Compute:** `L` (45.00 optional node-hours); obtain approval before the HPC run

### Research Question

At equal verified accuracy on fully documented hardware, is constitutive AD
faster than element AD for the selected high-order Plasticity3D confirmation,
after both routes reach the same endpoint-accuracy contract?

### Frozen Case

- \(P_4(L_1)\);
- \(\lambda_{\mathrm{sr}}=1.55\);
- 8 and 32 MPI ranks on one documented Karolina CPU node;
- identical mesh, quadrature, boundary conditions, initial state,
  preconditioner, globalization, inexact-solve rule, and calibrated stopping
  criteria;
- element AD and constitutive AD. Colored recovery is a visible prespecified
  non-attempt motivated by pilot memory risk; no resource threshold is claimed.

### Prepared Execution Path

The Karolina matrix, paired executor, and strict endpoint analyzer implement:

1. enforce the common residual and correction gates;
2. use randomized balanced route order;
3. save full final states and representative iteration states;
4. record ten independent cold-process paired blocks per route and rank;
5. alternate the hash-seeded route order exactly five-first/five-second;
6. retain every rank-local wall time and its MPI collective maximum;
7. capture the full timing and resource schema from Step 4;
8. retain failures and censored runs;
9. run `experiments/analysis/analyze_plasticity3d_route_endpoints.py` before
   exposing any timing.

The representative low-order confirmation is `P1(L1)` at 8 ranks. Execution is
defined entirely by `campaign_matrix.csv`; the historical derivative-ablation
workflow is not a publication path.

### Statistical Analysis

Report:

- all individual observations;
- median and interquartile range;
- bootstrap 95% intervals for medians and pairwise time ratios;
- coefficient of variation;
- cold-process results, with compilation and setup reported separately;
- Newton, Krylov, derivative, assembly, and preconditioner work;
- full-state, residual, HVP, and branch-map differences.

Treat ten observations as a pilot minimum, not an automatic sample-size
justification. Define allocation/process blocks and distinguish repeated solves
within one compiled process from between-process and between-allocation
variation. Predeclare the element-versus-constitutive timing contrast and a
minimum practically relevant speedup in addition to statistical significance.

### Acceptance Criteria

- Every retained run reaches identical residual and correction thresholds.
- Route-to-route derivative, branch, and endpoint differences pass Step 5.
- Timing metadata identify processor, software, affinity, threads, and JIT
  policy completely.
- Claim one route is faster only when the prespecified primary or
  multiplicity-adjusted time-ratio interval excludes 1 and exceeds the minimum
  practically relevant effect.
- If the comparison is statistically unresolved, say so and remove the exact
  ranking from the abstract.
- The historical 222/288/372-second numbers may appear only as historical
  context unless reproduced by this campaign.

### Failure Action

If correctness fails, repair the route before retiming. If timing is
inconclusive, report statistical equivalence or scope the result to the
observed configuration. Never retain a preferred ranking because it matches the
historical result.

---

## Step 8. Run EXP-ROUTE-001: Derivative-Route Crossover And Predictive Cost Model

**Type:** `EXPERIMENT`  
**Priority:** `P0` for the selected route/cost contribution  
**Depends on:** Steps 5--6 and at least one retained, verified route-comparison
case; Step 7 only when Plasticity3D remains the headline  
**Compute:** `L`

### Objective

Replace the output-entry-count argument with a model and experiment showing
when each derivative route is preferable.

### Frozen Analysis Contract

The exact 13-feature, no-intercept OLS model, train/holdout split, minimum row
and group counts, condition-number gate, inverse-log percentage errors, tie
band, and terminal decisions are frozen in
`paper/protocols/EXP-ROUTE-001-analysis-contract.json`.

### Analytical Cost Model

Include at least:

- element degrees of freedom \(m_e\);
- constitutive dimension \(s\);
- quadrature count \(n_q\);
- local and maximum color count;
- local elements, owned rows, and overlap size;
- AD/HVP work;
- \(B_q^\top C_qB_q\) contraction cost;
- sparse insertion and communication;
- compilation and cache reuse;
- peak memory and allocation volume as observed capacity diagnostics, not as
  predictors in the frozen timing selector;
- rank count and imbalance.

The comparison \(m_e^2\) versus \(n_qs^2\) may be one term, but it is not a
complete cost model because all routes must ultimately provide the same owned
sparse operator or operator action.

Because the fitted response is an MPI collective maximum, contract version 2
uses the maximum rank-local overlap workload. With
\(n_{e,r}^{\mathrm{loc}}\) overlap elements and \(c_r\) colors on rank \(r\),
the single route-work feature is `log1p` of

\[
\chi_{\mathrm{elem}}=\max_r n_{e,r}^{\mathrm{loc}}n_qm_e^2,
\quad
\chi_{\mathrm{color}}=\max_r(n_{e,r}^{\mathrm{loc}}c_r)n_qm_e,
\quad
\chi_{\mathrm{const}}=\max_r n_{e,r}^{\mathrm{loc}}n_q
  (s^2+s^2m_e+sm_e^2).
\]

These are structural operation-shape counts, not exact FLOP or compiler-cost
models. The constitutive proxy includes the dense \(s\)-space tangent and both
contractions in \(B_q^\top C_qB_q\). The feature count remains 13.

### Experimental Design

Use three stages:

1. **Fixed-state screening:** measure one verified Hessian construction/action
   at identical stored states for \(P_1(L_1)\), \(P_1(L_2)\), \(P_2(L_1)\),
   and \(P_4(L_1)\), representative smooth and branch-structured laws, and
   ranks such as 1, 8, and 32 where feasible. Use at least five warm repetitions
   per point.
2. **Prespecified full-solve ordering confirmation:** run ten paired blocks at
   the fixed representative low- and high-order cases in the reviewed matrix,
   using identical accuracy and solver policy. These rows confirm ordering at
   their declared points; they are not predicted-crossover confirmations.
3. **Conditional post-fit crossover confirmation:** only if the untouched
   holdout admits the selector, select candidate crossover points from the
   frozen fitted model without inspecting any confirmation outcome. Freeze a
   second, hash-bound matrix and analysis contract, obtain a separate human
   release, and execute those cases as a new campaign. Without this third
   stage, the paper may claim a held-out timing selector and observed route map
   but not an experimentally confirmed crossover location.

Separate degree, quadrature, mesh, constitutive complexity, and rank effects.
The current `p3d_derivative_degree` section of
`run_paper_reviewer_gap_experiments.py` is a useful pilot matrix, but its
one-Newton, single-observation rows are not the final crossover campaign.

Degree alone is not an identifiable experimental factor because it changes
element size, quadrature count, color count, contraction work, and memory
together. Add independently varied microbenchmarks: synthetic constitutive
kernels at fixed element layout, decoupled quadrature, fixed sparsity patterns
with varied colorings, and fixed local kernels with varied overlap/rank. Define
a train/holdout split before fitting the model. If the selection rule is meant
to generalize beyond one CPU family, validate it on a second hardware
architecture; otherwise state the platform scope explicitly. The replicated
synthetic component benchmark is a descriptive mechanism diagnostic only; its
non-route-faithful stage times are not selector features.

### Required Metrics

- cold and warm timing decomposition;
- color counts and recovery probes;
- contraction, insertion, communication, and Krylov costs;
- tracked and measured memory;
- derivative equivalence at every point;
- fitted model coefficients and held-out prediction error.

### Acceptance Criteria

- Every route remains derivative-equivalent at each comparison point.
- Failed routes remain visible as censored results; unattempted routes motivated
  only by pilot memory risk are labeled non-attempts without a threshold claim.
- A predictive selection rule is claimed only if a prespecified model predicts
  held-out route ordering or crossover within its declared error band.
- A crossover-location claim additionally requires the separately released
  post-fit confirmation stage; the current optional Tier-B matrix alone cannot
  support that claim.
- If prediction is weak, publish a finite empirical map and restrict conclusions
  to the tested configurations.
- If the clean train/holdout coverage or prediction gates fail, remove the
  selector claim rather than substituting the synthetic factor proxy.
- The final discussion states conditions, not a universal route ranking.

---

## Step 9. Run EXP-GLOB-001: Exact And Controlled Globalization Evidence

**Type:** `THEORY AND EXPERIMENT`  
**Priority:** `P1`  
**Depends on:** Steps 4--5  
**Compute:** `M--L`

### 9.1 Specify Reproducible Algorithms

Use the exact solver contracts already frozen in Step 4 to replace the current
common policy sketch with separate algorithms for every reported method. If any
rule is still undefined, return to Step 4 before running this comparison:

1. inexact Newton with Armijo backtracking;
2. Steihaug trust-region Newton;
3. hybrid trust-region plus post-subproblem line search;
4. any residual-bisection or continuation method retained in a central case.

For each algorithm, specify:

- objective or merit function;
- gradient and model Hessian;
- norm used for the trust region;
- inner Krylov stopping condition and forcing parameter;
- negative-curvature and non-descent handling;
- exact trial-step construction;
- exact acceptance inequalities;
- trust thresholds and radius decrease/increase factors;
- boundary test for radius expansion;
- rejection and retry logic;
- line-search interpolation or contraction rule;
- NaN/Inf and failed-linear-solve behavior;
- maximum retries;
- residual, correction, stagnation, and iteration-cap exits;
- meaning of each nonlinear and linear iteration counter.

Replace terms such as `normally accept`, `configured`, `approximately
minimize`, and `sufficiently accurate` with formulas or named implementation
choices. Add tests for threshold equality, negative curvature, rejected trials,
failed linear solves, and roundoff termination.

### 9.2 Separate Two Scientific Questions

Run and report two explicitly different tiers:

1. **Controlled algorithmic comparison.** Hold the discrete model, Hessian
   action, preconditioner, forcing schedule, starting point, and final accuracy
   fixed wherever mathematically possible. This tier may support a conclusion
   about globalization.
2. **Production policy-bundle comparison.** Permit GMRES/CG for line search and
   STCG for trust-region subproblems when appropriate, but describe the result
   only as a comparison of complete bundles.

Do not attribute a difference to globalization when the linear algebra changes
at the same time.

### 9.3 Experimental Matrix

Use a bounded set of loads or starting states for:

- one smooth convex scalar problem;
- nonconvex Ginzburg--Landau;
- one hyperelastic problem;
- the branch-structured problem only if Step 3 established compatible
  regularity.

Use at least five timing repetitions per case and method after correctness is
established. The existing `run_globalization_method_compare.py` and the
`gl_globalization` section of the reviewer-gap runner are starting points, not
final evidence.

Equal residual tolerances do not guarantee that nonconvex methods reach the same
stationary point. Cluster endpoints by energy, branch observables, and weighted
state norm. Compare time only within the same endpoint class, or report success
profiles over a prespecified set of loads and initializations. Timing
repetitions estimate machine noise; robustness conclusions require distinct
problem instances, loads, or starting points as the statistical units.

### Required Metrics

- success, failure, cap, and timeout rates;
- function, gradient, Hessian/HVP, and preconditioner evaluations;
- accepted and rejected steps;
- line-search evaluations and trust-radius history;
- negative-curvature events;
- Newton and Krylov work;
- scaled residual history and final accuracy;
- state, energy, and time distribution.

### Acceptance Criteria

- Every successful timing reaches the same final accuracy contract.
- An independent reader can reproduce every accept/reject decision from the
  algorithm, parameter table, and stored history.
- Claims about globalization alone use only the controlled tier.
- Time-capped rows retain work counters and the last state.
- Fixed-iteration experiments with materially different energies or residuals
  are labeled `fixed_work`, not used as solver rankings.
- If this study is not central to the contribution selected in Step 1, move it
  to the supplement rather than expanding the main text.

---

## Step 10. Run EXP-DISC-001: Separated Degree, Mesh, Quadrature, And Tolerance Studies

**Type:** `EXPERIMENT`  
**Priority:** `P1`  
**Depends on:** Steps 4--6  
**Compute:** `M` (13.25 prepared node-hours)

### Objective

Replace the historical nine-case endpoint trend, which changes finite-element
degree, mesh, quadrature, and effective algebraic accuracy simultaneously, with
experiments that isolate these effects.

### Implementation Prerequisite

Named 1-, 11-, 24-, and positive streamed 125-point rules, explicit runner
selection, caches, result metadata, and a common 24/125-point evaluator are
implemented. Mandatory common-rule evaluation failures propagate to the outer
row. The schema-v2 adjudicator is implemented and tested: it requires the
complete six-row family; archives residual vectors, actions, and pointwise
branch maps; checks own-rule and common-rule residuals; and cryptographically
binds each case to job metadata, environment, execution log, Slurm stdout and
stderr, and settled accounting. The remaining work is execution and admission
of the frozen six-row sequence: P4(L1)
smoke, P4(L1) 24-point solve, P4(L1) 125-point solve, P4(L2) 24-point solve,
P4(L2) 125-point solve, and the tight-tolerance P4(L1) row. A strict
discretization decision must admit every row before interpretation.

### Phase A: Quadrature Convergence

For each retained degree:

1. fix the mesh, nonlinear route, solver, initial state, and stopping contract;
2. increase quadrature order until consecutive enriched rules agree;
3. evaluate all endpoint energies and observables with one common
   over-integrated reference rule;
4. record changes in state, stress, branch map, energy, and work.

Select the production quadrature only after this phase. A recommended initial
gate is less than 0.1% change in reference-evaluated energy and principal
observables under one further enrichment, but freeze a scale-aware value
before the final run.

### Phase B: Mesh Refinement At Fixed Degree

For each scientifically retained degree:

1. hold the verified quadrature policy fixed;
2. use the mesh-aware residual from Step 4;
3. solve below the estimated discretization error;
4. compare states in a common projected space;
5. report energy, work, displacement, stress, branch fractions, and error
   relative to the finest verified reference.

### Phase C: Degree Comparison

Only after Phases A and B, compare \(P_1\), \(P_2\), and \(P_4\) on comparable
meshes or comparable degrees of freedom. Hold quadrature accuracy, model,
boundary conditions, continuation, and stopping interpretation fixed.

### Phase D: Nonlinear Tolerance Sensitivity

At one representative discretization, tighten the nonlinear and KSP accuracy
by an order of magnitude. Demonstrate that the reported discretization trend is
not an algebraic stopping artifact.

### Acceptance Criteria

- Quadrature, mesh, degree, and nonlinear accuracy are varied in separate
  blocks.
- Every comparison reports a mesh-meaningful residual and common-reference
  observable.
- Convergence language is used only when a consistent error reduction is
  demonstrated.
- If nonconvex branch changes prevent a convergence interpretation, call the
  result an endpoint-sensitivity study and do not use it as accuracy evidence.
- If this redesign is too expensive or peripheral to Step 1, remove the
  nine-case study from the main paper instead of retaining it with a disclaimer.

---

## Step 11. Run EXP-TOPO-001 Or Remove Topology From The Core Paper

**Type:** `DECISION GATE, THEORY, AND EXPERIMENT`  
**Priority:** `P0` for SIOPT; `P2` for the SISC fallback  
**Depends on:** Steps 1, 3--6  
**Compute:** `L--XL` if retained

### 11.1 Audit The Existing Problem Contract

The historical implementation mixed material measure and normalized material
fraction. On the area-2 domain, a material measure of 0.4 is a normalized
fraction of 0.2; treating both as the same quantity invalidated feasibility and
rank-parity interpretations. This defect is corrected. The maintained API and
records now expose `target_material_measure` and
`target_normalized_fraction`, enforce their consistency, use a canonical
normalized internal target, and test serial, parallel, and initialization
parity. The legacy `--volume_fraction_target` alias remains only as an explicit
normalized-fraction compatibility path and is absent from publication runner
commands. Before any future optimization campaign, rerun these semantic tests
and verify that every archived quantity carries its unit/normalization field.

Freeze one discrete constrained problem, for example

\[
  \begin{aligned}
    \min_{\theta_h}\quad &
      \mathcal C_h(\theta_h)+\alpha_{\mathrm{reg}}R_h(\theta_h),\\
    \text{subject to}\quad &
      M_h(\theta_h)=M_\star,\\
    & \theta_{\min}\leq\theta_h\leq 1,\\
    & \theta_h=1\quad\text{on prescribed pads}.
  \end{aligned}
\]

Fix the final \(p_{\mathrm{SIMP}}\), regularization, filter or phase-field
length, mesh, loads, pads, bounds, and initial design.

Before interpreting KKT residuals, verify feasibility of the material target,

\[
  M_{\min}\leq M_\star\leq M_{\max},
\]

where the bounds account for fixed pads, density floors, and all prescribed
design values. State and verify an appropriate constraint qualification for the
selected finite-dimensional formulation.

### 11.2 Derive And Verify The Reduced Problem

1. Decide whether the optimization variable is physical density or a latent
   logistic variable. If latent, discuss saturation and the relationship
   between latent-space and density-space stationarity. A small latent gradient
   caused by a nearly singular sigmoid Jacobian is not a density-space KKT
   certificate. Choose one variable representation and define feasibility,
   projection, and complementarity consistently in that representation.
2. Derive the exact reduced compliance gradient, the design-map chain rule, and
   the volume derivative.
3. Verify both by directional Taylor tests.
4. Define and implement:

   - relative material feasibility;
   - bound violation;
   - projected Lagrangian-gradient or KKT residual;
   - complementarity residual;
   - mechanics residual;
   - design-step norm.

5. Require continuation to reach the fixed final \(p_{\mathrm{SIMP}}\).
   A run stalled at a smaller exponent is intermediate or failed, not a solution
   of the final problem.

### 11.3 Analyze Or Replace The Frozen Reciprocal Model

Prove its value and first-order consistency at the current design. Determine
whether it is a majorizer, local upper model, tangent model, or heuristic. The
analysis must cover the complete subproblem model or Lagrangian, including the
multiplier term, move term, regularization, and continuation state. Consistency
of the compliance term alone does not justify a controller whose other terms
change. The phase-field term and logistic map may reintroduce nonconvexity even
when the reciprocal term is separably convex in positive density.

If the method is the paper's SIOPT contribution:

- replace the empirical quantile controller with a fully specified sequential,
  trust-region, or augmented-Lagrangian algorithm;
- use actual-versus-predicted reduction or a justified filter test;
- state assumptions on positive densities, smooth state maps, mechanics
  accuracy, multiplier updates, and model decrease;
- provide a convergence theorem or a rigorous stationarity result.

If no new method is claimed, use an established optimizer as the main method
and present the reciprocal construction only as a secondary model.

### 11.4 Run A Matched Baseline Campaign

Compare with at least one accepted constrained topology method, such as optimality
criteria, MMA, projected or augmented-Lagrangian optimization, or an appropriate
PETSc TAO method. Avoid a new dependency unless justified and licensed.

Use:

- identical objective, constraints, design map, initialization, and mechanics
  accuracy;
- the same final \(p_{\mathrm{SIMP}}\);
- common feasibility and KKT tolerances;
- at least three meshes with physically consistent regularization length;
- ranks 1, 2, 4, 8, 16, and 32 under one deterministic schedule where
  resources allow.

Because the problem is nonconvex, include a small prespecified set of feasible
initial designs or deterministic perturbations when making an algorithmic
robustness claim. Report distributions of feasible objective and KKT measures.
Keep mesh refinement at a fixed rank separate from rank consistency on a fixed
mesh unless resources explicitly support a full factorial design.

### Metrics

- feasible compliance and regularization term;
- relative material and bound violations;
- projected KKT and complementarity residuals;
- mechanics and adjoint errors;
- iterations, rejected updates, final continuation value, and time;
- weighted density differences and feature measures across ranks and meshes.

### Recommended Prespecified Gates

- exact-gradient Taylor remainder with second-order behavior;
- relative material violation at most \(10^{-3}\), or another justified
  optimization tolerance;
- projected KKT residual in the \(10^{-4}\)--\(10^{-3}\) range after scaling,
  fixed before the final campaign;
- identical final model parameters before comparing compliance;
- competitiveness claimed only under a predeclared feasible-compliance margin;
- rank consistency claimed only when feasibility, KKT, compliance, and density
  differences all meet declared thresholds.

Define the norm, denominator, and safe scale for every tolerance. The numerical
values above are planning targets, not dimensionless definitions by
themselves.

### Decision Outcomes

- **Core SIOPT result:** retain only if the method and experiments satisfy the
  mathematical, feasibility, stationarity, baseline, and reproducibility gates.
- **Supplementary software demonstration:** retain as a distributed design-loop
  stress test without KKT, convergence, or solution claims.
- **Remove:** use this option if the section does not serve the central paper
  after the main contribution is frozen.

### Hard Gate

For a SIOPT submission, failure to make topology rigorous or to establish
another optimization contribution returns the project to Step 1. Existing
endpoints at different material fractions and continuation exponents cannot be
used as optimization evidence.

---

## Step 12. Run EXP-SCALE-001: Controlled Scaling, Memory, And Deployment

**Type:** `EXPERIMENT`  
**Priority:** conditional `P1`; `REMOVED FROM SCOPE` if no scaling result is
retained  
**Depends on:** correctness, validation, stopping, and policy gates for each
retained scaling case; removed or unrelated Steps 7--11 do not block it  
**Compute:** `L`; 22.50 node-hours if retained plus 17.50 optional node-hours;
obtain approval

This step does not block the selected derivative-placement contribution. Its
matrix rows become required only after the authors retain a scaling claim; in
that case the fixed-policy evidence must pass every gate below.

### Current Implementation Status

The machine-readable analysis contract, strict analyzer, and accounting
collector are implemented and tested. The analyzer treats required
Hyperelasticity and optional Plasticity3D as separate evidence families,
requires every declared grid point and five independent process repetitions,
reconstructs collective maxima from raw rank timings, checks common commit and
policy, admits endpoint equality and accuracy before time, and reports paired
bootstrap intervals, speedup, and node-based efficiency only after all gates
pass. Accounting collection is offline by default; live `sacct` access requires
an explicit option and is intended only after jobs have settled. No scale job
has been submitted or run, so this infrastructure currently supports no scaling
claim.

### Preconditions

Do not launch scaling campaigns until the canonical derivative route,
globalization, preconditioner, stopping rule, and problem definition are frozen.
Scaling an invalid derivative, unequal-accuracy solve, or changing optimization
problem cannot repair the scientific evidence.

### Separate Two Series

1. **Fixed-policy strong scaling.** Keep mesh, discrete problem, coarse-grid
   policy, preconditioner hierarchy, solver parameters, accuracy, and output
   scope fixed across ranks.
2. **Tuned production scaling.** Permit rank-dependent coarse groups or solver
   tuning, but label the result as deployment evidence rather than conventional
   strong scaling.

Changing coarse groups at high rank counts must not be hidden inside the
fixed-policy series.

### Retained Cases

Choose only cases serving the central contribution, for example:

- hyperelasticity \(L_5\), a fixed-policy P1 first-step viability series only;
- one completed high-order Plasticity3D endpoint if Steps 3--7 pass;
- topology only if Step 11 produces rank-consistent KKT-quality endpoints.

### Protocol

- Use at least five repetitions per rank after a pilot.
- Randomize rank-case scheduling where the batch system permits it.
- Report one-node and multi-node results separately.
- Use identical accuracy and compare full states across ranks.
- Record setup, assembly, communication, coarse solve, preconditioner, Krylov,
  globalization, and total time.
- Report median, interval, speedup, and efficiency.
- Compare replicated and rank-local construction at identical states.
- Retain changes in iteration count as part of the result.

### Memory Protocol

Report:

- peak per-rank and per-node RSS;
- proportional set size where available;
- tracked allocated arrays and matrix storage;
- local element and overlap counts;
- imbalance across ranks.

Do not interpret summed RSS as exact total memory because shared pages may be
counted repeatedly.

### Acceptance Criteria

- Fixed-policy scaling contains no hidden algorithmic changes.
- Every rank satisfies the same residual and endpoint tolerances.
- Scaling adjectives are tied to an explicitly reported efficiency threshold.
- Rank-local memory claims use per-node/PSS or tracked allocation evidence, not
  summed RSS alone.
- Tuned results are presented separately from fixed-policy results.
- If the selected paper no longer makes a scaling contribution, retain only the
  smallest experiment needed to demonstrate distributed viability.

---

## Step 13. Adjudicate Every Claim Before Rewriting The Manuscript

**Type:** `SCIENTIFIC GATE`  
**Priority:** `P0`  
**Depends on:** Steps 3--12  
**Compute:** `S`

### Objective

Decide what the completed evidence actually supports. The manuscript must be
rewritten from this adjudicated evidence, not from the intended story.

### Tasks

1. Complete one row per atomic claim using
   `docs/codex/templates/claim_evidence_table.md`:

   - exact claim;
   - intended manuscript location;
   - proof, citation, experiment, or artifact;
   - evidence level;
   - uncertainty;
   - limitation;
   - decision: keep, narrow, move, or delete.

2. Apply explicit word contracts:

   - `termination criteria satisfied`: the declared residual and secondary
     termination gates passed for one run;
   - `converged algorithm` or `convergent method`: supported by a theorem or a
     documented convergence history, not merely one terminal residual;
   - `approximate stationary point`: a defined first-order residual passed;
   - `approximate local minimizer`: first-order stationarity plus a justified
     second-order condition passed;
   - `global minimizer`: convexity, a global optimality certificate, or a
     rigorous global bound supports the claim;
   - `best observed point`: empirical comparison only, with no optimality
     implication;
   - `equivalent`: direct derivative or mathematical equivalence passed;
   - `validated`: an independent matched validation passed;
   - `faster`: the prespecified time-ratio interval supports the ordering;
   - `scalable`: a defined efficiency measure supports the adjective;
   - `optimization solution`: feasibility and KKT gates passed.

3. Record negative and ambiguous outcomes. A failed route, absent crossover,
   nonconverged topology run, or statistically unresolved timing comparison is
   a result, not an artifact to remove.

4. Keep historical and new campaigns visually and textually distinct. Do not
   merge measurements with different hardware or stopping contracts into one
   performance table.

5. Decide the final paper path:

   - SIOPT Route A, B, or C from Step 1;
   - SISC fallback;
   - halt if no path has adequate evidence.

### Acceptance Criteria

- Every abstract, introduction, discussion, and conclusion claim maps to a
  proof, archived result, or verified primary citation.
- Every numerical claim has uncertainty and a stopping contract where needed.
- Unsupported or redundant results are removed rather than defended with
  repeated disclaimers.
- The final central contribution is unchanged by deleting noncentral examples.

---

## Step 14. Refocus And Shorten The Manuscript

**Type:** `MANUSCRIPT`  
**Priority:** `P0`  
**Depends on:** Step 13  
**Compute:** none

### Required Style Preparation

Before manuscript edits, read in order:

1. `paper/style_guide/AGENTS.md`;
2. `paper/style_guide/style_fingerprint/agent_quick_reference.md`;
3. `paper/style_guide/style_fingerprint/agent_cookbook.md`.

Preserve the manuscript's current `cleveref` and `\Cref` conventions where
they intentionally differ from the generic fingerprint.

### Recommended Main-Text Structure

1. Introduction and precise research questions.
2. Common discrete optimization or stationarity setting.
3. Derivative routes and mathematical equivalence.
4. Distributed cost and memory model.
5. Verification and validation protocol.
6. Primary controlled experiments.
7. Secondary solver-policy or scaling evidence.
8. Limitations and applicability.
9. Conclusions.
10. Supplementary mathematical details and noncentral diagnostics.

### Scope Reduction

For a derivative-placement paper, retain in the main text:

- one smooth reference problem;
- one branch-structured or nonconvex problem;
- one large distributed performance case;
- topology only if Step 11 passes as a central contribution.

Move standard model derivations, extensive material branch algebra, secondary
multigrid variants, fixed-work tables, and noncentral benchmark states to the
supplement. Consolidate the three overlapping architecture figures into one,
and combine the early capability/protocol tables into one experimental-design
table.

The historical broad draft has already been replaced by a compact working
manuscript. Preserve that compression and add displays only when admitted
evidence answers a central research question; coherence and evidentiary
necessity are more important than an arbitrary page count.

### Narrative Rules

- Begin with the optimization or computational bottleneck, not the software
  inventory.
- Introduce assumptions and definitions before dense notation.
- Put derivative verification before performance.
- Keep validation conceptually separate from timing and scaling.
- Interpret every retained display in body text.
- State limitations once in a dedicated, precise form rather than repeatedly
  interrupting the argument.
- Avoid campaign names, local paths, repository labels, and internal method
  spellings in manuscript prose.
- Use affirmative, evidence-scoped conclusions instead of defensive rankings.

### Acceptance Criteria

- A reader can state the paper's thesis after reading the abstract and first
  two introduction pages.
- Every main-text benchmark is necessary to answer a research question.
- Numerical results appear once in the section where they are interpreted.
- Removing the supplement leaves a complete proof/evidence chain for the
  central contribution.

---

## Step 15. Rewrite And Verify Every Manuscript Section

**Type:** `MANUSCRIPT`  
**Priority:** `P0`  
**Depends on:** Step 14  
**Compute:** none

Use the following section contracts. Do not move to the next section until the
current contract passes a claim and notation check.

### 15.1 Title And Abstract

**Required content:** exact problem class, central method/result, principal
assumptions, strongest replicated evidence, and reproducibility statement.

**Remove or avoid:** broad `toolset` novelty unless the paper is explicitly a
software paper; exact 222/288/372-second values unless Step 7 reproduces them;
repository inventory; unsupported universality.

**Acceptance:** every abstract sentence maps to Step 13, and the optimization
advance is visible without reading the implementation section.

### 15.2 Introduction

1. Motivate the optimization or derivative-placement bottleneck.
2. Define the gap relative to the closest optimization, AD, finite-element, and
   sparse-recovery literature.
3. State no more than three research questions.
4. List contributions in the same order as the paper supplies proof and
   evidence.
5. State reproducibility and limitations concisely.

Refresh the Step 1 primary-source literature search when this section is
revised. Include inexact Newton/Krylov forcing, trust-region Newton-CG,
matrix-free versus assembled second-order methods, consistent-tangent AD,
distributed sparse recovery, and topology algorithms only insofar as they
support the chosen contribution. This is a freshness check; the novelty
decision must already have been made in Step 1.

### 15.3 Mathematical Framework

- Define the discrete functional/residual, affine constraints, quadrature, and
  free-variable map before derivative routes.
- Insert the propositions and assumptions from Step 3.
- Classify nonsmooth sets and state the applicable stationarity notion.
- Define all residual and correction norms from Step 4.
- Integrate displayed equations into punctuated prose.

### 15.4 Computational Construction

- Describe rank-local ownership, ghost dependencies, structural recovery, and
  constitutive contraction only to the depth needed for the scientific claim.
- Present the cost model before performance evidence.
- Separate AD-HVP recovery from finite-difference recovery.
- State JAX precision, compilation, distribution, and synchronization policy.
- Move API inventories and secondary engineering variants to the supplement or
  software documentation.

### 15.5 Benchmark Definitions

- Retain only central benchmarks.
- Give exact model, domain, boundary conditions, discretization, quadrature,
  solver, and stopping contract.
- Do not include result interpretation in a definition section.
- Use one consistent mesh notation, such as \(P_4(L_1)\), after defining the
  hierarchy.
- Distinguish continuum models from endpoint surrogates and fixed objectives.

### 15.6 Verification And Validation

Present evidence in this order:

1. Taylor and derivative-route checks;
2. distributed assembly equivalence;
3. manufactured or patch tests;
4. an optional matched cross-backend comparison, only if retained;
5. branch and nonsmooth diagnostics;
6. remaining qualitative companion comparisons.

For every comparator, state what is identical and what differs. Do not mix
validation thresholds with regression gates.

### 15.7 Numerical Results

- Answer one prespecified question per subsection.
- Give problem size, ranks, hardware, repetitions, stopping status, and timing
  scope with each central result.
- Report uncertainty and effect size, not only point estimates.
- Explain the measured mechanism using timing and cost-model components.
- Separate cold-start, warm solve, fixed-work, complete solve, strong scaling,
  and tuned deployment results.
- Retain failures and negative results when they delimit applicability.

### 15.8 Discussion

- State the supported decision rule for derivative placement.
- Explain where the cost model succeeds and fails.
- Distinguish conclusions supported by theory, verification, validation, and
  performance data.
- Discuss branch regularity, topology status, hardware specificity, and
  scalability limits once and precisely.
- Do not introduce new claims or unpublished observations.

### 15.9 Conclusion

- Answer the research questions in order.
- State the optimization or scientific-computing advance.
- Give the practical selection rule and its tested scope.
- Identify unresolved questions without restating a long limitation inventory.
- Ensure every quantitative statement already appears with evidence in the
  results.

### 15.10 Availability And Supplement

- State the licensed release, archive DOI, environment, and reproduction entry
  point.
- Put detailed branch identities, complete parameter tables, noncentral model
  derivations, policy sweeps, and full raw-result summaries in the supplement.
- Do not expose local paths or claim that restricted comparator material is
  public.

### Section-Level Acceptance Criteria

- Definitions precede use; symbols and acronyms are not overloaded.
- Normal article sentences and paragraphs remain reasonably compact.
- Every figure and table is interpreted in body text.
- Every strong claim is tied to an assumption, proof, citation, or archived
  numerical result.
- Status terminology is identical across abstract, results, discussion, and
  conclusion.

---

## Step 16. Regenerate The Minimal Figure, Table, And Supplement Surface

**Type:** `ARTIFACT`  
**Priority:** `P1`  
**Depends on:** Steps 14--15  
**Compute:** `S`

### Main-Text Display Plan

Prefer a small set of displays that directly answers the research questions:

- one unified architecture and derivative-route figure;
- one derivative-correctness/Taylor figure;
- one mathematical-status and experimental-protocol table;
- one headline replicated-performance table or figure with uncertainty;
- one route-crossover/cost-model display;
- one controlled solver-policy or scaling display if central;
- one topology KKT/baseline display only if Step 11 passes.

Move complete benchmark inventories, branch diagrams, full policy matrices,
secondary states, and raw tables to supplementary material.

### Generation Rules

1. Modify generation scripts before generated outputs.
2. Generate figures as vector PDF at final physical size.
3. Use bitmap layers only for fields or rendered meshes, at 600 dpi where
   required; keep labels, axes, legends, lines, and colorbars vector.
4. Match the manuscript font family and use absolute point sizes.
5. Use relative table widths, centered numerical columns, one quantity per
   column, units in headers, and precision supported by uncertainty.
6. Use LaTeX scientific notation rather than raw machine strings.
7. Show intervals or distributions wherever performance is compared.
8. Do not edit final generated assets manually.
9. Preserve source manifests and deterministic generation.

### Generation And Validation Sequence

```bash
make -C paper figures
make -C paper tables \
  REVISION_EVIDENCE_ROOT=/path/to/admitted/clean/evidence \
  REVISION_EVIDENCE_CLASS=publication \
  REVISION_EVIDENCE_MANIFEST=/path/to/source_evidence_manifest.json
make -C paper submission-bundle
make -C paper publish-check
make -C paper submission-check
```

### Acceptance Criteria

- Every included display supports a retained claim.
- Every generated display has a source and manifest entry.
- Every central number is linked to raw archived observations.
- Captions state conditions and quantities; interpretation remains in body
  text.
- Repeated generation is stable where intended.
- No display depends on an unarchived local source.

---

## Step 17. Prepare A Clean Release Candidate And Reproduction Bundle

**Type:** `ARTIFACT AND ARCHIVE`  
**Priority:** `P0`  
**Depends on:** Steps 13--16  
**Compute:** `M--L`

### 17.1 Assemble The Clean Revision Campaign

Publication-grade experiments must already have run from the clean, identified
experiment commits required by Step 2. This step packages those immutable
outputs into a release candidate; it does not rerun the campaign after claim
adjudication and manuscript rewriting. If an experiment must change here, return
to its original step, create a new protocol version, rerun it, and repeat Steps
13--16.

For every retained experiment, archive:

- experiment card and run manifest;
- exact command and standard output/error;
- scheduler script and job identifier;
- environment, affinity, and hardware record;
- mesh/config/input hashes;
- raw per-rank data and convergence histories;
- state snapshots needed for derivative or endpoint checks;
- generated table/figure inputs;
- validation result and known limitations.

### 17.2 Legal And Archival Preparation

1. Choose and add a root software license.
2. Audit licenses for external comparators and copied or adapted code.
3. Remove restricted literature full texts from the release.
4. Create an untagged or prerelease source and artifact candidate for review.
5. Include the complete new revision campaign, not only the historical July 8
   bundle.
6. Reserve a durable archival DOI if the archive supports reservation, but do
   not finalize the public deposit before Step 18 findings are resolved.
7. Prepare the provisional availability statement and deposit inventory.
8. Stage source, inputs, manifests, raw results, reproduction commands, and
   environment definitions for final deposit in Step 19.

### 17.3 Tiered Independent Reproduction

Download the deposited archive into a clean directory or second machine. From
that snapshot:

1. **Tier 1, mandatory raw-to-display reproduction:** build the environment,
   regenerate every retained table and figure from archived raw outputs, build
   the PDF, and verify hashes and archive-neutral paths.
2. **Tier 2, mandatory small numerical recomputation:** rerun all unit,
   derivative, Taylor, manufactured, and small MPI verification cases and
   compare them with archived tolerances.
3. **Tier 3, policy-defined HPC reproduction:** rerun at least one central
   performance case on the documented system when allocation permits. A full
   repetition of every `L`/`XL` campaign is desirable but may be declared
   optional if all raw observations, commands, and environments are archived.

State the achieved tier explicitly. Do not claim that a second environment
reproduced the complete numerical campaign when it only regenerated displays.

### Acceptance Criteria

- A root `LICENSE` or `COPYING` exists and covers the release candidate.
- Every main numerical row has the lineage: manuscript row -> generated table
  -> raw run -> command -> commit -> input hashes.
- Comparator code is legally archived or removed as a reproducible baseline.
- Tier 1 and Tier 2 independent reproduction pass; the achieved Tier 3 scope is
  recorded honestly.
- The provisional DOI and availability statement are ready for finalization
  after Step 18.
- All science, artifact, archive, and manuscript checks pass from the candidate
  snapshot. Because journal-template accommodation is excluded by scope,
  `make -C paper release-check` may remain blocked only by the explicitly
  excluded target-template/declaration gate; no other blocker may remain.

---

## Step 18. Run Three Independent Red-Team Reviews

**Type:** `REVIEW`  
**Priority:** `P0`  
**Depends on:** Steps 14--17  
**Compute:** none

Assign three reviews without asking reviewers to edit the same text during the
audit.

### Review A: Mathematical Correctness

Check:

- problem definitions, assumptions, signs, dimensions, and units;
- potential/stress identities and branch partitions;
- regularity and solver compatibility;
- derivative and recovery propositions;
- topology reduced gradient, KKT system, and convergence claims;
- notation consistency and all equation references.

### Review B: Numerical Evidence And Reproducibility

Check:

- direct derivative and Taylor evidence;
- stopping and KSP contracts;
- equal-accuracy comparisons;
- timing statistics and phase scopes;
- scaling controls and memory interpretation;
- failed-run handling;
- command, environment, raw data, and asset lineage.

### Review C: SIOPT Fit, Narrative, And Claim Discipline

Check:

- identifiable optimization contribution;
- distinction from the closest literature;
- logical section order and main-text focus;
- necessity of every benchmark and display;
- abstract/conclusion support;
- clarity, length, notation load, and style-guide alignment.

### Review Record

Create a response matrix with:

| Finding | Severity | Evidence | Action | Changed location | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- |

Classify severity as critical, major, minor, or editorial. A response such as
`clarified` is incomplete unless it identifies the evidence and exact change.

### Acceptance Criteria

- No critical or major finding remains unresolved.
- Every resolved mathematical or numerical finding has a focused test or
  evidence check where possible.
- A final skeptical review can identify the contribution, assumptions,
  termination meaning, and evidence without consulting repository history.

---

## Step 19. Complete Final QA, Tag, Archive, And DOI Deposit

**Type:** `FINAL QA`  
**Priority:** `P0`  
**Depends on:** all previous steps  
**Compute:** `S`

### Automated Checks

Run the focused new tests, including the campaign, discretization, scaling,
route endpoint/cost, synthetic-factor, and tranche-aggregation contracts, plus
at least:

```bash
./.venv/bin/python -m pytest tests/test_paper_revision_karolina_campaigns.py
./.venv/bin/python -m pytest tests/test_plasticity3d_discretization_analysis.py
./.venv/bin/python -m pytest tests/test_exp_scale_001_analysis.py
./.venv/bin/python -m pytest tests/test_plasticity3d_route_cost_model_analysis.py
./.venv/bin/python -m pytest tests/test_plasticity3d_route_endpoint_analysis.py
./.venv/bin/python -m pytest tests/test_route_factor_microbenchmarks.py
./.venv/bin/python -m pytest tests/test_route_tranche_manifest_aggregation.py
./.venv/bin/python -m pytest tests/test_paper_reviewer_gap_experiments.py
./.venv/bin/python -m pytest tests/test_submission_bundle_manifest.py
./.venv/bin/python -m pytest tests/test_paper_manuscript_hygiene.py
./.venv/bin/python -m pytest tests/test_paper_float_placements.py
./.venv/bin/python -m pytest tests/test_paper_table_semantics.py
./.venv/bin/python -m pytest tests/test_current_doc_paths.py
./.venv/bin/python -m pytest tests/test_docs_publication.py
make -C paper figures
make -C paper tables \
  REVISION_EVIDENCE_ROOT=/path/to/admitted/clean/evidence \
  REVISION_EVIDENCE_CLASS=publication \
  REVISION_EVIDENCE_MANIFEST=/path/to/source_evidence_manifest.json
make -C paper submission-bundle
make -C paper publish-check
make -C paper submission-check
make -C paper release-blockers
git diff --check
```

Also run `make -C paper release-check` as a diagnostic. Under the deliberately
template-neutral scope of this plan, it may fail only on the target-template or
venue-declaration blocker. Any scientific, license, DOI, archive, provenance,
or asset failure remains a release failure.

Add and run focused tests for the new derivative checks, KKT metrics, stopping
norms, statistical summaries, run schema, and archive manifests.

### Text Audit

Search the final manuscript for:

- `converged`, `stationary`, `minimizer`, `validated`, `equivalent`, `faster`,
  `scalable`, and `optimization solution`;
- raw scientific notation and unsupported precision;
- internal run labels, local paths, private hostnames, and code identifiers;
  retain publication-facing HPC system and processor names when they are needed
  for reproducibility;
- undefined acronyms and overloaded symbols;
- underscores in ordinary manuscript prose;
- claims in captions that are not interpreted in body text;
- results without a stopping status or evidence reference.

Verify every occurrence against the Step 13 word contracts.

### Visual PDF Audit

Render and inspect every page at the final intended size. Check:

- equation breaks, punctuation, and reference resolution;
- figure and table order;
- caption/body consistency;
- font family and readable absolute size;
- cropped axes, labels, colorbars, and legends;
- grayscale and color-vision readability;
- table widths, numerical alignment, units, and rounding;
- blank regions, isolated floats, and appendix transitions;
- PDF metadata, embedded fonts, and final page count.

### Release Consistency Audit

After Step 18 findings and all resulting corrections are resolved:

1. create the final clean commit and require an empty
   `git status --porcelain`;
2. regenerate the bundle and recompute manifest hashes from that commit;
3. create and verify the final release tag, requiring `HEAD` to equal the tag
   commit and the manifest commit;
4. finalize the archive deposit and DOI;
5. download the deposited archive and rerun the Tier 1 checks from Step 17;
6. confirm that the final PDF was built from the tagged archived source;
7. confirm that the DOI, release identifier, commit, and availability statement
   agree;
8. confirm that the archive contains every source used by a final figure or
   table and no restricted material.

### Final Acceptance Criteria

- All automated checks pass from the release snapshot.
- No unsupported abstract or conclusion claim remains.
- No failed, capped, stalled, or fixed-work run is mislabeled as converged.
- All main figures and tables regenerate from the archived raw evidence.
- The PDF passes page-by-page visual inspection.
- The deposited release, manuscript, DOI, and availability statement are
  mutually consistent.
- `git status --porcelain` is empty, and the checked-out commit, release tag,
  bundle manifest, and deposited archive identify the same source state.
- Any failure of `make -C paper release-check` is limited to the explicitly
  excluded journal-template/declaration gate; every scientific, artifact,
  license, archive, and DOI gate passes.

---

## Hard Stop/Go Rules

1. Failure of the potential or stress derivation blocks the corresponding
   plasticity claims and experiments.
2. Failure of derivative equivalence blocks every derivative-route timing
   claim.
3. Failure of the mesh-aware stopping contract invalidates affected endpoint,
   discretization, and timing comparisons.
4. Failure of a retained matched comparison blocks `validated against` wording
   and any associated cross-backend or physical-agreement claim; removing that
   claim satisfies the dependency for the selected paper.
5. Statistically inconclusive timing removes a resolved route ranking from the
   abstract.
6. Failure of topology feasibility or KKT gates removes topology from the core
   optimization evidence.
7. A scaling series with changing algorithmic policy is deployment evidence,
   not conventional strong scaling.
8. Expensive scaling begins only after correctness, validation, and stopping
   gates pass.
9. A green build or asset check never overrides a failed scientific gate.
10. If the SIOPT contribution gate fails, retarget the paper rather than
    overstating the evidence.

## Minimal Progress Record

Update this table whenever a step changes state. Link the evidence rather than
writing a project diary.

Before Step 2 is marked complete, replace every blank responsibility cell for a
`P0` step with an owner and a separate evidence reviewer.

| Step | Status | Evidence or blocker | Responsible person | Last updated |
| ---: | --- | --- | --- | --- |
| 1 | `DONE` | `paper/venue_and_contribution_decision.md`; derivative-placement route selected after current primary-source audit | Codex / internal independent-style mathematical and evidence reviews completed; author venue confirmation remains | 2026-07-10 |
| 2 | `IN PROGRESS` | baseline manifest, campaign-root isolation, strict run records, protocol cards, fail-closed table promotion, fresh-root/atomic partial-submission handling, a 24-source reviewed hash map, versioned human-authorization schema/example, and queued-job commit/matrix/source locks are implemented and tested; a clean immutable experiment commit and independent protocol signatures remain | Codex / independent evidence reviewer required | 2026-07-10 |
| 3 | `DONE` | conditional fixed-functional and rank-local owned-row recovery propositions are in the manuscript; Plasticity3D is a synthetic surrogate, Plasticity2D has a finite invariant apex convention, and topology is supplementary | Codex / mathematical red team completed; final clean-draft check required | 2026-07-10 |
| 4 | `IN PROGRESS` | scalar, Plasticity3D, and HyperElasticity Riesz maps pass implementation smokes with numerical inertia and true-residual checks; cross-mesh/tolerance calibration and clean promotion remain | Codex / independent numerical-protocol reviewer required | 2026-07-10 |
| 5 | `IN PROGRESS` | smooth, P1/P2/P4 branch-interior, all-five-branch, and canonical one/two-rank fixed-state diagnostics pass; the full one/two/four-rank colored owned-matrix/action gate and clean rerun remain | Codex / independent evidence reviewer required | 2026-07-10 |
| 6 | `IN PROGRESS` | independent p-Laplace, Ginzburg--Landau, affine hyperelastic, and four-level nonaffine hyperelastic diagnostics pass, including load-quadrature refinement; clean promotion remains. The matched DOLFINx comparison is removed from required scope; an optional future comparison remains blocked by the unapproved ADIOS2 ABI repair | Codex / independent verification reviewer required | 2026-07-10 |
| 7 | `IN PROGRESS` | the exact two-route Tier-B comparison and low-order confirmation are prepared, but execution waits for clean correctness/stopping gates, active Karolina allocation, and explicit authorization; no timing result exists | Codex / independent timing reviewer required | 2026-07-10 |
| 8 | `IN PROGRESS` | P0 for the selected paper: the 13-feature no-intercept train/holdout analyzer, paired hash-seeded all-route blocks, raw-rank/collective-max timing, replicated fixed-total-work mechanism diagnostics, four actions plus gradient/residual, direct feasible CSR checks, and strict endpoint analyzer are implemented fail-closed. The required route/cost subtotal is 105 Slurm rows, 273 in-allocation route/mechanism executions, and 64.20 node-hours; the optional Tier-B route tranche is 30 rows, 60 executions, and 45.00 node-hours. No cluster job was submitted and no selector is fitted | Codex / independent model/timing reviewer required | 2026-07-10 |
| 9 | `IN PROGRESS` | controlled Newton--Armijo and reduced-trust--Armijo algorithms and failure semantics are implemented; one diagnostic GL contrast exists, while clean Riesz endpoints, distinct instances, clustering, and repetitions remain | Codex / independent algorithm reviewer required | 2026-07-10 |
| 10 | `IN PROGRESS` | named 1/11/24/125-point rules, mandatory common evaluation, failure propagation, and the strict six-row schema-v2 adjudicator are implemented and tested. Every case must carry hash-bound job/environment/log/stdout/stderr/settled-accounting evidence. The fixed-state pilot rejects energy-only adequacy and records one near-switch P2 sample; the six solve rows are prepared but not run | Codex / independent discretization reviewer required | 2026-07-10 |
| 11 | `EVIDENCE COMPLETE` | material-measure/fraction semantics and one/two-rank corrected-unit diagnostics are complete for the declared supplementary software role; no KKT or optimization-solution claim is retained | Codex / final claim reviewer required | 2026-07-10 |
| 12 | `IN PROGRESS` | conditional P1: the scale contract/analyzer/accounting collector and Karolina scripts/cards/manifests are implemented and tested. Canonical v9 dry runs prepared 115 required commands, 30 optional Tier-B route commands, three optional P3D scaling commands, and a 12-block workstation plan, all with `source_dirty: true` and no submission. Real submission requires one explicit experiment, explicit tiers, a clean commit, a 24-source queued freeze, current allocation/account/QoS revalidation, and schema-valid human release authorization. This step may be removed without blocking the selected route/cost paper | Codex / independent HPC protocol reviewer required if retained | 2026-07-10 |
| 13 | `IN PROGRESS` | mathematical claim dictionary, historical rewrite audit, live execution report, and fail-closed generated evidence manifest exist; clean/cluster terminal outcomes remain | Codex / independent claim reviewer required | 2026-07-10 |
| 14 | `IN PROGRESS` | a focused working draft replaces the historical broad manuscript; final architecture still depends on whether held-out crossover evidence passes | Codex / independent scientific editor required | 2026-07-10 |
| 15 | `IN PROGRESS` | all manuscript sections have been rewritten to the style guide and diagnostic limitations are explicit; clean numerical replacement and final result-dependent edits remain | Codex / independent scientific editor required | 2026-07-10 |
| 16 | `IN PROGRESS` | managed 14-source finalization, semantic admission, exact four-table binding, path confinement, and independent byte regeneration are implemented and internally red-team tested; clean publication data remain | Codex / external reproducibility review required for the release candidate | 2026-07-10 |
| 17 | `BLOCKED` | clean commit, admitted experiments, licensed archive, refreshed bundle, release tag, and DOI do not yet exist | Codex / independent release reviewer required | 2026-07-10 |
| 18 | `IN PROGRESS` | internal simulated mathematical and reproducibility red-team reviews found theorem, evidence-promotion, route-design, accounting-binding, queued-job-drift, authorization, and reporting issues. The local text/implementation corrections and a dated disposition matrix are complete; genuinely independent release-candidate review remains mandatory | Codex / external independent reviewers still required | 2026-07-10 |
| 19 | `BLOCKED` | waits for resolved red-team findings and all prior gates | Codex / independent release reviewer required | 2026-07-10 |

## Final Definition Of Done

The revision is complete when a skeptical reviewer can answer all of the
following from the paper and its archive:

1. What precise optimization or scientific-computing problem is solved?
2. What is new relative to the closest methods?
3. Under which assumptions are the potential, stress, gradient, and tangent
   identities valid?
4. What happens at branch switches and repeated eigenvalues?
5. How were stationarity, feasibility, and convergence measured?
6. Do the derivative routes agree directly at identical states?
7. Were compared methods run to the same accuracy?
8. Are timing differences statistically supported and mechanistically
   explained?
9. Are topology endpoints feasible and approximately KKT stationary, if they
   are presented as optimization results?
10. Can every central number, table, and figure be reproduced from the licensed
    archive?

If any answer is missing, the corresponding step remains open.
