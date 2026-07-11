# EXP-ROUTE-001: Derivative-Route Cost Map on a Second CPU Architecture

## Status and decision

**Local finite-map diagnostic complete; expanded Karolina campaign prepared,
not submitted.** The local rows are dirty-worktree diagnostic pilots, not
publication evidence. The train/holdout split, cost-model features, error
thresholds, ordering gate, and structural-censor policy are frozen in
[`EXP-ROUTE-001-analysis-contract.json`](EXP-ROUTE-001-analysis-contract.json)
as contract version 2 before any Karolina result exists. The exact executable Karolina rows are
frozen in
[`campaign_matrix.csv`](../../experiments/runners/paper_revision_karolina/campaign_matrix.csv).
No predictive model may be fit unless the machine-readable preflight passes;
otherwise the terminal result is a finite empirical map only. The design
release gate is now implemented: every comparison block runs all active routes
as separate processes inside one Slurm allocation, derives a base permutation
from a frozen hash seed, balances route positions over independent allocations,
and records MPI collective-max timing with the underlying per-rank values. No
Karolina result yet exists.

The publication artifact directory remains named `analysis_contract_v1`
because it denotes machine-readable `analysis_schema_version: 1`, not the
scientific contract revision. The separately hash-bound
`EXP-ROUTE-001-analysis-contract.json` is `contract_version: 2`. The stable
artifact-layout name therefore does not imply that a version-1 scientific
contract is being applied.

## Scientific question

At an identical finite-element state and MPI decomposition, how do element AD,
colored sparse finite differences, and constitutive AD compare in Hessian
construction time, memory, and recovered tangent action? Does the route ordering
observed at fixed state predict full-solve ordering for selected high-order
cases?

The experiment does not assume that degree alone predicts cost. The analysis
must retain quadrature points, element DOFs, local/maximum color count, owned and
overlap DOFs, local element counts, MPI ranks, and matrix insertion work as
predictors. Peak rank memory and tracked allocations remain diagnostic outputs;
they are not predictors because they are observed only after running a route.

### Collective-max structural work proxy

The timing response is the MPI collective maximum, so its route-work proxy uses
the busiest rank's overlap workload rather than the global number of uniquely
owned elements. Let $n^{\mathrm{loc}}_{e,r}$ be the number of overlap elements
evaluated on rank $r$, $c_r$ its local color count, $m_e$ the element DOF count,
$n_q$ the quadrature count, and $s$ the constitutive dimension. Contract version
2 fixes the single structural feature to

\[
\begin{aligned}
  \chi_{\mathrm{elem}}
  &:=\max_r n^{\mathrm{loc}}_{e,r}\,n_qm_e^2,\\
  \chi_{\mathrm{color}}
  &:=\max_r\bigl(n^{\mathrm{loc}}_{e,r}c_r\bigr)\,n_qm_e,\\
  \chi_{\mathrm{const}}
  &:=\max_r n^{\mathrm{loc}}_{e,r}\,n_q
    \bigl(s^2+s^2m_e+sm_e^2\bigr).
\end{aligned}
\]

The first expression counts dense element-Hessian contribution shape, and the
second counts the vector work propagated by the exact AD-HVP seeds. In the
third expression, $s^2$ counts the dense constitutive tangent, while $s^2m_e$
and $sm_e^2$ count the two contractions used to form $B_q^\top C_qB_q$. These
are prespecified structural operation-shape counts, not exact FLOP, cache, or
compiler-cost models. The production OLS model uses $\log(1+\chi)$ as its one
route-work feature; no feature is added, so the design remains 13-dimensional.

## Frozen cases

### Tier A: identical fixed-state screening

- Problem: heterogeneous Plasticity3D, glued bottom, strength-reduction factor 1.55.
- Configurations: `P1(L1)` with the named one-point rule, `P1(L2)` with the
  one-point rule, `P2(L1)` with the named 11-point rule, and `P4(L1)` with the
  named 24-point rule.
- Routes: element AD, colored SFD, and constitutive AD, except that colored SFD
  at `P4(L1)` is prespecified as not attempted because of pilot memory risk
  and has no job. This is not an out-of-memory threshold claim.
- MPI ranks: 1, 8, and 32 on one Karolina node.
- States: analytic mesh field version 1 with amplitudes `2e-4` (elastic target)
  and `2e-2` (mixed-branch target).
- Split: rank 1 and rank 8 are training; rank 32 is untouched holdout. Existing
  workstation rows may be training/calibration only and can never be relabeled
  as holdout.
- Repetitions: three independent Slurm blocks per three-route comparison and
  four blocks per two-route `P4` comparison. Within each route process, one
  compilation/warmup assembly precedes five measured assemblies.
- Outputs: identical-state SHA-256, full tangent-action NPZ, gradient and matrix
  norms, wall times, callback decomposition, color counts, rank-local element,
  overlap, ownership, nonzeros, tracked allocation, and process peak-RSS
  summaries.

The five warm measurements characterize within-process variability; inference
uses the independent allocation-block medians, not the five samples as
independent replicates. Three-route blocks use seeded randomized cyclic order
and two-route blocks use seeded randomized alternating balanced order. Every saved block must
pass exact state, gradient/residual, four deterministic tangent-action, branch,
and covariate gates. At feasible one-rank `P1` points, full CSR patterns and
values are also compared directly.

Three additional paired `P2(L1)` designs vary only quadrature (1, 24, and 125
points) at fixed mesh, degree, rank, and analytic state. A separate synthetic
calibration runner varies element size, quadrature count, constitutive
dimension, color count, insertion density, communication volume, and imbalance
one factor at a time at 1, 8, and 32 ranks, with three independent allocation
blocks per rank. It measures raw-rank cold/warm timing and memory for every case
but does not independently vary cache or memory. The imbalance cases conserve
total work and record the realized max/mean load; nonunit imbalance is marked
inapplicable at one rank. These outputs are synthetic, non-route-faithful
mechanism diagnostics and are never presented as production route timing or
used as selector features.

The amplitudes passed the one-rank local `P2(L1)` classification gate. At
`2e-4`, all quadrature points were elastic, the minimum normalized decision
margin was `8.18e-1`, and no point lay within `1e-8` of a branch boundary. At
`2e-2`, the plastic fraction was `0.2212` (shear `0.1590`, left edge `0.0488`,
right edge `0.00949`, apex `0.00395`), the minimum normalized margin was
`3.65e-5`, and the near-boundary fraction was zero. The Karolina runner repeats
the diagnostic from its own distributed state; a changed classification blocks
timing interpretation and requires matrix/card review.

The raw calibration records are
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-ROUTE-001/state_calibration/`.
They were generated from the dirty revision worktree and are therefore pilot
gate evidence only, not publication timing data.

A subsequent one-rank fixed-state screen evaluated all three routes at both
states. Element AD and colored SFD produced bitwise-identical saved tangent
actions. Constitutive AD differed from element AD by $2.25\times10^{-16}$
relative in the elastic state and $2.34\times10^{-16}$ in the mixed state; the
maximum absolute action differences were $1.09\times10^{-11}$ and
$7.28\times10^{-12}$. This passes the local algebraic admission gate and, for
the mixed state, covers elastic, shear, left-edge, right-edge, and apex labels.
It does not replace the distributed Karolina screen, an independent residual
assembly, or the stopping-calibrated full-solve gate. The records are under
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-ROUTE-001/local_route_equivalence_v1/`.

The version-2 local `P1(L1)` design pilot also evaluated both analytic states
with all three routes. Element AD and colored SFD gave bitwise-identical saved
tangent actions; constitutive AD differed by $1.66\times10^{-16}$ relative in
the elastic state and $1.76\times10^{-16}$ in the mixed state. The elastic
state placed all 18,419 quadrature points on the elastic branch with minimum
normalized margin 0.919. The mixed state contained 14,390 elastic, 2,838
shear, 937 left-edge, 193 right-edge, and 61 apex points; its minimum margin
was $6.86\times10^{-6}$. The recorded cost inputs include element size
$m_e=12$, constitutive dimension 6, one quadrature point, 18,419 owned
elements, 10,526 free DOFs, 90 SFD colors, matrix nonzeros, rank overlap,
tracked allocations, and process peak RSS. Descriptive warm medians were about
0.0716 s for element AD/SFD and 0.0476 s for constitutive AD, but fixed route
order, one process per route, one rank, one architecture, and a dirty worktree
make every speed or crossover interpretation inadmissible. These rows are
diagnostic calibration inputs only; their dirty provenance excludes them from
the publication-model fit, which requires a clean committed rerun. Records and
their report are under
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-ROUTE-001/local_cost_model_p1l1_v2/`.

The strict analysis program
[`analyze_plasticity3d_route_cost_model.py`](../../experiments/analysis/analyze_plasticity3d_route_cost_model.py)
reconstructs each saved state and action, verifies their SHA-256 values,
requires exact state arrays and branch counts, applies the $10^{-8}$ relative
action gate for the single declared probe, and only then exposes a diagnostic
timing in the finite map. This does not establish full derivative equivalence;
publication admission additionally needs gradient/residual agreement and
multiple prespecified actions or a direct matrix comparison. Its current
local diagnostic output is
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-ROUTE-001/analysis_contract_v1/`.
It returns an empty finite map and `not_fit_insufficient_data`. After the raw-rank-timing gate was
frozen, the six legacy route files remain as 12 explicit missing map slots with six
invalid-record diagnostics because they saved only the collective maximum, not
the underlying rank values. Zero rows are publication-model eligible. This is
intentional fail-closed treatment; the dirty legacy files are not rewritten,
and the clean-workstation, Karolina, factor-diagnostic, and train/holdout gates
remain unsatisfied.

### Tier B: selected full-solve confirmation

- Problem: heterogeneous Plasticity3D, `P4(L1)`, glued bottom, strength-reduction factor 1.55.
- Rule: named 24-point tetrahedral quadrature.
- Routes: element AD and constitutive AD.
- MPI ranks: 8 and 32 on one Karolina node.
- Solver: identical rank-local assembly, fixed Hypre PMG policy, Armijo plus
  trust-region safeguards, `ksp_rtol=1e-8`, KSP cap 1000, and at most 80
  nonlinear iterations. Stopping is gradient-only in the numerically checked
  reference-elastic-energy Riesz metric. The terminal dual residual is divided
  by the initial residual of the same solve, with target $10^{-6}$ for
  `P1(L1)` and the fixed tight-reference target $10^{-7}$ for `P4(L1)`; the
  absolute residual tolerance is zero. The relative correction threshold
  $2\times10^{-3}$ is reported only as a diagnostic and cannot terminate or
  reject a solve. The separately audited Riesz solve uses CG/Jacobi with the
  unpreconditioned residual norm, relative tolerance $10^{-10}$, zero absolute
  tolerance, cap 5000, and a $10^{-8}$ independent true-residual gate.
  Both nonlinear routes build this metric through the same forced element-AD
  elastic reference operator; the nonlinear plastic tangent route remains the
  factor under comparison.
- Repetitions: ten independent cold-process paired blocks per route and rank;
  no within-process warmup is embedded. Route order is hash-seeded and exactly
  balanced five-first/five-second.

The route matrix deliberately excludes colored SFD at `P4(L1)` because its
pilot memory risk motivated a prespecified non-attempt. It must appear in the
final route map as `not attempted: memory risk; no threshold evidence`, not as
missing data or a measured out-of-memory result.

Tier B is optional executable preparation: ten independent paired blocks are
defined for `P4(L1)` at 8 and 32 ranks and for the representative low-order
`P1(L1)` case at 8 ranks. The strict
`analyze_plasticity3d_route_endpoints.py` program checks matrix policy, Riesz
evidence, clean commit and Slurm-job identity, per-rank timing provenance,
reference-elastic Riesz distance between endpoint states, physical maximum
state error, exact canonical pointwise branch-map hashes, branch counts and
margin, energy/work/observable,
initial-relative residual, iteration, and Krylov gates before exposing timing;
correction size remains diagnostic. Before either Tier-B phase may contact the
scheduler, its manifest must archive the same detached version-3 final
`EXP-STOP-001` adjudication. That adjudication must bind the admitted local
calibration, the checksum-sealed cluster archive, the clean adjudicator
commit/hash, all three MPI-consistency checks, and acceptance of the fixed
`p3d_p4_nonlinear_1em07_cluster` reference. Preparation without this artifact
is intentionally non-submittable. The analyzer independently revalidates the
artifact and its pre-submission manifest binding. It
reports deterministic paired nonparametric-bootstrap 95% intervals for route
medians and time ratios (10,000 resamples, base seed 20260710). Failed or
missing blocks remain censored. Endpoint-correct timing admission, descriptive
timing availability, and comparative ranking admission are separate statuses.
A faster-route statement requires the paired ratio interval and both
first-route execution-order strata to clear the frozen 10% practical-tie band.
The fixed Tier-B rows are ordering confirmations, not model-selected crossover
confirmations. A crossover-location claim requires a later hash-bound matrix
selected only after the training fit and untouched holdout decision are frozen,
followed by a separate human release and execution.

The tight linear tolerance is a correctness control, not a tuned production
claim. Completed local route pilots showed that `ksp_rtol=1e-2` produced
route-sensitive one-step states, whereas `ksp_rtol=1e-8` reduced relative state
disagreement to approximately `5e-11`. The matrix validator therefore rejects
every Tier-B row whose linear, relative-gradient, or Riesz parameters differ
from the hash-bound policy; the obsolete loose comparison cannot be admitted
accidentally. The P1 target is selected from admitted local calibration. The
P4 target is deliberately the tight cluster reference rather than a post hoc
loosest accepted target, and execution remains blocked until that reference is
accepted. Any future policy change requires revising all Tier-B rows together,
re-establishing route equivalence, and regenerating the reviewed matrix hash
before submission.

## Completed local route diagnostic

The dirty-worktree campaign `EXP-ROUTE-001-p3d-smoke`, run UUID
`53e2e2c3-e5a2-48e4-81dc-33813dd13985`, used two ranks, `P1(L1)`, the
one-point rule, strength-reduction factor 1.55, one Newton step, and `ksp_rtol=1e-2`.
Element AD and colored SFD had identical reported energy, gradient norm,
external work, and maximum displacement. Relative to element AD, constitutive
AD differed by `1.45e-7` in energy, `1.67e-3` in the final coefficient-gradient
norm, and `4.18e-7` in maximum displacement. No state was saved by this smoke
campaign.

The `EXP-STOP-001` follow-up retained the same problem and one-step cap,
compared element AD with constitutive AD at `ksp_rtol=1e-8`, and saved both
states. Both routes used 26 initial-guess KSP iterations and 34 Newton KSP
iterations. Their relative displacement difference was `5.28e-11`, relative
energy difference was `4.81e-14`, and relative final-gradient-norm difference
was `5.16e-11`. Both records still report maximum-iteration termination.

The collapse of the discrepancy under the tighter linear policy is consistent
with inexact-solve sensitivity, not a demonstrated derivative-route defect. It
does not prove equivalence across degrees, quadrature rules, material branches,
iterations, or architectures, and it supports no timing comparison. Detailed
records are under
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-ROUTE-001/` and
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-STOP-001/`.

## Prespecified checks

1. The state SHA-256 must be exactly equal across routes within a state/rank
   group.
2. Compare saved tangent actions using
   `||y_route-y_reference||_2 / max(||y_reference||_2, tiny)`. The required gate
   is `<=1e-8`; also report maximum absolute error. A failed route is not timed
   as equivalent.
3. Endpoint comparisons use energy, work, maximum displacement, scaled gradient
   norm, correction norm, nonlinear iterations, and total Krylov iterations.
   Timing ratios are reported only for endpoints within the final accuracy and
   state-equivalence gates.
4. Analyze cold setup separately from warm Hessian assembly and full solve.
   Report medians and percentile/bootstrap intervals without treating repeated
   timings as distinct physical problem instances.
5. Fit the frozen log-time ordinary least-squares model only after at least 48
   training rows, 20 holdout rows, six training and four holdout rows per route,
   12 training groups, eight holdout groups, and training data from both the
   workstation and Karolina pass admission. The 13 fixed features are route
   indicators, route-specific Karolina architecture shifts, a declared
   route-work proxy above, owned nonzeros, maximum rank overlap, rank count,
   plastic fraction, and owned-element/overlap imbalance. The work proxy, owned
   nonzeros, and maximum overlap use `log1p`; rank count uses `log`; plastic
   fraction and both imbalance ratios remain untransformed. Peak/tracked memory is
   reported but excluded to avoid post-run predictor leakage. No feature or
   split may be changed after seeing holdout timings.
6. Admit a predictive selector only if holdout median absolute percentage error
   is at most 25%, the 90th percentile is at most 50%, resolved ordering accuracy
   is at least 90%, at least four holdout groups are resolved beyond the 10%
   practical-tie band, and at least two distinct observed route winners occur.
   The separate synthetic factor diagnostic fits warm stage log-times on ranks
   1 and 8 and validates without refitting at rank 32, with MAPE gates of 50%
   median and 100% at the 90th percentile. It is non-route-faithful and never
   enters the production feature vector. Otherwise publish the finite empirical
   map without a selector claim.
7. Require the paired-block, collective-max, multiple-derivative, and hash-bound
   Tier-B endpoint gates to pass on clean records. Retain the descriptive
   factor diagnostic and its failures, but do not use it as a selector gate;
   implementation alone never releases a numerical claim.
8. Verify the route-specific label in every saved state and run record. **The
   writer defect is fixed and regression-tested:** a verification rerun stores
   `local_constitutiveAD` in the constitutive NPZ. Historical malformed states
   remain invalid and are not rewritten.
9. Serialize raw solver histories as schema-valid JSON. **Implemented and
   verified:** raw optional sentinels are now `null` under `allow_nan=False`,
   while validated publication records continue to reject nonfinite evidence.

## Resource and failure policy

The clean-workstation driver launches exactly one route process at a time. An
execution is admissible only when the driver's POSIX address-space soft limit
is finite, positive, and no larger than 64 GiB. The driver freezes that limit,
checks it immediately before every direct child process, records the inherited
limit in each process receipt, and verifies it again at campaign termination.
Preparation without execution may record an unlimited address space, but such
a record does not authorize execution. A changed limit, interruption, timeout,
or worker failure leaves a diagnostic archive. The publication workflow never
appends to or resumes that archive; a retry uses a fresh output root so that the
admissible inventory remains exactly 36 launches in 12 independent blocks.

The required Karolina preparation is 99.95 node-hours, including 64.20
node-hours for 78 paired baseline blocks, 18 factorized-quadrature blocks, and
nine component-diagnostic rows, plus the separately retained discretization
and Hyperelasticity campaigns. The optional paired high/low-order confirmation
tranche is 45.00 node-hours; it is selected separately and never combined with
all optional campaigns under the 100-node-hour guard. Every job is isolated below
`artifacts/reproduction/paper_revision_karolina/<campaign-id>/cases/<case-id>/`.
Timeout, out-of-memory, compilation, nonfinite, and solver-cap outcomes remain
explicit censored rows with their last available counters.

No result is publication evidence unless it comes from the reviewed matrix
hash, a clean commit, a revalidated allocation, and a complete environment and
Slurm record.

## Terminal decisions

- **PASS:** all retained routes are derivative-equivalent and the held-out model
  predicts ordering within its declared band.
- **EMPIRICAL MAP:** equivalence passes but prediction fails; restrict claims to
  tested configurations and architectures.
- **CENSORED ROUTE:** a route fails correctness, memory, or wall-time gates; keep
  it visible and do not impute a timing.
- **INVALID:** states differ, accuracy differs, branch classification is
  unresolved, or provenance is incomplete; rerun after correcting the cause.
