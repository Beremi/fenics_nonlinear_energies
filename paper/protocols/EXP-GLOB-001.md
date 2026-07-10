# EXP-GLOB-001: Controlled Globalization Evidence

## Research Questions

1. With the discrete functional, Hessian action, preconditioner, forcing rule,
   starting state, and final accuracy fixed, how do Armijo Newton and the
   reduced-subspace trust-region method with the same Armijo safeguard differ?
2. How do complete production bundles differ when the linear solver or other
   policy components are intentionally allowed to change?

Only the first tier can support a claim about globalization itself.

## Algorithm Freeze

Before the publication campaign, give each method a separate reproducible
algorithm specifying merit/model, step equation, KSP forcing, negative
curvature, descent failure, accept/reject inequalities, trust-radius updates,
line-search contraction, NaN/Inf handling, retry limits, and every terminal
condition. Add boundary tests for equality at thresholds, negative curvature,
failed linear solves, repeated line-search rejection, and roundoff acceptance.

The trust-region norm and stopping Riesz metric are different objects unless an
algorithm explicitly makes them the same. Record both.

## Case Matrix And Analysis

Use a smooth convex scalar case, nonconvex Ginzburg--Landau, hyperelasticity,
and only branch-stable synthetic plasticity states. Use at least five machine-
noise repetitions after correctness. Robustness units must be distinct loads
or starting states, not repeated timing of one instance.

Cluster nonconvex endpoints by weighted state, energy, and problem observables.
Compare time only inside one endpoint class. Report success, cap, timeout,
function/gradient/HVP/preconditioner counts, accepted/rejected steps, line-search
evaluations, negative-curvature events, Krylov work, Riesz-scaled residual, and
complete timing histories.

The maintained runner exposes two non-interchangeable tiers:

- `--comparison-tier controlled` compares Armijo Newton with the
  reduced-subspace trust-region method while keeping the ordinary Hessian
  solve, KSP type, preconditioner, tolerances, initial state, and stopping
  contract fixed. The reduced trust path also uses Armijo. The retained local
  smoke contains Ginzburg--Landau and the first hyperelastic load only.
  Hyperelastic continuation is restricted to one load so a later method-
  specific warm start cannot enter the comparison. Plasticity3D is excluded
  until a branch-stable nonlinear case can use the same controlled solve
  contract. The p-Laplace row is also excluded: a code-local random generator,
  even when seeded in the current implementation, is not treated as a
  prescribed scientific starting-state artifact, and no new random comparison
  is introduced here.
- `--comparison-tier production_bundle` retains Newton/ordinary-KSP,
  Steihaug/STCG, and hybrid/STCG bundles. It answers only the second research
  question.

Example preparation-only commands are:

```bash
./.venv/bin/python experiments/runners/run_globalization_method_compare.py \
  --mode smoke --comparison-tier controlled --dry-run
./.venv/bin/python experiments/runners/run_globalization_method_compare.py \
  --mode full --comparison-tier controlled --dry-run
```

Publication execution is deliberately local and controlled-only. It refuses
an existing campaign path, a dirty worktree, a non-40-character Git commit, a
change of commit during execution, and every case above four MPI ranks. Thus,
`--mode full` is preparation-only in this runner. A complete local smoke is:

```bash
./.venv/bin/python experiments/runners/run_globalization_method_compare.py \
  --mode smoke --comparison-tier controlled \
  --raw-root artifacts/reproduction/EXP-GLOB-001-local-clean-v1/raw \
  --report-root artifacts/reproduction/EXP-GLOB-001-local-clean-v1/reports
```

The default is five cold-process machine-noise repetitions. With two problems,
three deterministic starts, two methods, and five repetitions, this command
launches exactly 60 two-rank solves. For a correctness-only pilot plan, use
`--instance nominal --timing-repetitions 1 --dry-run`; timing from that reduced
grid is explicitly inadmissible.

Before any output directory is created, the runner admits one clean Git commit.
It then records a strict-JSON, versioned campaign manifest and one validated
publication run record per launch. These records include normalized argv,
UTC start/finish times, source/configuration/input SHA-256 hashes, the exact
single-thread CPU child environment, termination/censoring, solver counts,
terminal residuals, and artifact identities.

## Reviewed full-rank Karolina path

The local runner does not execute the full-rank matrix. The separate
scheduler-free preparer
`experiments/runners/prepare_exp_glob_001_karolina.py` freezes the same
controlled algorithms at the intended ranks: Ginzburg--Landau level 10 at 16
ranks and the first HyperElasticity level-4 load at 32 ranks. For each problem
it creates the nominal and two signed deterministic starts once, then binds the
same immutable NPZ to both methods and all five cold-process repetitions.

The resulting matrix has exactly 60 one-node jobs: 30 Ginzburg--Landau
launches with a 10-minute allocation ceiling and 30 HyperElasticity launches
with a 15-minute ceiling. Its total ceiling is 12.5 Karolina CPU node-hours.
Every row requires one thread per rank and the frozen CPU/JAX/XLA environment,
and retains `output.json`, `final_state.npz`, stdout, stderr, job metadata,
environment identity, and a validated publication run record. Preparation
writes only a plan, source/input hashes, common-start artifacts, and exact
shell-quoted command text:

```bash
./.venv/bin/python experiments/runners/prepare_exp_glob_001_karolina.py \
  prepare --output-root artifacts/reproduction/EXP-GLOB-001-karolina-<commit>

./.venv/bin/python experiments/runners/prepare_exp_glob_001_karolina.py \
  preflight --campaign-root artifacts/reproduction/EXP-GLOB-001-karolina-<commit>
```

Neither command contacts Slurm. Without reviewed environment setup and lock
files, the manifest is explicitly ineligible for submission. In a future
authorized execution, the pre-seal analyzer must reconstruct all 60 run
records and pass the common-start, terminal-identity, exact repetition, and
bounded-instance audit. Offline accounting settlement then reparses each raw
record against its 16- or 32-rank allocation and checksum-seals the archive.
Only a detached post-copy adjudication using that checksum digest can expose a
timing or tested-instance comparison decision. Population-level robustness
generalization remains inadmissible by construction. No full-rank Karolina
launch has been submitted or run.

## Deterministic Robustness Instances And Repetitions

Each retained problem has three prescribed instances: the nominal state and a
positive/negative closed-form perturbation. For Ginzburg--Landau, a
dimensionless amplitude of $\pm 0.025$ multiplies a smooth mode that vanishes on
the boundary and is added only at free nodes. For first-load hyperelasticity, a
$\pm 10^{-5}\,\mathrm{m}$ transverse beam mode is added only to free vector
components. These perturbations preserve constrained degrees of freedom by
construction and are small relative to the maintained geometry.

Exactly one NPZ is created for each problem--instance pair. Both methods and
all timing repetitions read that same immutable file; repetitions never create
new robustness units. The three starts form a bounded deterministic sensitivity
set, not draws from a declared population. Consequently the audit can admit a
comparison on the tested instances, but it always sets
`robustness_generalization_claim_admissible` to false.

## Common-Start And Endpoint Identity Contract

Before launching either controlled method, the orchestrator writes exactly one
canonical NPZ start for each retained benchmark and deterministic instance
under `_canonical_starts/`.
For Ginzburg--Landau this is the closed-form sine state on the constrained
scalar mesh. For hyperelasticity it is the reference deformation
(y(X)=X). The start manifest records both the NPZ file SHA-256 and a dtype-
and-shape-aware state-content SHA-256.

Every controlled command must contain both `--state-in` and `--state-out`.
The solver validates the stored mesh level, coordinates, connectivity,
problem identity where applicable, state dimension, and finiteness before it
converts the state to the solver's distributed ordering. Both method rows must
then report the same input file and content hashes. A missing or inconsistent
hash fails the controlled identity audit.

After each terminal solver return, the gradient is evaluated again outside the
nonlinear stopping decision. The result records:

- the final-state NPZ file and content hashes;
- the distributed endpoint-state hash;
- the independently evaluated dual residual and coefficient-space residual;
- the distributed residual-vector hash; and
- the Riesz-evaluation diagnostics, including the true residual of an
  inverse-Riesz solve when that metric is selected.

Endpoint hashes determine whether two completed rows occupy the same canonical
endpoint class. Different endpoint hashes do not get silently averaged or
timed as one comparison. The identity audit remains non-performance evidence;
repetitions and distinct robustness instances are still required before a
timing or general robustness claim.

## Pilot Interpretation Rule

A method at the iteration cap is `fixed_work`, regardless of finite energy. A
line search that repeatedly rejects an identical state is a failed/capped
algorithmic path, not slow convergence. Methods using different KSP types are
production bundles and cannot isolate globalization.

## Current Pilot Status (2026-07-10)

The controlled two-rank smoke matrix has been implemented and run for
p-Laplace L5, Ginzburg--Landau L5, and two hyperelastic L2 load steps. The
strict-JSON rerun is recorded under
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-GLOB-001/controlled_v2/`.
It is dirty-worktree pilot evidence only.

The controlled GL observation is a first-iteration Armijo failure versus a
12-iteration reduced-trust convergence under the same GMRES/Hypre contract.
The p-Laplace pair is excluded because each cold process used an unseeded
random initializer and neither initial vector was stored. Hyperelasticity
executes through both loads, but only the first load has a common identity
start; the second warm-starts from each method's first endpoint. Its
$8.15\times10^{-5}$ final-energy difference, missing canonical states, and
loose smoke tolerances fail the endpoint-equivalence gate. No timing claim is
admitted from any row.

The common-start, final-state, independent-residual, deterministic-instance,
repetition, clean-commit, and provenance interfaces are prepared. The corrected
60-launch controlled smoke has not yet been executed. No timing or robustness-
generalization claim is admitted from the historical pilots.
