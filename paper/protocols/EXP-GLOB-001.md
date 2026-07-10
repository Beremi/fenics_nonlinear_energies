# EXP-GLOB-001: Controlled Globalization Evidence

## Research Questions

1. With the discrete functional, Hessian action, preconditioner, forcing rule,
   starting state, and final accuracy fixed, how do Armijo Newton, Steihaug
   trust-region Newton, and hybrid trust-region plus line search differ?
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
  contract fixed. The reduced trust path also uses Armijo. This tier currently
  admits p-Laplace, Ginzburg--Landau, and hyperelasticity. Plasticity3D is
  excluded until a branch-stable nonlinear case can use the same controlled
  solve contract.
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

The publication campaign remains pending a clean worktree, frozen Riesz
stopping, canonical state output, independent residual evaluation, endpoint
clustering, distinct robustness instances, and repeated timing after all
correctness gates pass.
