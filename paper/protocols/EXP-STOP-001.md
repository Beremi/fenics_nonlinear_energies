# EXP-STOP-001: Stopping and Inexact-Solve Calibration

## Status and evidence decision

**Narrow route-sensitivity diagnostic complete; full calibration not run.** A
two-route `P1(L1)` pilot shows that tightening the KSP relative tolerance from
the historical `1e-2` to `1e-8` collapses a one-step element-AD versus
constitutive-AD state discrepancy. Both tight rows were deliberately capped
after one Newton step and were produced from a dirty worktree. They are
diagnostic pilot evidence only, not publication evidence, convergence evidence,
or timing evidence.

The shared Riesz-metric interface, a scalar P1 lumped-$L^2$ map, a
Plasticity3D glued-free-space reference-elastic map, and a HyperElasticity
two-end-constrained reference-elastic map are implemented and unit-tested.
The Plasticity3D and HyperElasticity maps both pass two-rank P1 smokes with an
numerical SPD inertia check and independently recomputed norm-solve residual. The
HyperElasticity smoke also passes on one rank and verifies the same metric
provenance in every load-step and terminal record. These are implementation
checks from the current worktree, not clean publication evidence. Cross-mesh
and tolerance calibration are still required. Until this card reaches a
terminal decision, all affected full-solve route, discretization, scaling, and
timing rows are blocked from publication admission.

## Executable staged local campaign

`experiments/runners/run_exp_stop_001_local_calibration.py` now freezes and
executes the locally feasible portion of this card. Its default plan contains
40 required-local rows:

- deterministic lumped-$L^2$ Ginzburg--Landau endpoints on levels 5 and 6 at
  relative dual-residual targets `1e-2`, `1e-4`, `1e-6`, and `1e-8`;
- HyperElasticity reference-Riesz setup/terminal-residual checks on levels 1
  and 2 at norm-solve tolerances `1e-8`, `1e-10`, and `1e-12`;
- one-load-step nonlinear HyperElasticity endpoints on levels 1 and 2 at the
  four residual targets above;
- Plasticity3D P1/P2 fixed-state tangent solves at requested KSP tolerances
  `1e-2`, `1e-4`, `1e-6`, `1e-8`, and `1e-10`, with an independently rebuilt
  true residual; and
- full nonlinear reference-Riesz Plasticity3D P1/P2 endpoints at the four
  residual targets above.

The default plan also retains 12 explicit censors: five P4 fixed-state rows
whose local feasibility has not been attested, four nonlinear P4 rows, and one
publication-rank MPI-consistency row for each problem family. P4 fixed-state
rows become required-local only when preparation uses both `--p4-policy local`
and `--confirm-p4-local-feasible`. Nonlinear P4 and MPI rows remain cluster
computations. The existing HyperElasticity state export does not retain the
reference-operator action, so its same-mesh coefficient displacement
difference is reported only as a diagnostic and is not relabeled as a Riesz
state difference.

The publication sequence is deliberately non-overwriting:

```bash
./.venv/bin/python experiments/runners/run_exp_stop_001_local_calibration.py \
  prepare \
  --run-kind publication \
  --output-root artifacts/reproduction/exp_stop_001_local_<commit>

./.venv/bin/python experiments/runners/run_exp_stop_001_local_calibration.py \
  execute \
  --plan artifacts/reproduction/exp_stop_001_local_<commit>/plan.json \
  --all-local \
  --confirm-all-local-execution

./.venv/bin/python experiments/runners/run_exp_stop_001_local_calibration.py \
  analyze \
  --plan artifacts/reproduction/exp_stop_001_local_<commit>/plan.json
```

Publication preparation and execution require the frozen commit and a clean
worktree. Diagnostic preparation from a dirty tree must say both
`--run-kind diagnostic` and `--allow-dirty`; such output is never publication
evidence. A missing row leaves the analysis incomplete. A failed local command
is retained as an unclassified runtime censor, and failure of the tightest
same-discretization reference invalidates that comparison group.

## Scientific questions

1. How much of the route-to-route endpoint difference is caused by an inexact
   linear solve rather than by a derivative inconsistency?
2. What KSP forcing policy makes algebraic error negligible relative to the
   intended observable precision and estimated discretization error?
3. What mesh-aware Riesz-scaled residual and correction thresholds give the
   same scientific accuracy across polynomial degree, mesh, and MPI count?

## Completed diagnostic matrix

All rows used the heterogeneous Plasticity3D problem, glued-bottom constraint,
`P1(L1)`, strength-reduction factor 1.55, the named one-point tetrahedral rule, two MPI ranks,
the same single-level Hypre preconditioner, Armijo line search, and a one-Newton-
step cap.

| Diagnostic | Route | KSP relative tolerance | KSP cap | Newton KSP iterations | Terminal status | Saved state |
| --- | --- | ---: | ---: | ---: | --- | --- |
| loose route smoke | element AD | `1e-2` | 100 | 10 | maximum-iteration cap | no |
| loose route smoke | colored SFD | `1e-2` | 100 | 10 | maximum-iteration cap | no |
| loose route smoke | constitutive AD | `1e-2` | 100 | 10 | maximum-iteration cap | no |
| tight follow-up | element AD | `1e-8` | 500 | 34 | maximum-iteration cap | yes |
| tight follow-up | constitutive AD | `1e-8` | 500 | 34 | maximum-iteration cap | yes |

At the loose tolerance, element AD and colored SFD produced identical reported
one-step scalars. Constitutive AD differed from element AD by `0.4286076` in
energy (`1.45e-7` relative), `22.9152` in final coefficient-gradient norm
(`1.67e-3` relative), and `3.1403e-7` in maximum displacement (`4.18e-7`
relative).

At the tight tolerance, the saved element-AD and constitutive-AD displacement
arrays differed by `1.3435e-9` in Euclidean norm, or `5.2837e-11` relative to
the element-AD displacement norm, with maximum coefficient difference
`6.8496e-11`. The relative differences were `4.8098e-14` for energy,
`5.1644e-11` for final coefficient-gradient norm, `7.1562e-13` for external
work, and `7.8258e-12` for maximum displacement. Each route used 26 KSP
iterations for the initial guess and 34 for the Newton equation.

This reduction is consistent with KSP-tolerance sensitivity. It does not prove
route equivalence because only one degree, rule, load, MPI count, iteration,
and material trajectory were checked, and no independent fixed-state global
matrix or tangent-action comparison was recorded here.

## Scalar Lumped-$L^2$ Pilot

The shared scalar driver now implements a positive row-sum P1 mass map on the
exact reordered free space, uses the unit-field lumped-$L^2$ norm as its
default state scale, and stores canonical global node-ordered final states. A
two-rank level-5 dirty-worktree pilot compares baseline and tenfold tighter
KSP, residual, energy-change, and correction policies.

The retained reduced-subspace trust-region Ginzburg--Landau pair starts from
the deterministic field stated in the manuscript. Its state difference is
$3.81\times10^{-7}$, or $2.22\times10^{-7}$ relative, and its energy difference
is $1.20\times10^{-13}$. This is a local endpoint-sensitivity observation, not
timing or robustness evidence. The attempted p-Laplace pair is invalid as a
controlled sensitivity comparison: each cold process called an unseeded random
initializer, and neither initial vector was stored. Its numerical comparison
has therefore been removed from the manuscript. Cross-mesh calibration and
clean reruns remain open. Full diagnostic records are under
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-STOP-001/scalar_lumped_l2_v1/`.

## HyperElasticity Reference-Map Pilot

The JAX/PETSc HyperElasticity path now copies the exact P1 tangent at the
bitwise-verified undeformed map $y(X)=X$ and certifies it after both end-face
constraints are eliminated. A level-1 diagnostic with 2,133 free DOFs was run
on one and two MPI ranks with zero Newton iterations, so it tests setup and the
terminal residual evaluation rather than nonlinear convergence. Both rows
reported inertia $(0,0,2133)$, the identical state scale
1031.6632492383628, energy 31.589493767862614, and coefficient-gradient norm
4643.865597353255. The Riesz dual residuals differed by
$7.43\times10^{-12}$ relatively.

The CG/Jacobi inverse map took 1,283 iterations on one rank and 1,252 on two
ranks. Its independently recomputed relative residuals were
$1.20\times10^{-10}$ and $9.35\times10^{-11}$, respectively, below the
prespecified $10^{-8}$ gate. The implementation therefore passes this narrow
contract smoke, but Jacobi is not yet an accepted publication preconditioner.
The HyperElasticity-specific safety cap is 5,000; cross-mesh tolerance and
preconditioner calibration remain open. Records and the dirty-evidence warning
are under
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-STOP-001/hyperelasticity_reference_riesz_v1/`.

## Required implementation and provenance repairs

Complete these tasks before the calibration matrix is run:

1. **Implemented in the maintained DOF solver:** build and document the
   Plasticity3D Riesz map. Record the primal state norm,
   dual residual norm, relative residual, and correction norm with their units
   and normalization. The map uses the elastic tangent at zero displacement,
   is restricted to glued-bottom free DOFs, and requires symmetry plus
   Cholesky/inertia certification. Keep coefficient Euclidean norms as
   diagnostics, not stopping criteria. Integration into the derivative-route
   runner remains open.
2. **Implemented in the JAX/PETSc HyperElasticity DOF and trust-runner paths:**
   assemble and copy the exact discrete tangent at $y(X)=X$ before the first
   load-step boundary update. Verify exact reference-state equality, the
   two-end constrained free-space dimension, symmetry, and direct-factor
   inertia. Use the fixed map for every load step, record the fact that the
   nonlinear state is $y$ rather than $u$, and use the initial
   $\lVert y\rVert_{K_{\rm ref}}$ as the default dimensionful state scale.
   The FEniCS runner remains outside this contract. Clean cross-mesh and
   tolerance calibration remain open.
3. **Implemented for the maintained DOF and norm-solve paths:** add a
   route-independent true-residual recomputation after every linear solve
   used for calibration. Record requested KSP tolerance, convergence reason,
   recursive residual, independently recomputed residual, right-hand-side norm,
   and achieved relative residual. The backend-mix route runner must expose the
   same terminal contract before its rows are admitted.
4. **Pilot implementation verified; clean rerun remains:** save the exact
   command, environment, commit, worktree cleanliness, mesh
   checksum, state checksum, and route in a versioned run record. The raw-state
   route metadata is now fixed and regression-tested, and a dirty verification
   rerun records the exact commands. The clean versioned terminal run record
   and complete environment capture remain required.
5. **Implemented and verified:** raw histories map inapplicable nonfinite
   sentinels to standards-compliant JSON `null` under `allow_nan=False`; the
   clean publication schema retains strict rejection of nonfinite evidence.
6. **Implemented in the verification rerun:** extend the passing NPZ route/rule
   metadata regression to read the terminal run record and verify mesh, degree,
   load, ranks, state hash, and output hash as one cross-artifact identity
   contract. Repeat it from the clean publication commit.

## Publication calibration matrix

Execute in the listed order. Later tiers are admission-blocked until the prior
tier passes.

1. **Fixed-state algebraic gate.** For `P1(L1)`, `P2(L1)`, and `P4(L1)`, save
   one common initial state and at least one branch-stable intermediate state.
   Assemble element-AD and constitutive-AD residuals and tangent actions at the
   identical state and partition. Include colored SFD where memory permits.
   Require relative tangent-action error at most `1e-8`, report absolute error,
   and retain branch labels and boundary margins. Do not time a route that
   fails this gate.
2. **Linear-tolerance sweep.** At every admitted state, run the identical
   preconditioner and Krylov method with requested relative tolerances
   `1e-2`, `1e-4`, `1e-6`, and `1e-8`, plus a tighter reference if attainable.
   Record the achieved true residual and state correction for every row.
3. **Nonlinear stopping sweep.** On a small smooth case, HyperElasticity on at
   least two mesh levels, and Plasticity3D `P1(L1)`, `P2(L1)`, and `P4(L1)`,
   cross the accepted linear policy with
   Riesz-scaled nonlinear residual targets from `1e-2` through `1e-6` and a
   correction criterion. Save every successful endpoint. A cap, divergence,
   nonfinite value, or branch-changing endpoint remains an explicit censored
   row.
4. **Observable adjudication.** Compare energy, work, maximum displacement,
   branch fractions, scaled state, nonlinear work, and Krylov work with the
   tightest successful same-discretization reference. A candidate policy is
   provisionally acceptable only when each primary observable changes by less
   than one quarter of its intended final rounding unit and the Riesz state
   difference passes the prespecified same-discretization bound.
5. **Discretization check.** After `EXP-DISC-001`, require the accepted
   algebraic state and observable errors to be materially smaller than the
   estimated discretization error. If they are not, tighten the policy and
   rerun every dependent endpoint and timing row.
6. **MPI consistency check.** Repeat selected accepted rows at the publication
   rank counts. Use the same accuracy gates; do not infer accuracy from KSP
   iteration count or convergence reason alone.

## Timing exclusion

No timing ratio is admissible from the completed pilots. They contain one
observation per route, JIT/setup effects, different measured callback costs,
and no converged endpoint. Timing begins only after the fixed-state derivative
gate and this stopping contract pass, then uses warmups, independent process
repetitions, complete counter accounting, and accuracy-equivalent endpoints.

## Terminal decisions

- **PASS:** one common KSP/nonlinear policy meets the true-residual, Riesz-state,
  observable, discretization, and MPI gates for every retained publication row.
- **SCOPED PASS:** different degrees or problem classes require explicitly
  different calibrated policies; report each policy and prohibit cross-policy
  timing comparisons.
- **CENSORED:** the tight reference is infeasible for a declared case; retain
  the failure and remove any accuracy or route-timing claim for that case.
- **INVALID:** route, state, mesh, rule, solver policy, or provenance differs
  within a comparison group; repair and rerun.
