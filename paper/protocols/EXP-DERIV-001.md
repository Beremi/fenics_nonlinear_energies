# EXP-DERIV-001: Derivative Correctness

## Research Question

For each retained discrete functional and stored state, do element AD,
constitutive AD, assembled sparse matrices, and matrix-free Hessian actions
represent the same constrained discrete first and second derivatives?

## Hypotheses

1. On a smooth or fixed-branch state, independently assembled constitutive
   residuals and tangents agree with element-energy AD to FP64 accuracy.
2. Centered finite differences of the scalar energy and residual agree with
   the reported gradient and Hessian action over a non-roundoff step window.
3. Serial and distributed routes produce equivalent canonical free-space
   objects within a reduction-order tolerance fixed before the distributed
   run.

No hypothesis is made about differentiability at a plastic branch interface.
Those states belong to EXP-MC-001 and cannot be certified by a fixed-branch
test.

## Primary Metrics And Gates

- relative element-AD versus constitutive residual error: at most `1e-8`;
- relative element-AD versus constitutive Hessian error: at most `1e-8`;
- Hessian symmetry defect: at most `1e-10` for scalar smooth branches;
- centered finite-difference directional-gradient and HVP errors: at most
  `1e-7` at the prespecified gate step, with the complete step sweep retained;
- energy Taylor remainder: observed second-order regime over at least three
  pre-roundoff steps;
- gradient/Hessian Taylor remainder: expected fixed-branch order over at least
  three pre-roundoff steps;
- branch label unchanged at both centered finite-difference states for every
  fixed-branch plasticity check;
- assembled matrix versus HVP, constrained/free-map, and serial/distributed
  errors: at most `1e-8` after scale normalization, unless a conditioning audit
  freezes a different threshold before execution.

The branch diagnostic replays the production predicates and is not an
independent material reference. It records normalized active-predicate margin,
distance to any switch surface, principal-value gap, denominator margin, and
tie-break scale so an apparently smooth finite-difference result cannot hide a
known switch.

## Case Matrix

| Block | Cases | States | Distribution |
| --- | --- | --- | --- |
| scalar smooth | p-Laplace small/medium; Ginzburg--Landau regular and indefinite | initial, stored intermediate, final, deterministic perturbation | 1 and 2 ranks |
| mechanics smooth | hyperelasticity small/medium | admissible initial, stored intermediate, final | 1 and 2 ranks |
| branch-structured diagnostic | Plasticity3D `P1(L1)`, `P2(L1)`, `P4(L1)` | fixed branch interiors first; interface states handled by EXP-MC-001 | element pilot locally; assembled 1 and 2 ranks later |

All cases use FP64, identical quadrature and constrained spaces across routes,
canonical state ordering, and deterministic directions. A stored state and its
input hash are part of the case identity.

## Current Fixed-Element Command

For each degree `d` in `1,2,4`:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
./.venv/bin/python experiments/runners/run_paper_derivative_verification.py \
  --degree d \
  --states 5 \
  --fd-step-sizes 1e-6,3e-7,1e-7,3e-8,1e-8 \
  --fd-gate-index 2 \
  --route-tolerance 1e-9 \
  --symmetry-tolerance 1e-10 \
  --fd-tolerance 1e-7 \
  --assembled-route-equivalence \
  --output OUTPUT.json
```

The assembled-route flag adds a deterministic serial full-mesh comparison of
the element-AD, exact local-SFD/JVP, and constitutive-AD matrices without
calling a linear or nonlinear solver. It does not replace the prescribed
distributed matrix/action checks. The current command is a local pilot because
the implementation and worktree are not yet a clean immutable publication
commit.

## Inputs

- runner: `experiments/runners/run_paper_derivative_verification.py`;
- Plasticity3D energy: `src/problems/slope_stability_3d/jax/jax_energy_3d.py`;
- constrained same-mesh HDF5 cases under
  `data/meshes/SlopeStability3D/hetero_ssr/`;
- deterministic base seed `1729`, incremented once per state;
- default local element index `0` and strength-reduction value `1.5`.

## Required Outputs

- one versioned terminal record per case and repetition;
- raw Taylor and centered-finite-difference arrays;
- state, direction, canonical ordering, constrained-DOF map, and hashes;
- route gradients, Hessians/HVPs, branch diagnostics, and symmetry metrics;
- distributed ownership/partition metadata and rank-aggregated errors;
- a report retaining every failed, capped, or nonfinite case.

## Success And Failure Rules

The block passes only when every prespecified case passes every applicable
primary gate. A branch change at the fixed-branch finite-difference gate, a
nonfinite derivative, an unexplained loss of Taylor order, a missing canonical
state, or a route error above threshold fails the case and blocks timing of
that route. A roundoff-limited Taylor tail is retained but is not fitted.

Passing the current Plasticity3D element pilot establishes only local
fixed-branch formula consistency. It does not establish assembled correctness,
distributed recovery, interface regularity, solver convergence, or physical
validation.
