# Venue And Contribution Decision

Last updated: 2026-07-11.

## Decision

The primary manuscript route is a scientific-computing paper centered on
derivative placement in distributed nonlinear finite-element Newton methods.
The natural primary venue is the *SIAM Journal on Scientific Computing* or a
closely aligned scientific-computing journal. The revision will retain the
mathematical and evidentiary standards identified by the SIOPT-style review,
but it does not claim a new optimization algorithm.

Route performance, crossover modeling, scaling, topology optimization, and
KKT claims are removed from the selected submission scope. The
branch-structured Mohr--Coulomb endpoint surrogate is retained only as a
derivative and branch-programming benchmark, not as path-consistent plasticity
or physical slope-stability validation.

## Central Contribution

The contribution is to identify sufficient conditions under which
element-level automatic differentiation, quadrature-point constitutive
differentiation, and colored sparse Hessian recovery represent the same
constrained quadrature functional and distributed operator. The paper combines
a branchwise chain-rule result, an owned-row colored-recovery proposition, and
independent finite verification across smooth, finite-strain, and
branch-structured problems. It separates complete local Hessian or sparse-matrix
comparisons from finite tangent-action checks and makes no performance
selection claim.

## Research Questions

1. **Correctness.** Under which fixed-functional, quadrature, branch, constraint,
   and ownership assumptions do the three derivative routes produce equivalent
   residuals and Hessian actions?
2. **Distributed realization.** Under which structural-pattern, coloring,
   overlap-state, and row-ownership conditions do local Hessian actions recover
   the canonical distributed sparse operator?
3. **Claim boundary.** Which analytic, manufactured, full-matrix,
   tangent-action, stopping, and endpoint checks are needed to distinguish
   finite correctness evidence from unsupported nonlinear or performance
   conclusions?

## Main-Text Evidence

- A smooth scalar reference problem for Taylor, residual, Hessian-action, and
  sparse-recovery verification.
- Hyperelasticity for matched constitutive/element differentiation and
  distributed assembly evidence.
- A branch-structured Mohr--Coulomb endpoint surrogate as the high-order and
  nonsmooth test, with ordinary Hessian claims restricted to strict branch
  interiors.
- Canonical one-/two-/four-rank assembly checks, distributed colored-recovery
  tangent actions, fixed-state quadrature sensitivity, deterministic local
  stopping calibration, and bounded globalization outcomes.

## Deferred Future Evidence Outside This Submission

- Prepared route-timing, crossover, scaling, and cluster stopping protocols,
  which are future work and not dependencies of this manuscript.
- Plasticity2D, topology optimization, secondary preconditioner studies, and
  noncentral state figures, all retained only as possible future work.
- Complete machine-readable run inventories and state archives, which belong in
  the public reproducibility deposit rather than the main text.

## Explicitly Excluded Claims

- A new automatic-differentiation rule, Krylov method, or universal nonlinear
  solver hierarchy.
- Universal superiority of JAX+PETSc or constitutive AD.
- A fastest derivative route, crossover predictor, timing comparison, or
  scaling law.
- Path-consistent incremental Mohr--Coulomb plasticity without an incremental
  history campaign.
- A topology-optimization solution without a fixed problem, feasibility, and
  KKT evidence.
- Continuum convergence from a study that changes degree, quadrature, mesh, and
  algebraic accuracy simultaneously.
- Strong-scaling claims from a series that changes solver or coarse-grid policy.

## Closest-Work And Novelty Audit

The following primary sources were refreshed on 2026-07-10.

| Source | Verified scope | Consequence for this paper |
| --- | --- | --- |
| [JetSCI, arXiv:2604.22087](https://arxiv.org/abs/2604.22087) | A hybrid JAX--PETSc framework for scalable differentiable simulation, including finite-element micromechanics and distributed-memory execution. | JAX--PETSc integration is not the novelty. The paper must distinguish its conditional derivative-placement results and verified PETSc-owned assembly contract. |
| [FE-MAD, arXiv:2606.05199](https://arxiv.org/abs/2606.05199) | JAX-FEM material learning with automatically differentiated Newton tangents and inverse-problem loss gradients. | Constitutive differentiation and differentiable finite elements are established. The contribution must be route equivalence and distributed verification, not the availability of AD tangents. |
| [Differentiable finite-strain plasticity, arXiv:2606.17390](https://arxiv.org/abs/2606.17390) | GPU-accelerated JAX finite elements with constitutive differentiation, sparse assembly, nonlinear solution, and inverse characterization. | Plasticity plus JAX is not novel by itself. The manuscript must keep its endpoint-surrogate status explicit and justify the branch verification independently. |
| [Locality-Aware Automatic Differentiation, arXiv:2509.00406](https://arxiv.org/abs/2509.00406) | Per-element forward-mode differentiation and sparse Hessian assembly for GPU mesh computations, with explicit locality and memory-traffic arguments. | Element-local differentiation and sparse Hessian assembly are not new. The paper must isolate its distributed PETSc ownership contract, constitutive route, and common-functional checks. |
| [tatva, arXiv:2602.12365](https://arxiv.org/abs/2602.12365) | A globally applied AD finite-element framework using matrix-free products and coloring-based sparse tangent materialization, with GPU demonstrations at million-DOF scale. | The local-versus-global AD tradeoff and scalable colored materialization are already explicit in recent work. Novelty cannot rest on applying AD globally or recovering a sparse tangent; it must rest on the conditional three-route equivalence and distributed recovery conditions. |
| [FEniCSx external operators, DOI 10.46298/jtcam.14449](https://doi.org/10.46298/jtcam.14449) | Quadrature-point external constitutive operators differentiated with JAX, including Mohr--Coulomb, Taylor tests, and documented software. | Constitutive-point AD for complex plasticity is established. The branch surrogate and route comparison must be positioned as a fixed-functional derivative-placement study, not a new constitutive-AD capability. |
| [Sparser, Better, Faster, Stronger, TMLR 2025](https://openreview.net/forum?id=GtXSN52nIW) | Automatic local/global sparsity detection and efficient sparse Jacobian/Hessian differentiation with coloring. | Sparse AD and coloring are established. Any colored-recovery contribution must concern the finite-element ownership contract, verified recovery conditions, and controlled comparison, not coloring as a new method. |
| [Analysis of the SiMPL Method, DOI 10.1137/24M1708863](https://epubs.siam.org/doi/10.1137/24M1708863) | An analyzed density-based topology-optimization method published in SIOPT. | The current topology controller cannot supply SIOPT fit without comparable mathematical definition, optimality measures, and convergence or baseline evidence. |

Within the inspected primary-source set, no source reports the same combination
of a direct identical-functional three-route equivalence statement, explicit
owned-row distributed colored-recovery conditions, and the present finite
verification hierarchy. This is an inference from the scoped audit, not proof
of absence from the full literature. The manuscript therefore states novelty
at that narrow methodological level and does not infer a performance result.

## Decision Gate

The route is accepted because each research question has a named proposition,
verification block, or bounded numerical study and does not depend on
repository structure or software breadth. The branch-structured surrogate is
retained only within its verified mathematical status and explicit physical
limitations.
