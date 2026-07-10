# Venue And Contribution Decision

Last updated: 2026-07-10.

## Decision

The primary manuscript route is a scientific-computing paper centered on
derivative placement in distributed nonlinear finite-element Newton methods.
The natural primary venue is the *SIAM Journal on Scientific Computing* or a
closely aligned scientific-computing journal. The revision will retain the
mathematical and evidentiary standards identified by the SIOPT-style review,
but it will not claim a new optimization algorithm unless the topology or
nonsmooth-mechanics work later establishes one.

The topology design loop is supplementary unless EXP-TOPO-001 produces one
fixed feasible problem, verified reduced derivatives, and KKT-quality endpoints.
Plasticity3D remains a central branch-structured benchmark only if the potential,
stress, tangent, and switch behavior pass the mathematical and numerical gates
in Steps 3--6 of [`publication_action_plan.md`](publication_action_plan.md).

## Central Contribution

The intended contribution is to establish and experimentally test when element-level automatic
differentiation, quadrature-point constitutive differentiation, and colored
sparse Hessian recovery are mathematically equivalent and computationally
preferable in distributed nonlinear finite-element Newton methods. The completed
paper will combine fixed-state derivative verification, a prespecified timing model based
on pre-run structural and route-work covariates, separate descriptive
kernel/contraction/assembly/communication diagnostics, visible prespecified
capacity non-attempts, and equal-accuracy route-ordering experiments on
PETSc-owned sparse operators. Memory is reported as an observed capacity
diagnostic; no predictive memory selector or measured failure threshold is
claimed.

This contribution is a conditional selection result and reproducible empirical
methodology. It is not a universal ranking of software frameworks or derivative
routes.

## Research Questions

1. **Correctness.** Under which fixed-functional, quadrature, branch, constraint,
   and ownership assumptions do the three derivative routes produce equivalent
   residuals and Hessian actions?
2. **Crossover.** Which measurable local and distributed costs determine the
   fastest admissible route as polynomial degree, quadrature, constitutive
   complexity, coloring, overlap, and MPI rank count change, and where do
   measured resource use and prespecified non-attempts constrain the supported
   route map? Memory is a capacity diagnostic, not a fitted response, unless a
   separate validated capacity study is added before execution.
3. **Solver interaction.** How do derivative placement and PETSc-owned sparse
   assembly interact with nonlinear accuracy, preconditioner setup, Krylov work,
   and rank-local distributed execution on smooth and branch-structured
   problems?

## Main-Text Evidence

- A smooth scalar reference problem for Taylor, residual, Hessian-action, and
  sparse-recovery verification.
- Hyperelasticity for matched constitutive/element differentiation and
  distributed assembly evidence.
- Plasticity3D as the high-order branch-structured and distributed case, only
  after its mathematical-status and branch diagnostics pass.
- The replicated headline comparison and a controlled fixed-state route map;
  a crossover-location claim remains conditional on a separately released
  post-fit confirmation matrix.

## Supplementary Or Conditional Evidence

- Ginzburg--Landau globalization diagnostics.
- Secondary preconditioner and fixed-work studies.
- One fixed-policy scaling result, only if it satisfies the common accuracy
  contract and materially clarifies the route-selection conclusion.
- Plasticity2D branch formulas and diagnostics unless they directly support the
  central equivalence theorem.
- Topology optimization unless feasibility, KKT, baseline, and rank-consistency
  gates pass.
- Complete benchmark inventories, raw policy matrices, and noncentral state
  figures.

## Explicitly Excluded Claims

- A new automatic-differentiation rule, Krylov method, or universal nonlinear
  solver hierarchy.
- Universal superiority of JAX+PETSc or constitutive AD.
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
| [JetSCI, arXiv:2604.22087](https://arxiv.org/abs/2604.22087) | A hybrid JAX--PETSc framework for scalable differentiable simulation, including finite-element micromechanics and distributed-memory execution. | JAX--PETSc integration is not the novelty. The paper must distinguish its verified derivative-placement comparison, PETSc-owned assembly details, cost model, and CPU crossover evidence. |
| [FE-MAD, arXiv:2606.05199](https://arxiv.org/abs/2606.05199) | JAX-FEM material learning with automatically differentiated Newton tangents and inverse-problem loss gradients. | Constitutive differentiation and differentiable finite elements are established. The contribution must be route equivalence and cost selection in distributed solves, not the availability of AD tangents. |
| [Differentiable finite-strain plasticity, arXiv:2606.17390](https://arxiv.org/abs/2606.17390) | GPU-accelerated JAX finite elements with constitutive differentiation, sparse assembly, nonlinear solution, and inverse characterization. | Plasticity plus JAX is not novel by itself. The manuscript must keep its endpoint-surrogate status explicit and justify the branch/cost experiment independently. |
| [Locality-Aware Automatic Differentiation, arXiv:2509.00406](https://arxiv.org/abs/2509.00406) | Per-element forward-mode differentiation and sparse Hessian assembly for GPU mesh computations, with explicit locality and memory-traffic arguments. | Element-local differentiation and sparse Hessian assembly are not new. The paper must isolate its distributed CPU/PETSc ownership contract, constitutive route, common-functional checks, and crossover design. |
| [tatva, arXiv:2602.12365](https://arxiv.org/abs/2602.12365) | A globally applied AD finite-element framework using matrix-free products and coloring-based sparse tangent materialization, with GPU demonstrations at million-DOF scale. | The local-versus-global AD tradeoff and scalable colored materialization are already explicit in recent work. Novelty cannot rest on applying AD globally or recovering a sparse tangent; it must rest on the three-route equivalence conditions and controlled distributed selection evidence. |
| [FEniCSx external operators, DOI 10.46298/jtcam.14449](https://doi.org/10.46298/jtcam.14449) | Quadrature-point external constitutive operators differentiated with JAX, including Mohr--Coulomb, Taylor tests, and documented software. | Constitutive-point AD for complex plasticity is established. The branch surrogate and route comparison must be positioned as a fixed-functional derivative-placement study, not a new constitutive-AD capability. |
| [Sparser, Better, Faster, Stronger, TMLR 2025](https://openreview.net/forum?id=GtXSN52nIW) | Automatic local/global sparsity detection and efficient sparse Jacobian/Hessian differentiation with coloring. | Sparse AD and coloring are established. Any colored-recovery contribution must concern the finite-element ownership contract, verified recovery conditions, and controlled comparison, not coloring as a new method. |
| [Analysis of the SiMPL Method, DOI 10.1137/24M1708863](https://epubs.siam.org/doi/10.1137/24M1708863) | An analyzed density-based topology-optimization method published in SIOPT. | The current topology controller cannot supply SIOPT fit without comparable mathematical definition, optimality measures, and convergence or baseline evidence. |

Within the inspected primary-source set, no source reports the same combination
of a direct, identical-functional three-route derivative-equivalence contract,
a distributed finite-element cost model, and replicated equal-accuracy CPU/MPI
crossover experiments. This is an inference from the scoped audit, not proof
of absence from the full literature. That narrower combination is the working
contribution. The claim remains provisional until
EXP-DERIV-001, EXP-MC-001 where retained, and both tiers of EXP-ROUTE-001
pass.

## Decision Gate

The route is accepted for implementation because each research question has a
named proof or experiment in the action plan and does not depend on repository
structure or software breadth. If Plasticity3D fails its mathematical gate, the
same contribution will be evaluated on the smooth scalar and hyperelastic cases;
the failed branch-structured case will be reported as a limitation rather than
retained as a headline result.
