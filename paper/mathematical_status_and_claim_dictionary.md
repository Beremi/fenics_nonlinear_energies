# Mathematical Status and Claim Dictionary

Last updated: 2026-07-10. This is an internal publication-control document for
the scientific-computing route selected in
`paper/venue_and_contribution_decision.md`. It records what the manuscript may
claim from the mathematical definitions and the evidence currently present in
the repository. It is not itself a substitute for a theorem in the manuscript
or for a publication experiment.

File-and-line references retained from the initial audit are navigation hints
to that historical snapshot, not immutable anchors for the rewritten
manuscript. The final clean release must regenerate exact claim-to-source and
claim-to-artifact references after all experiment and prose changes freeze.

## 1. Scope decision

The defensible primary route is a SISC-style paper about derivative placement
in distributed nonlinear finite-element Newton methods. The central
mathematical statement is conditional equivalence of element differentiation,
constitutive differentiation, and colored recovery for one fixed discrete
functional. The conditions include the quadrature rule, affine lift,
free-variable map, strain convention, branch, eigenvalue ordering, sparsity
pattern, ghost coverage, and row ownership.

The route map and held-out cost model are P0 evidence because route selection
is the selected empirical contribution. Fixed-policy scaling is conditional P1
evidence: it enters only if it clarifies that route-selection result and may be
removed without changing the central claim.

The benchmark roles are fixed as follows.

| Family | Role in the selected route | Decision at this audit |
| --- | --- | --- |
| $p$-Laplace | Smooth scalar correctness reference | Main text. It is the only family for which uniqueness of the discrete minimizer is established. |
| Ginzburg--Landau | Smooth but nonconvex scalar globalization case | Secondary main-text or supplementary evidence. Report stationary endpoints, not minima. |
| Hyperelasticity | Smooth finite-strain mechanics and distributed assembly case | Main text on the orientation-preserving domain. Report discrete equilibrium, not a certified local minimum. |
| Plasticity2D | Branch-formula and solver diagnostic | Supplementary. The current scalar formula is a synthetic endpoint surrogate; its global projection and stress-potential interpretation are unestablished. |
| Plasticity3D | High-order branch-structured derivative-placement case | Conditional main-text evidence only as a synthetic branch-structured endpoint surrogate. It is not incremental plasticity. Fixed-element checks establish elastic branch-interior behavior; an assembled analytic state and material-point matrix exercise all five labels away from switches. Neither establishes switch regularity or physical constitutive validity. |
| Topology | Coupled design--mechanics software demonstration | Supplementary. No KKT-quality endpoint or convergence result exists for one fixed constrained optimization problem. |

Consequently, the paper must not be presented as a new optimization algorithm,
an analyzed nonsmooth Newton method, or a validated incremental-plasticity
solver. Plasticity3D may support the derivative-placement contribution without
supporting a constitutive-model contribution, provided the fixed-branch and
branch-diagnostic gates in this document pass.

## 2. Evidence vocabulary

Every mathematical statement in the revision should carry one or more of the
following internal statuses.

| Code | Status | Meaning |
| --- | --- | --- |
| **P** | Proved | A displayed derivation or standard finite-dimensional argument establishes the statement under explicit assumptions. Code is inspected only to confirm that it implements the named object; code alone is not a proof. |
| **S** | Source-backed | The statement is standard or is supported by a cited primary source. This audit does not elevate a cited continuum model into validation of the implemented surrogate. |
| **N** | Numerically tested | A focused test or traceable artifact checks the statement on specified inputs and tolerances. It is not a proof outside those inputs. |
| **I** | Inferred | The statement follows plausibly from implementation inspection or observed behavior, but a complete proof or direct test is absent. It may guide work but may not appear as an established conclusion. |
| **U** | Unestablished | The required definition, proof, diagnostic, or evidence is absent, contradictory, or insufficient for publication. |

The status applies to an atomic statement, not to an entire family. For
example, the Plasticity3D element/constitutive Hessian identity is **P** on a
fixed smooth branch and **N** at tested fixed states, while global regularity
across branch switches remains **U**.

## 3. Common discrete objects

Let $x\in\mathbb{R}^n$ denote the free coefficient vector, set
$y_e:=R_e x$, and let the local finite-element state have the affine form

\[
u_e(x)=L_e y_e+\bar u_e.
\]

Here, $R_e$ restricts the global free vector, $L_e$ places free coefficients in
the full element vector, and $\bar u_e$ contains prescribed values. For the
families covered by the derivative-placement result, let

\[
\Phi_e(x)
=\sum_{q=1}^{n_q}w_{eq}
  \mathcal W_{eq}\!\left(\varepsilon_{eq}(x);m_{eq}\right),
\qquad
\varepsilon_{eq}(x)=B_{eq}u_e(x),
\]

where $m_{eq}$ contains fixed material or history data. The global potential is

\[
\Pi_h(x)=\sum_e\Phi_e(x)-f_{\mathrm{ext}}^\top x.
\]

This object is introduced in `paper/sections/methodology.tex:23-100`. The
assembly in `paper/sections/methodology.tex:67-89` is valid on a neighborhood
where all selected element contributions are twice differentiable.

### 3.1 Fixed-branch derivative-equivalence proposition

**Proposition (conditional element/constitutive equivalence).** Assume that:

1. the element, constitutive, and HVP routes evaluate the same scalar
   quadrature functional, with identical weights and local data;
2. $u_e(x)$ is affine in the same constrained free variables for every route;
3. $B_{eq}$ is independent of $x$ on the neighborhood considered;
4. every selected function $\mathcal W_{eq}(\cdot;m_{eq})$ is $C^2$ on that
   neighborhood;
5. all branch predicates and, where relevant, principal-value orderings remain
   fixed on the neighborhood; and
6. all routes use the same engineering/tensor shear convention.

Define

\[
\sigma_{eq}:=\partial_\varepsilon\mathcal W_{eq},
\qquad
C_{eq}:=\partial^2_{\varepsilon\varepsilon}\mathcal W_{eq},
\qquad
A_{eq}:=B_{eq}L_e.
\]

Then, in exact arithmetic,

\[
\nabla_{y_e}\Phi_e
=\sum_qw_{eq}A_{eq}^\top\sigma_{eq},
\qquad
\nabla^2_{y_e y_e}\Phi_e
=\sum_qw_{eq}A_{eq}^\top C_{eq}A_{eq}.
\]

**Proof.** The affine lift has zero second derivative. Applying the chain rule
to each quadrature contribution gives

\[
D\mathcal W_{eq}(\varepsilon_{eq})[d]
=\sigma_{eq}^\top A_{eq}d.
\]

Differentiating once more, with $A_{eq}$ fixed, gives

\[
D^2\mathcal W_{eq}(\varepsilon_{eq})[d,r]
=d^\top A_{eq}^\top C_{eq}A_{eq}r.
\]

Summation over quadrature points proves both identities. Equality of mixed
partials makes $C_{eq}$ and the assembled element Hessian symmetric. This proof
does not apply at a branch switch, at an unresolved repeated-principal-value
point, or when a route changes the quadrature, lift, constraints, or strain
map. $\square$

**Status.** The core chain-rule argument is **P** and already appears in shorter
form in `paper/sections/methodology.tex:126-151`. The implementation mirrors it
for Plasticity3D in
`src/problems/slope_stability_3d/jax/jax_energy_3d.py:262-357`. The current
pilot artifacts under
`artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-DERIV-001/`
provide **N** evidence for P1, P2, and P4 fixed elements: the maximum reported
residual differences are between $2.03\times10^{-16}$ and
$2.35\times10^{-16}$, the maximum Hessian differences are between
$1.20\times10^{-16}$ and $2.34\times10^{-16}$, and the maximum symmetry defects
are at most $1.25\times10^{-16}$. The enhanced runner also records the
production branch label, normalized active-branch margin, principal-value gap,
denominator margin, and tie-break scale, and replays each centered
finite-difference perturbation through the production branch predicates. All
sampled points and perturbations remained in the elastic branch. Across the
pilot, the minimum normalized active-branch margin was $0.971$, the minimum
normalized principal-value gap was $0.0691$, the minimum normalized
denominator margin was $0.703$, and the largest tie-break scale relative to the
unperturbed strain-matrix norm was $2.65\times10^{-8}$. Centered
energy-derivative and Hessian-action errors were at most
$7.53\times10^{-15}$ and $1.60\times10^{-14}$, respectively. These diagnostics
upgrade the tested elastic branch-interior statement to **N**. A separate
assembled `P2(L1)` pilot uses an analytic state whose production predicate
replay contains 157787 elastic, 32218 shear, 9881 left-edge, 1922 right-edge,
and 801 apex quadrature points. No point lies inside the normalized
$10^{-8}$ switch band, and the minimum recorded branch margin is
$3.65\times10^{-5}$. At that identical state, the colored-SFD tangent action
is bitwise identical to element AD and the constitutive-AD action differs by
$2.34\times10^{-16}$ relative. This is **N** evidence for constructed
branch-interior coverage.

EXP-MC-001 adds a dimensionless material-point matrix with one strict interior
for each of the five branches, all five nonsingular adjacent interfaces from
both sides at three offsets, three seeded rotations per branch, and seven exact
or nearly repeated spectra. Its maximum centered energy/HVP errors at
$h=10^{-6}$ are $1.06\times10^{-10}$ and $1.54\times10^{-9}$, and maximum
rotation energy/stress/tangent errors are $1.82\times10^{-14}$,
$3.41\times10^{-14}$, and $9.33\times10^{-15}$. The selected Hessian changes
by approximately $0.396$ to $1$ in scaled norm across the closest interface
pairs even while energy/gradient differences shrink; these are descriptive
one-sided observations, not a regularity theorem. Exact repeated spectra are
finite only under the coordinate-dependent tie break. The NumPy comparator is
a separate algebra transcription, not an independently derived material law.
Thus fixed interior, rotation-away-from-degeneracy, and finite tie-broken
output statements are **N**; generalized switch behavior and independent
constitutive validity remain **U**.

### 3.2 Conditional colored-recovery proposition

Let $H=\nabla^2\Pi_h(x)$ at a point where this Hessian exists. Let
$\mathcal P$ be a structural pattern containing every nonzero of the owned rows
of $H$. Color columns so that no two columns of one color occur in the same
owned row of $\mathcal P$. For a color class $\mathcal C_c$, define
$v_c=\sum_{j\in\mathcal C_c}e_j$. Then each entry $H_{ij}$ with
$j\in\mathcal C_c$ is the unique contribution to $(Hv_c)_i$ in that row.
Consequently, exact AD-HVPs recover the selected owned rows in exact arithmetic.

This result additionally requires complete ghost coverage for all owned-row
dependencies and unique global row ownership. Rank-local color labels may
differ because only row interference matters. The proof is the one-term sum

\[
(Hv_c)_i=\sum_{j\in\mathcal C_c}H_{ij}=H_{ij_i},
\]

where pattern separation leaves at most one active $j_i$ in row $i$.

**Status.** The algebraic recovery statement is **P** under the stated pattern
and ownership assumptions. The implementation result is **regression-tested,
not archived publication evidence**. The manuscript states the conditions in
`paper/sections/methodology.tex:295-331`, and the owned-row interference rule is
tested in `tests/test_reordered_element_sfd_coloring.py:8-21`. EXP-DERIV-001
now includes a deterministic `P1(L1)` `MPI.COMM_SELF` assembled-state
regression with 10526 free degrees of freedom. Element AD and local colored
SFD produce bitwise-identical CSR matrices; constitutive AD differs by
$1.01\times10^{-16}$ in relative Frobenius norm, and all symmetry defects are
approximately $4.6\times10^{-17}$. The P2 all-branch pilot independently saves
one tangent action per route and gives the action errors stated in Section 3.1.
This supplies development regression coverage and dirty diagnostic support for
serial assembled recovery; it must not be promoted as clean **N** publication
evidence. Canonical
multi-rank ownership/ghost comparisons and an independently assembled global
residual remain **U**. Finite-difference probes would add truncation and
roundoff error and must never be called exact recovery; the reported recovery
uses AD-HVPs.

## 4. Benchmark status overview

| Family | Exact reported discrete object | Regularity relevant to the solver | Convexity | Computed-object label | Strongest permissible conclusion now |
| --- | --- | --- | --- | --- | --- |
| $p$-Laplace | Quadrature-defined restriction of $\int |\nabla u|^3/3-fu$ to constrained P1 functions | $C^2$ in coefficients; the Hessian is not uniformly positive at zero gradient | Strictly convex | Approximate first-order stationary point of a strictly convex discrete functional | The mathematical discrete problem has a unique minimizer; a run approximates that target only under its stated residual contract. |
| Ginzburg--Landau | $\mathcal E_{h,2}$ for JAX routes; $\mathcal E_{h,4}$ for FEniCS | $C^\infty$ in coefficients | Nonconvex | Approximate first-order stationary point or capped solve | Fixed-functional derivative statements apply only to the two JAX routes using $\mathcal E_{h,2}$. No minimum is certified. |
| Hyperelasticity | P1 discrete potential on states with positive element determinants | $C^\infty$ on the open orientation-preserving set | Nonconvex in displacement | Approximate discrete equilibrium at a load step | The endpoint satisfies the stated first-order test for the named discrete potential. No uniqueness or local minimality is established. |
| Plasticity2D | AD of the displayed piecewise scalar endpoint surrogate | Smooth only inside regular branch regions; global class unestablished; the repeated-stress apex now has an invariant finite-AD implementation, but switch regularity remains unestablished | Unestablished | Approximate branchwise stationary endpoint or fixed-work diagnostic | Software behavior of a synthetic scalar endpoint surrogate only. |
| Plasticity3D | AD of the displayed eigenvalue- and branch-defined scalar endpoint surrogate | $C^2$ on a fixed branch with separated principal values; global class unestablished | Unestablished and generally not assumed convex | Approximate branchwise stationary endpoint or fixed-work diagnostic | Fixed-state derivative-route equivalence for the implemented surrogate, subject to branch and eigenvalue margins. No incremental-plasticity conclusion. |
| Topology | A sequence of mechanics solves and changing frozen reciprocal design objectives | Each finite-$z$ frozen objective is smooth; the outer continuation/controller is not one fixed objective | Nonconvex | Reduced-design endpoint or fixed-work diagnostic | A software demonstration and rank-sensitivity study. It is not an optimization solution or a KKT point. |

The coefficient-space gradient and step norms used by historical runs are not
mesh- or degree-independent. A common Euclidean/diagonal/matrix Riesz-metric
interface and dual-residual/primal-correction history now exist in
`src/core/petsc/metrics.py` and `src/core/petsc/minimizers.py`, with their
contract documented in `docs/reference/convergence_metrics.md`. The shared
scalar driver now has an optional positive P1 lumped-mass map, with the unit-
field lumped-$L^2$ norm as its default state scale and exact diagonal dual
evaluation. Plasticity3D also has an optional reference-elastic-energy map on
the glued constrained free space. A two-rank P1 smoke certified symmetry and
inertia $(0,0,10526)$ and passed the independently recomputed Riesz-solve
residual gate. Hyperelasticity now has a separate fixed reference-energy map:
the exact discrete tangent is copied at $y(X)=X$ after both end-face constraints
are eliminated, and the absolute deformation map's initial energy norm supplies
the default state scale. Its one- and two-rank P1 smokes certified inertia
$(0,0,2133)$ and passed the same true-residual gate. This is **N** infrastructure
evidence, not a calibrated stopping policy.
Until each retained benchmark map is calibrated and used to regenerate its
endpoints, `completed` means only that the recorded solver-specific tests were
satisfied.

## 5. Family audits

### 5.1 $p$-Laplace

**Defined object.** The continuous weak problem uses $p=3$ on the L-shaped
domain with homogeneous Dirichlet data. The discrete functional is the
constrained P1 restriction in `paper/sections/benchmarks.tex:101-150`. The JAX
implementation evaluates

\[
\mathcal E_h(x)
=\sum_e\frac{|e|}{3}|\nabla u_h|_e^3-f_h^\top u_h,
\]

as seen in `src/problems/plaplace/jax/jax_energy.py:7-15`. The P1 gradient is
constant on each affine triangle; the internal integral is therefore exact for
the implemented discrete state. The load is the common assembled vector and
must be treated as part of the discrete definition.

**Assumptions.** We require $p>1$, a conforming subspace of
$W_0^{1,p}(\Omega)$, a nonempty constrained space, and a consistently assembled
load. The L-shaped domain supports a weak formulation, but no classical
solution regularity at the reentrant corner is claimed.

**Mathematical status.** The map $z\mapsto|z|^p/p$ is strictly convex for
$p>1$. Since homogeneous Dirichlet conditions make the discrete gradient map
injective modulo the zero function, the discrete functional is strictly
convex. Coercivity in the finite-dimensional constrained space gives existence,
and strict convexity gives uniqueness. This is **P**. For $p=3$, the density is
$C^2$, with

\[
\nabla_z\frac{|z|^3}{3}=|z|z,
\qquad
\nabla_z^2\frac{|z|^3}{3}
=|z|I+\frac{zz^\top}{|z|}\quad(z\ne0),
\]

and the Hessian extends continuously by zero at $z=0$. It is not uniformly
positive definite there. Thus strict convexity does not by itself establish
the nonsingularity and local-rate assumptions of every Newton iteration.

**Evidence status.** The continuous model and weak formulation are **S**. The
strict-convexity and uniqueness statement is **P**. Existing energy agreement
and solver runs are **N** for their named meshes and stopping tests. An
independent NumPy/SciPy manufactured problem with exact solution
$x+y+0.1\sin(\pi x)\sin(\pi y)$ gives successive P1 $L^2$ rates 1.986, 1.996,
and 1.999 and $H^1$-seminorm rates 0.990, 0.998, and 0.999 on 8--64
subdivisions; every level passes a $10^{-8}$ relative weak-residual gate. This
is **N** for that smooth unit-square problem, not for reentrant-corner
regularity of the production L-shaped case. A production-case Riesz-scaled
residual audit and Newton-rate theorem remain **U**.

**Permitted wording.** Use `the unique discrete minimizer` for the exact
mathematical argmin. For a computed vector, use `an approximation to the unique
discrete minimizer satisfying the stated first-order test`. Do not call a
computed vector the exact minimizer.

### 5.2 Ginzburg--Landau

**Defined object.** The paper uses the real scalar double-well functional

\[
\mathcal E(u)=\int_\Omega
\frac{\varepsilon}{2}|\nabla u|^2+\frac14(u^2-1)^2\,\mathrm dx,
\qquad \varepsilon=10^{-2},
\]

with homogeneous Dirichlet conditions
(`paper/sections/benchmarks.tex:198-257`). Both JAX routes use the stored
three-point degree-two triangle rule, implemented in
`src/problems/ginzburg_landau/jax/jax_energy.py:7-16` and
`src/problems/ginzburg_landau/jax_petsc/parallel_hessian_dof.py:27-80`.
FEniCS uses a degree-four rule. Since $u_h^4$ is quartic on a P1 element, the
degree-four rule is the exact polynomial rule while the degree-two rule defines
an underintegrated, but explicit, discrete functional.

**Variables and basin.** The free P1 coefficients are initialized by the fixed
sine field in `src/problems/ginzburg_landau/jax/mesh.py:20-23`. The prescribed
initial state is part of the benchmark because the functional is nonconvex.

**Mathematical status.** Both finite-dimensional functionals are polynomial
and therefore $C^\infty$. Their Hessians can be indefinite. A global discrete
minimizer exists under the finite-dimensional coercive model, but the Newton
run is not a global minimization method and no second-order test is reported.
The selected endpoint may be a local minimum, saddle, or another stationary
point. Only stationarity is tested.

**Evidence status.** The model setting is **S**; the exact discrete distinction
between $\mathcal E_{h,2}$ and $\mathcal E_{h,4}$ is **P** by inspection of the
quadrature-defined functionals. Same-functional element/colored comparisons
within JAX are **N** where directly tested. A FEniCS--JAX derivative-equivalence
claim is **U** because the quadrature differs. A source-extended manufactured
problem using the production-style three-point rule stays on the controlled
positive branch $u_h\ge0.8$ and gives successive $L^2$ rates 2.010, 2.003, and
2.001 and $H^1$-seminorm rates 0.997, 0.999, and 1.000. This is **N** for the
selected smooth branch and spatial discretization; it does not establish basin
robustness for the zero-source benchmark. Local minimality and basin uniqueness
remain **U**.

**Permitted wording.** Use `stationary endpoint reached from the prescribed
initial state`, `nonconvex globalization diagnostic`, and `endpoint-energy
comparison across different quadrature-defined functionals`. Do not use
`identical-functional verification` for the FEniCS comparison.

### 5.3 Hyperelasticity

**Defined object.** For $F=I+\nabla u$ and $J=\det F>0$, the density is

\[
W(F)=C_1\bigl(\operatorname{tr}(F^\top F)-3-2\log J\bigr)
     +D_1(J-1)^2,
\]

with $C_1,D_1>0$. The load-step problem has a left clamp, a rotating prescribed
right-face displacement, and traction-free remaining faces
(`paper/sections/benchmarks.tex:292-385`). On affine P1 tetrahedra, $F$ is
cellwise constant, so the one-point volume evaluation is exact for this
discrete potential.

**Nondimensional system.** Let $L_0$ and $S_0$ denote reference length and
stress scales. The manuscript reports coordinates divided by $L_0$, energy
density and stress-like parameters divided by $S_0$, and body-force density
multiplied by $L_0/S_0$. The stored inputs are already scaled and therefore use
$L_0=S_0=1$ computationally. No SI attribution is made.

**Admissible set and lift.** The finite-dimensional domain is the open set of
free coefficient vectors for which every evaluated element determinant is
positive. The prescribed rotation is an affine lift. The JAX implementations
store deformed coordinates rather than displacement; this is an affine change
of variables and does not alter stationarity. The main JAX+PETSc density uses
the signed determinant by default
(`src/problems/hyperelasticity/jax_petsc/parallel_hessian_dof.py:17-61`). The
serial pure-JAX diagnostic uses $|\det F|$
(`src/problems/hyperelasticity/jax/jax_energy.py:7-30`) and is therefore a
different extension outside the orientation-preserving domain.

**Mathematical status.** The density and discrete potential are $C^\infty$ on
$J>0$. The first Piola stress $P=\partial W/\partial F$ and its consistent
element tangent follow by differentiation, which is **P**. The energy is
nonconvex as a function of the displacement coefficients. The load-step solver
targets the first variation, not a second-order sufficient condition. No
uniqueness, local minimality, or path-independent trajectory is established.

**Evidence status.** The finite-strain model is **S**. The element-energy and
rank-local assembly definitions are code-backed, and replicated versus
rank-local energy, gradient, and Hessian agreement is **N** in
`tests/test_reordered_element_base.py:292-355`. On a unit-cube affine patch
with $J=1.139187$, the production JAX energy/gradient/Hessian agree with an
independent analytic Piola/tangent assembly to $4.58\times10^{-15}$,
$2.67\times10^{-15}$, and $4.38\times10^{-16}$ relative. Boundary-traction
balance, objectivity, Piola covariance, and rigid-translation checks are also
near roundoff. This is **N** for the exactly representable admissible patch.
An independent nonaffine manufactured problem with
$y=(X+0.05\sin(\pi X)\sin(\pi Y)\sin(\pi Z),Y,Z)$ gives last-pair P1 rates
1.887 in displacement $L^2$, 1.006 in deformation-gradient error, and 0.983
in first-Piola stress on 4--24 subdivisions. Order-4/6/8 load checks bound the
largest response change by $8.29\times10^{-6}$ of the finite-element error and
the load change by $1.78\times10^{-5}$ of the exact-interpolant consistency
residual. All levels pass a $10^{-10}$ relative algebraic gate, and the minimum
discrete determinant is 0.844. This
is **N** for the independently assembled manufactured weak problem, not for
the production rotating-beam backend or load trajectory.
Endpoint comparison with JAX-FEM is **N** for the stated observables and common
post-evaluated energy, not for a common nonlinear path. Orientation
preservation of every trial in the production load path and local minimality of
reported endpoints are **U**.

**Permitted wording.** Use `approximate discrete equilibrium`, `stationary
load-step endpoint`, and `orientation-preserving endpoint`. Use
`same-functional derivative comparison` only when both routes use the signed
determinant, the same quadrature, lift, and state. Do not infer trajectory-level
or constitutive-law equivalence from endpoint-energy agreement.

### 5.4 Plasticity2D

**Defined object.** The code defines a plane-strain, in-plane-principal-value
branch surrogate. It is not a full three-principal-stress plane-strain return.
The endpoint sets the previous plastic strain to zero, applies Davis-B-reduced
parameters, and differentiates the scalar in
`paper/sections/benchmarks.tex:490-669`. The corresponding implementation is
`src/problems/slope_stability/jax/jax_energy.py:11-108` and
`src/problems/slope_stability/jax/jax_energy.py:121-197`. P1, P2, and P4 use
1-, 7-, and 19-point triangle rules, respectively. These rules are part of the
surrogate; no nonpolynomial exactness claim is permitted.

**Parameter assumptions.** The elastic matrix is positive definite when
$E>0$ and $-1<\nu<1/2$. The Davis-B map requires
$\lambda_{\mathrm{sr}}>0$ and a nonzero denominator
$1-\sin\phi_1\sin\psi_1$. The line-return denominator is positive if the
principal-space compliance is positive definite and the yield normal is
nonzero. The apex formula additionally requires

\[
a_\lambda-b_\lambda=2\sin\phi_\lambda\ne0.
\]

The reported parameters satisfy positive friction, but the displayed formula
is not valid as an apex formula at zero friction. The code's safe replacement
of a small denominator is an algorithmic fallback, not a proof that the
singular material case is defined.

**Projection and envelope status.** If $\sigma^\star(\varepsilon)$ were proved
to be the unique maximizer of

\[
\sigma^\top\varepsilon^e-\frac12\sigma^\top S\sigma

\]

over one fixed closed convex admissible stress set, a standard envelope
argument would give $\nabla_\varepsilon\Phi=\sigma^\star$. That admissible set,
its KKT system, and equivalence of every implemented branch formula have not
been supplied. Moreover, the code regularizes the principal radius used by the
branch decision but has no derived regularized projection problem. Therefore,
the global projection theorem and envelope identity are **U** and must not be
asserted.

The formula is smooth inside an elastic region and inside a nondegenerate
selected line or apex region. Its continuity, $C^1$ matching, Lipschitz class,
and tangent definiteness across yield and line--apex interfaces remain **U**.
The radius regularization removes one square-root singularity but does not
regularize the branch switches.

**Resolved repeated-principal implementation defect.** The original kernel
evaluated the derivative of `atan2(0,0)` while reconstructing a hydrostatic
plastic-apex stress. At the reported $E=40000$, $\nu=0.3$, Davis-B parameters
for $c_0=6$, $\phi=45^\circ$, $\psi=0$, and
$\lambda_{\mathrm{sr}}=1$, the scalar value at trial strain
$(10^{-2},10^{-2},0)$ was finite but its AD gradient was not. The implementation
now constructs the selected apex stress directly as the invariant isotropic
tensor $(s_{\mathrm a},s_{\mathrm a},0)$ and evaluates the principal angle only
inside the non-apex line-return branch. This change removes an irrelevant
undefined angle without changing the regular line-return formula.

Focused regressions give value $0.119532$, gradient $(6,6,0)$, and a zero
selected-branch Hessian at that apex state; all are finite. They also check
centered directional derivatives, rotation covariance, exact and nearly
hydrostatic states, a repeated-principal yield neighborhood, preservation of a
nondegenerate line-return reference, serial JAX/PETSc agreement, and unchanged
energies on 256 random nondegenerate states. This is **N** evidence for the
implemented selected-branch convention. It is not evidence of classical
differentiability, continuity of the tangent, or a generalized derivative at
the yield or line--apex switch surfaces; those statements remain **U**.

**Endpoint meaning.** The solver differentiates the scalar surrogate itself.
Unless the missing projection result is established, its derivative is an
`algorithmic surrogate gradient`, not a verified physical stress update. A
completed run is at most an approximate branchwise stationary endpoint of the
scalar surrogate. Larger capped cases are fixed-work diagnostics.

**Permitted wording.** Use `2D synthetic Mohr--Coulomb-inspired endpoint
surrogate`, `active-branch AD derivative`, and `solver diagnostic`. Do not use
`path-consistent plasticity`, `return-mapping validation`, `physical failure
mechanism`, or `consistent elastoplastic tangent`.

### 5.5 Plasticity3D

**Defined object.** The implementation forms the six-component engineering
strain, converts shear components before computing principal strains, selects
one of the elastic, shear, left-edge, right-edge, or apex scalar branches, and
differentiates the selected scalar. The definitions are in
`paper/sections/benchmarks.tex:845-1097`; the implementation is in
`src/problems/slope_stability_3d/jax/jax_energy_3d.py:12-191`. Previous plastic
and history variables are reset. The global object is therefore a synthetic
branch-structured endpoint surrogate, not a load increment of an
elastoplastic evolution.

The historical P1, P2, and P4 solve rules use 1, 11, and 24 tetrahedral points.
The integrand is branchwise nonpolynomial, so none is an exactness guarantee.
The current implementation also defines a positive 125-point Duffy rule as an
independent fixed-state reference
(`src/problems/slope_stability_3d/support/mesh.py:1018-1248`). Its exactness for
polynomials through total degree seven does not make it exact for the branch
surrogate. Publication claims require fixed-state sensitivity results, not the
existence of the rule alone.

**Material and denominator assumptions.** Sufficient common assumptions are

\[
E>0,\qquad -1<\nu<\frac12,\qquad
\mu>0,\qquad K>0,\qquad -1<s_\lambda<1.
\]

The implementation computes

\[
\mu=\frac{E}{2(1+\nu)},\qquad
K=\frac{E}{3(1-2\nu)},\qquad
\lambda_{\mathrm L}=K-\frac{2\mu}{3}
=\frac{E\nu}{(1+\nu)(1-2\nu)}.
\]

Here $c_0$ and $E$ use the common nondimensional stress scale, while the unit
weight $\gamma$ uses stress per reference length. The local fixed-element
derivative diagnostic uses $\lambda_{\mathrm{sr}}=1.50$; the principal
assembled route, stopping, and discretization studies use
$\lambda_{\mathrm{sr}}=1.55$. Their evidence rows must remain separate.

Writing $\lambda_{\mathrm L}=K-2\mu/3$, the three non-apex denominators can be
rewritten as

\[
\begin{aligned}
D_{\mathrm s}&=4Ks_\lambda^2+4\mu(1+s_\lambda^2/3),\\
D_{\mathrm l}&=4Ks_\lambda^2
 +\mu(3-2s_\lambda+s_\lambda^2/3),\\
D_{\mathrm r}&=4Ks_\lambda^2
 +\mu(3+2s_\lambda+s_\lambda^2/3).
\end{aligned}
\]

They are strictly positive under these assumptions. The apex denominator
$D_{\mathrm a}=4Ks_\lambda^2$ is strictly positive only when
$s_\lambda\ne0$. Thus the displayed apex branch excludes zero friction. The
code's signed tiny-denominator replacement
(`src/problems/slope_stability_3d/jax/jax_energy_3d.py:33-35`) defines a
fallback program, not the displayed mathematical model at a singular
parameter.

For the four reported materials, $\psi=0$. Sampling
$\lambda_{\mathrm{sr}}=1$ and $1.55$ gives reduced values
$s_\lambda\in[0.3296,0.5243]$. A direct evaluation gives all four denominators
positive, with the smallest sampled value approximately $4.259\times10^3$.
This is **N** for the sampled material/end-point combinations; the algebraic
positivity result above is **P** under the stated assumptions.

**Branch and eigenvalue limitations.** On a neighborhood with one selected
branch and simple, consistently ordered principal values, every branch formula
is $C^2$ and the proposition in Section 3.1 applies. Across branch interfaces,
value matching, gradient matching, semismoothness, and membership of the
selected tangent in a generalized Jacobian are **U**. At repeated principal
values, individual ordered eigenvalues need not be differentiable.

The code adds

\[
\delta\operatorname{diag}(0,1,2),\qquad \delta=10^{-15},
\]

before calling `eigvalsh`
(`src/problems/slope_stability_3d/jax/jax_energy_3d.py:38-70`). This is a
coordinate-dependent perturbation, not a material parameter or a spectral
regularization theorem. Weyl's bound limits the absolute change of each
principal value from this perturbation by $2\delta$. Comparing two rotated
representations can therefore differ by at most $4\delta$ per principal value
through this device. The perturbation breaks exact rotational invariance at
that scale, does not guarantee a simple spectrum for every input, and can alter
a branch decision when a branch margin or eigenvalue gap is of comparable
size. The invariant $I_1$ is computed from the unperturbed strain while the
principal values use the perturbed tensor, so the perturbation must remain an
explicit implementation convention.

**Derivative status.** Element AD and constitutive AD differentiate the same
selected branch, and their equality is **P** under Section 3.1. Production
residuals remain element-energy derivatives while the constitutive route
changes the Hessian construction boundary
(`paper/sections/methodology.tex:126-131`). Fixed-element P1/P2/P4 agreement is
**N**, as summarized in Section 3.1. The enhanced element pilot records branch
maps, normalized branch margins, principal-value gaps, denominator margins,
tie-break scales, centered derivative errors, and the branch labels of both
finite-difference perturbations. The assembled P1 regression establishes
serial three-route matrix agreement on an elastic state. The assembled P2
fixed-state screen exercises every production branch label away from its
recorded switch band and establishes three-route tangent-action agreement
there. EXP-MC-001 further checks all five strict material-point interiors,
two-sided interface approaches, rotations away from spectral degeneracy, and
finite output at repeated spectra under the explicit tie break. These are
**N** results for the named constructed states. They do not establish
interface regularity, a generalized Jacobian, coordinate-free rotation
behavior at repeated spectra, independent constitutive correctness, or
publication-grade multi-rank equality; those statements remain **U**.

**Endpoint and stopping status.** A completed solve may be called an
`approximate branchwise stationary endpoint of the discrete surrogate` only
when its final scaled residual and correction satisfy the publication
contract. Historical results primarily use an unscaled Euclidean free-DOF
gradient or correction, so they are not comparable across mesh or degree until
regenerated. No second-order condition, generalized-stationarity residual, or
incremental history residual is available.

The reference-elastic-energy convergence map is implemented for glued-bottom
Plasticity3D. It hashes the mesh/free-space/permutation/material inputs,
certifies the elastic operator as SPD through a Cholesky/inertia check, and
records independently recomputed norm-solve residuals. Its P1 smoke passes,
but P1/P2/P4 tolerance calibration, cross-mesh scaling, and route-runner
integration are not yet complete. Therefore, no historical endpoint is
retrospectively upgraded by this implementation.

**Quadrature status.** The enhanced fixed-state evaluator independently
rebuilds the named-rule energy, free residual, deterministic Hessian action,
and branch diagnostics. Historical P2 and P4 solve-rule energies differ from a
125-point evaluation by only $1.84\times10^{-8}$ and
$5.10\times10^{-6}$ relative, but the same states have 125-point free-residual
norms $7.84\times10^2$ and $2.57\times10^3$ and Hessian-action changes of
2.08% and 4.35%. One P2 reference point also falls inside the
$10^{-8}$ branch-margin exclusion band. Consequently, the old energy-only
quadrature interpretation is rejected. Both enriched-rule problems must be
solved and pass their own residual gates before an endpoint comparison is
admissible; the 125-point rule is a common evaluator, not exact truth.

**Permitted wording.** Use `3D synthetic branch-structured endpoint surrogate`,
`selected-branch potential`, `active-branch gradient`, `active-branch tangent`,
and `fixed-state derivative-route agreement`. `Surrogate stress` is permitted
only when explicitly defined as
$\partial_\varepsilon\Phi_{\mathrm{MC},3\mathrm D}$; it must not be presented
as a validated return-mapping stress. Do not use `incremental Mohr--Coulomb
solution`, `path-consistent`, `classical consistent tangent`, `failure load`, or
`factor of safety`.

### 5.6 Topology

**Defined objects.** The implementation contains three distinct objects:

1. a linear-elastic mechanics equation for a fixed element density;
2. a reduced compliance observable
   $\mathcal C_h(\theta)=f^\top u(\theta)$; and
3. a changing frozen reciprocal design objective at each outer iteration.

The manuscript definitions are in
`paper/sections/benchmarks.tex:1214-1446`. The current material-measure semantics
are explicit:

\[
M_h(\theta)=\sum_e|e|\theta_e,
\qquad
\bar\theta_h=M_h(\theta)/|\Omega|.
\]

The current code implements this distinction in
`src/problems/topology/jax/jax_energy.py:109-122`,
`src/problems/topology/jax/parallel_support.py:1483-1504`, and
`src/problems/topology/support/volume.py:9-88`. The semantics are **N** in
`tests/test_topology_volume_semantics.py:11-60` and in the parallel smoke test
`tests/test_topology_parallel_topopt_smoke.py:80-102`. Historical artifacts
predating semantics version 2 must not be mixed with regenerated outputs
without an explicit schema conversion.

**Mechanics assumptions.** For $E>0$, an admissible Poisson ratio, positive
$\theta_{\min}$, positive element densities, and a clamp eliminating rigid
modes, the assembled mechanics matrix is symmetric positive definite on the
free space. Hence the exact discrete mechanics state is unique. Iterative KSP
error must be included when claiming reduced-gradient accuracy.

**Frozen-model consistency.** Let

\[
A(\theta)=\sum_e|e|\theta_e^pK_e,
\qquad A(\theta)u=f,
\qquad \mathcal C(\theta)=f^\top u.
\]

At the current exact state $u^m$, define
$q_e^m=(u_e^m)^\top K_eu_e^m$ and

\[
m_e^m(\theta_e)
=(\theta_e^m)^{2p}q_e^m\theta_e^{-p}.
\]

Then

\[
m_e^m(\theta_e^m)=(\theta_e^m)^pq_e^m,
\qquad
\frac{\mathrm d m_e^m}{\mathrm d\theta_e}(\theta_e^m)
=-p(\theta_e^m)^{p-1}q_e^m.
\]

Differentiating $A(\theta)u=f$ gives

\[
D\mathcal C(\theta^m)[d]
=-(u^m)^\top DA(\theta^m)[d]u^m
=-\sum_e|e|p(\theta_e^m)^{p-1}q_e^m d_e.
\]

Thus the reciprocal compliance term has the same value and first derivative as
the exact reduced compliance at the current design, provided the mechanics
state is exact and $p$ is fixed. The element-average/nodal chain rule preserves
this statement for the latent design variables. The exact regularization and
material terms remain differentiable, while the move term has zero value and
gradient at the current iterate. This local first-order consistency result is
**P** and matches
`src/problems/topology/jax/parallel_support.py:1385-1393` and
`src/problems/topology/jax/parallel_support.py:1362-1375`.

This proof does **not** show that the reciprocal model is a global majorizer, a
trust-region model with controlled error, or a convergent sequential
optimization method. Those statements are **U**.

**Outer algorithm status.** The multiplier combines an empirical sensitivity
quantile, a carried correction, and a residual penalty. The SIMP exponent,
frozen state, multiplier, and proximal center change with the outer iteration.
The stall termination can declare completion before the final exponent and
without enforcing the material tolerance
(`src/problems/topology/jax/solve_topopt_parallel.py:657-683`). Therefore, the
history is not descent for one fixed objective and the endpoint is not a KKT
point of a fixed constrained problem.

For a fixed constrained problem, publication-quality evidence would require at
least the mechanics residual, material feasibility, the exact reduced
gradient, a projected or Lagrangian stationarity residual, and bound
complementarity. None is currently reported together at the endpoint. The
logistic map enforces open physical bounds for finite latent variables and can
also make coefficient-space gradients small near saturation; a latent gradient
alone is not a sufficient KKT metric in physical density variables.

**Evidence status.** Mechanics and design kernels, rank smoke tests, and volume
units are **N**. The local reciprocal tangency statement is **P**. Overall
algorithm convergence, a fixed-problem KKT endpoint, comparison with an
accepted topology optimizer, and a topology-optimization solution are **U**.

**Permitted wording.** Use `distributed frozen-design demonstration`,
`reduced-design endpoint`, `material-measure residual`, `normalized material
fraction`, and `rank-sensitivity diagnostic`. Do not use `topology-optimization
solution`, `optimal topology`, `converged design`, or `KKT point` unless a new
fixed-problem campaign reports the required residuals.

## 6. Solver regularity decision

The nonlinear solver language must follow the regularity of the object being
differentiated.

| Case | Available derivative | Defensible solver interpretation | Missing theorem or diagnostic |
| --- | --- | --- | --- |
| $p$-Laplace | Ordinary gradient and Hessian; Hessian may degenerate where element gradients vanish | Globalized Newton applied to a strictly convex discrete functional | Run-specific nonsingularity and convergence-rate assumptions are not proved. |
| Ginzburg--Landau | Ordinary smooth gradient and Hessian | Globalized Newton stationarity solver for a nonconvex polynomial functional | No guarantee that the selected stationary point is a minimum. |
| Hyperelasticity | Ordinary smooth derivatives while every $J_e>0$ | Globalized Newton equilibrium solver on the admissible open set | No global admissibility or local-minimum certificate. |
| Plasticity2D/3D away from switches | Ordinary derivative of one selected smooth branch | Active-branch Newton heuristic for the synthetic surrogate | No proof that steps remain on one branch or that the tangent is a generalized Jacobian at a switch. |
| Plasticity2D/3D at switches or unresolved repeated principal values | Ordinary Hessian may not exist | No ordinary-Newton convergence claim is permitted | Global regularity, generalized stationarity, semismoothness, and finite repeated-value derivatives are unestablished. |
| Topology frozen subproblem | Smooth gradient for a fixed finite-$z$ model | Inexact descent on the current frozen objective | No convergence result for the changing outer sequence or KKT result for the original problem. |

The algorithm in `paper/sections/methodology.tex:170-268` is therefore a policy
description, not a convergence theorem. In particular, `active-branch tangent`
does not mean `semismooth Newton derivative`. That stronger label requires a
proved global regularity class and generalized-Jacobian membership.

## 7. Claim dictionary

### 7.1 Object labels

| Term | Permitted meaning | Required qualifier or evidence |
| --- | --- | --- |
| `discrete functional` | The exact scalar program defined by named mesh, element space, quadrature, lift, constraints, parameters, and branch convention | State every item when comparing routes. |
| `residual` | The named state-equation residual or the free-DOF gradient of a named scalar | Do not silently interchange coefficient, Riesz-scaled, true KSP, and preconditioned residuals. |
| `gradient` | Ordinary derivative at a differentiability point | At a plastic switch, use `selected-branch derivative` unless a generalized derivative is established. |
| `Hessian` | Second derivative of a named scalar at a twice differentiable point | At switches, use `selected-branch tangent matrix`; do not imply a global Hessian exists. |
| `stress` | For hyperelasticity, $\partial W/\partial F$; for a plastic surrogate, only a quantity explicitly defined and checked | Use `surrogate stress` or `active-branch potential derivative` for plasticity. |
| `consistent tangent` | Derivative of the same residual used by the nonlinear solve | For plasticity, say `consistent derivative of the selected scalar branch`, not `classical elastoplastic consistent tangent`. |
| `endpoint surrogate` | A one-state scalar construction with reset history | Mandatory for both plasticity families. |
| `equilibrium` | A state satisfying a named discrete force residual | Unqualified equilibrium is permitted for hyperelasticity only with a residual gate. For plasticity, say `surrogate stationarity`. |
| `stationary point` | A point satisfying a first-order residual tolerance for a differentiable fixed functional | Use `branchwise stationary endpoint` for plasticity and state the branch margin. |
| `minimizer` | An exact mathematical minimizer or a point with justified first- and second-order conditions | Unqualified uniqueness is permitted only for the exact discrete $p$-Laplace solution. |
| `design point` | A reported topology iterate | Does not imply feasibility or optimality. |
| `material measure` | $M_h=\sum_e|e|\theta_e$ | Always report units and target. |
| `normalized material fraction` | $\bar\theta_h=M_h/|\Omega|$ | Never call an unnormalized measure a fraction. |

### 7.2 Claim words

| Word or phrase | Use only when |
| --- | --- |
| `proved` | A displayed argument covers the exact stated assumptions. A passing test is not a proof. |
| `verified` | An independent derivative, invariant, manufactured solution, or same-functional implementation check passes a predefined tolerance. |
| `validated` | Evidence addresses the intended physical/modeling use, not merely another implementation. Current plasticity evidence does not meet this standard. |
| `equivalent` | The fixed-functional assumptions in Section 3.1 hold and a numerical statement reports its norm, scale, state, and tolerance. |
| `exact` | The statement concerns exact arithmetic or exact quadrature for the named integrand. For floating-point output, use `to roundoff` or a numerical tolerance. |
| `converged` | The final run satisfies the declared publication residual gate, correction safeguard, and required mechanics/linear residuals. A cap, stall alone, or solver message alone is insufficient. |
| `local minimizer` | First-order stationarity and an appropriate second-order sufficient condition are checked on the constrained space. |
| `global minimizer` | Convexity plus feasibility or another global certificate applies. |
| `optimization solution` | A fixed problem is defined and feasibility, stationarity, and complementarity pass stated tolerances. No current topology result qualifies. |
| `mesh convergence` | One mathematical quantity is compared under controlled discretization and quadrature refinement with algebraic error below discretization error. |
| `strong scaling` | Problem, derivative route, nonlinear policy, preconditioner/coarse policy, stopping target, and work definition remain fixed. |
| `faster` | Replicated timing under an equal-accuracy contract supports the comparison and uncertainty is reported. |

### 7.3 Safe sentence templates

- `For the same quadrature-defined functional and a branch-stable fixed state,
  element AD and constitutive AD agree in [residual/Hessian-action norm] to
  [tolerance].`
- `The colored AD-HVP route reconstructs the owned rows of the selected Hessian
  under the complete-pattern, noninterference, ghost-coverage, and unique-row-
  ownership assumptions.`
- `The solver returns an approximate first-order stationary point under the
  stated scaled residual; no second-order optimality test is applied.`
- `The Plasticity3D case is a synthetic branch-structured endpoint surrogate
  used to measure derivative placement; it does not represent an incremental
  plastic-history calculation.`
- `The topology run is a sequence of inexact frozen-model updates and is
  reported as a reduced-design endpoint, not as a KKT point.`

### 7.4 Hard prohibitions

The following claims are prohibited unless a later proof or experiment changes
their status and this dictionary is updated.

1. Do not claim a global projection, envelope, or stress-gradient theorem for
   Plasticity2D.
2. Do not claim that either plasticity family implements or validates
   path-consistent incremental Mohr--Coulomb plasticity.
3. Do not call a plasticity active-branch Hessian a semismooth Newton derivative
   or generalized Jacobian without a regularity proof.
4. Do not claim continuity, $C^1$ regularity, semismoothness, or tangent
   definiteness across plastic branch interfaces.
5. Do not interpret a localized plasticity strain image as a physical slip
   surface, collapse mechanism, failure load, or factor of safety.
6. Do not call a nonconvex Ginzburg--Landau, hyperelastic, or plasticity
   endpoint a local or global minimizer from a first-order test alone.
7. Do not claim a common admissible nonlinear path for signed-determinant and
   absolute-determinant hyperelastic implementations.
8. Do not claim fixed-functional FEniCS--JAX derivative equivalence for
   Ginzburg--Landau while their quadrature rules differ.
9. Do not call cross-degree Plasticity3D trends continuum convergence while
   element degree, quadrature, mesh, and stopping accuracy change together.
10. Do not claim three-route equivalence from element-versus-constitutive tests
    alone; colored recovery needs its own canonical-state matrix/HVP checks.
11. Do not call an AD-HVP recovery exact unless the structural pattern,
    interference coloring, ghost coverage, and row ownership conditions are
    verified.
12. Do not call the current topology endpoint an optimal design, converged
    topology, KKT point, or optimization solution.
13. Do not mix material measure and normalized material fraction, and do not
    reuse pre-semantics-version-2 topology artifacts as if they were regenerated
    evidence.
14. Do not compare methods as faster or more scalable when the final accuracy,
    nonlinear policy, or coarse/preconditioner policy differs.
15. Do not infer quadrature adequacy from endpoint energy alone when residual,
    tangent-action, or branch diagnostics differ materially.

## 8. Publication gates implied by this audit

The selected SISC route does not require a global plasticity or topology
convergence theorem. It does require the following gates before the associated
evidence enters the main text.

1. **Smooth-route gate.** **Local mathematics block passed; distributed backend
   block partly passed:** element Taylor/FD, analytic contractions, indefinite
   Ginzburg--Landau, admissible hyperelastic derivatives, manufactured scalar
   rates, the analytic hyperelastic affine patch, and the nonaffine
   hyperelastic manufactured convergence test pass. A controlled
   hyperelastic one-versus-two-rank fixed-state pilot gives exact topology,
   residual, and matrix agreement and roundoff-level action/correction
   differences. Four ranks, the factorized construction matrix, solved
   endpoints, and a clean rerun remain. The paper makes no matched DOLFINx
   claim, so that comparison is removed from the required gate; an optional
   future comparison remains blocked by the local ADIOS2 ABI mismatch.
2. **Plasticity3D branch gate.** **Local constructed-state block passed;
   clean/distributed block open:** the P2 analytic state exercises all five
   production labels and passes three-route tangent-action comparison.
   EXP-MC-001 adds strict per-branch margins, denominators, principal gaps,
   finite derivatives, rotations, repeated spectra, and two-sided interface
   probes. Promotion requires a clean-commit rerun; multi-rank FE ownership and
   route checks remain separate gates.
3. **Switch gate.** **Scoped away from switches:** two-sided diagnostics are
   recorded, including order-one selected-Hessian changes, but no generalized
   regularity is established. Retain branch-interior wording and allow no
   ordinary Hessian claim at a switch.
4. **Plasticity2D finite-AD gate.** **Passed for the selected apex convention:**
   the repeated-principal-stress NaN is removed and focused regressions cover
   the apex, regular line branch, rotation, and nearby states. The
   projection/envelope theorem remains unproved, so keep the family synthetic
   and supplementary and make no switch-regularity claim.
5. **Quadrature gate.** **Fixed-state diagnostic complete; solve gate open:**
   energy-only agreement is disproved as an adequacy test by the large P2/P4
   residual and tangent-action changes above. Solve both enriched-rule
   problems, require each own-rule residual gate, and compare endpoints with
   the common 125-point evaluator. Treat every near-switch sample as selected-
   branch AD. A higher-point rule is a reference, not exact truth.
6. **Stopping gate.** **Scalar, Plasticity3D, and Hyperelasticity implementations
   passed; calibration remains open:** the scalar P1 lumped-$L^2$ map passes its
   small solver smoke, the glued-free-space plasticity map passes its P1
   two-rank smoke, and the two-end-constrained hyperelasticity map passes P1
   smokes on one and two ranks. Calibrate them across retained meshes/degrees
   and integrate the mechanics map into every applicable route runner before
   regenerating central endpoints. Retain coefficient norms only as
   diagnostics.
7. **Route/cost gate.** **P0 and open:** admit the paired fixed-state map and
   held-out cost model only after clean route equivalence, common endpoint
   accuracy, collective-max timing provenance, and the prespecified coverage
   and uncertainty gates pass. If they fail, replace the selector with a
   bounded finite map or narrow the central claim.
8. **Scaling gate.** **Conditional P1:** retain scaling only under one fixed
   problem, derivative route, nonlinear policy, preconditioner policy, and
   endpoint-accuracy contract. Removing scaling does not block the selected
   route/cost contribution.
9. **Topology gate.** Keep topology supplementary unless one fixed constrained
   problem, exact reduced gradient, feasibility, projected stationarity,
   complementarity, baseline comparison, and rank consistency all pass. The
   local reciprocal tangency proof alone is insufficient.
10. **Language gate.** Audit every occurrence of `minimizer`, `equivalent`,
   `stress`, `tangent`, `converged`, `validation`, and `optimization solution`
   against Section 7 before release.

## 9. Checks executed for this audit

The focused check

```text
./.venv/bin/python -m pytest -q \
  tests/test_slope_stability_jax.py \
  tests/test_paper_derivative_verification_runner.py \
  tests/test_topology_volume_semantics.py \
  tests/test_plasticity3d_quadrature.py
```

was rerun after the apex and branch-diagnostic changes. The Plasticity2D file
contains nine passing focused tests; the derivative pilot additionally checks
five smooth element cases and P1/P2/P4 branch-interior plasticity elements.
Separate focused suites cover the quadrature registry and Riesz-metric solver
contract. The pilot artifacts are deliberately labeled dirty-worktree
diagnostics, not publication evidence. `git diff --check` and the full cited
path audit must be rerun after the remaining implementation work before this
document is frozen.

## 10. Repository anchors and audit boundary

The principal inspected anchors are:

- common discrete notation and derivative statements:
  `paper/sections/methodology.tex:3-168`;
- solver-policy and stopping limitations:
  `paper/sections/methodology.tex:170-285`;
- benchmark definitions: `paper/sections/benchmarks.tex:101-1495`;
- manuscript limitations: `paper/sections/discussion.tex:10-67` and
  `paper/sections/conclusion.tex:20-30`;
- $p$-Laplace, Ginzburg--Landau, and hyperelastic kernels under
  `src/problems/plaplace/`, `src/problems/ginzburg_landau/`, and
  `src/problems/hyperelasticity/`;
- plasticity kernels:
  `src/problems/slope_stability/jax/jax_energy.py` and
  `src/problems/slope_stability_3d/jax/jax_energy_3d.py`;
- topology kernels and controller:
  `src/problems/topology/jax/jax_energy.py`,
  `src/problems/topology/jax/parallel_support.py`, and
  `src/problems/topology/jax/solve_topopt_parallel.py`;
- focused derivative and semantics tests under
  `tests/test_slope_stability_3d.py`,
  `tests/test_paper_derivative_verification_runner.py`,
  `tests/test_reordered_element_base.py`, and
  `tests/test_topology_volume_semantics.py`.

This audit does not prove continuum well-posedness beyond standard cited
settings, independently revalidate every literature source, certify historical
benchmark artifacts, or establish any global plasticity projection theorem.
Those boundaries are deliberate. The central publishable result is the
conditional derivative-placement statement and its reproducible numerical
verification, not a broader constitutive or optimization claim.
