# Convergence Metrics For Publication Runs

Publication comparisons must use a norm attached to the discrete function
space, rather than an unscaled coefficient-vector norm. The implementation is
in `src/core/petsc/metrics.py`, and the PETSc Newton and gradient-descent
solvers accept it through `convergence_metric`.

## Discrete Contract

Let `W` be a symmetric positive-definite Riesz operator on the constrained
free-variable space. For a state vector `u`, residual functional `g`, and
accepted correction `du`, the solver records

\[
  \lVert u\rVert_W=(u^T W u)^{1/2},\qquad
  \lVert g\rVert_{W^{-1}}=(g^T W^{-1}g)^{1/2},
\]

and

\[
  s_{\mathrm{rel}}=
  \frac{\lVert du\rVert_W}
       {\max\{\lVert u\rVert_W,u_{\mathrm{scale}}\}}.
\]

This correction formula is the `metric_current_state` mode used automatically
for non-Euclidean metrics. The `auto` mode resolves an `EuclideanMetric` to
`legacy_coefficient`, preserving the historical raw coefficient-step norm and
the denominator `max(1, ||u_previous||_2)`. An explicit
`convergence_correction_mode` can select either behavior; the resolved mode is
recorded in history and terminal output.

The positive physical scale `u_scale` is passed as
`convergence_state_scale`. An experiment card must define its units and
value. Relative residuals use the initial dual residual as denominator. The
absolute dual residual, initial-relative residual, Riesz-solve diagnostics,
coefficient `l2` norm, state norm, correction norm, and relative correction
are all retained.

The coefficient norm remains part of the record because it is useful for
debugging. It is not a valid cross-mesh or cross-degree stopping measure.

## Available Implementations

- `EuclideanMetric` reproduces the legacy coefficient-vector metric. It is the
  backward-compatible default, not an automatic publication choice.
- `DiagonalRieszMetric` represents a strictly positive diagonal map, for
  example a verified lumped mass operator. Both primal and dual norms are
  evaluated exactly, with global MPI reductions.
- `MatrixRieszMetric` represents an assembled SPD PETSc matrix. Its dual norm
  uses a configured KSP solve and records the convergence reason, iteration
  count, PETSc residual, independently recomputed true residual, and relative
  true residual. It can impose an independent true-relative-residual gate. A
  failed solve or failed gate invalidates the norm and raises an error.
- `certify_spd_by_cholesky` checks symmetry at a scale-aware entry tolerance,
  performs a symmetric direct factorization, and requires inertia `(0, 0, n)`
  on the `n`-row free space. Failure to obtain that certificate is an error,
  not permission to assume positive definiteness.

The diagonal implementation rejects a nonfinite or nonpositive local weight.
The matrix implementation checks symmetry when constructed and rejects a
negative quadratic form. Its generic constructor does not run an expensive
factorization automatically: the problem integration must supply a numerical
certificate such as `certify_spd_by_cholesky`, or an equally explicit proof
and verification contract. A symmetry check alone does not establish positive
definiteness.

Global PETSc options are ignored by `MatrixRieszMetric` by default. If
`set_from_options=True` is requested, options are isolated behind a unique
metric-specific prefix (or an explicitly supplied prefix). Both requested and
effective KSP tolerances, plus the prefix, are recorded; an override therefore
cannot be silent.

## Scalar P1 Lumped-$L^2$ Metric

The shared JAX/PETSc p-Laplace and Ginzburg--Landau driver exposes
`--convergence-metric lumped_l2`. It constructs the diagonal row-sum P1 mass
map on the exact constrained free space,

\[
  (M_{\mathrm L})_{ii}=\sum_{e\ni i}\frac{|e|}{3},
\]

and reorders the positive weights with the same free-DOF permutation as the
state. The primal correction norm is
$\sqrt{u^T M_{\mathrm L}u}$ and the dual residual is
$\sqrt{g^T M_{\mathrm L}^{-1}g}$. Both are evaluated exactly up to the MPI
reductions; no iterative norm solve is involved.

If no explicit state scale is supplied, the driver uses the lumped-$L^2$ norm
of a unit scalar field on the free nodes,
$u_{\mathrm{scale}}=\sqrt{\sum_i(M_{\mathrm L})_{ii}}$. This remains positive
and has the correct units even when the nonlinear initial iterate is zero. The
record includes hashes of nodes, triangles, element areas, free DOFs, and the
free permutation, together with weight extrema and total free weight.

`coefficient_l2` remains the backward-compatible default. A publication
campaign must select and calibrate `lumped_l2` explicitly; implementation does
not retrospectively make historical coefficient-norm endpoints comparable
across meshes.

## HyperElasticity Reference Elastic-Energy Metric

The JAX/PETSc HyperElasticity element and SFD paths expose
`reference_elastic_energy` in addition to the legacy `coefficient_l2` default.
Before the first load-step boundary update, the solver verifies bitwise that
the free state is the undeformed map $y(X)=X$, assembles the exact discrete
Neo-Hookean Hessian in the solver's constrained ordering, and copies that
matrix as $K_{\rm ref}$. Both beam end faces have essential constraints, so the
free-space matrix has no rigid modes. A scale-aware symmetry check and a
Cholesky/LDL-transpose inertia certificate must report no negative or zero
pivots before the metric is admitted.

The state variable in this solver is the deformation map rather than the
displacement. The implemented norms are therefore

\[
  \lVert y\rVert_{K_{\rm ref}}=(y^T K_{\rm ref}y)^{1/2},
  \qquad
  \lVert g\rVert_{K_{\rm ref}^{-1}}
  =(g^T K_{\rm ref}^{-1}g)^{1/2}.
\]

Using the absolute map $y$ in the primal norm is dimensionally valid because
$K_{\rm ref}$ is the tangent with respect to the same deformation-map
coefficients and the reference coordinate system is fixed. Corrections use
$\lVert y_{k+1}-y_k\rVert_{K_{\rm ref}}$, so their value is unaffected by the
affine $y=X+u$ representation. If no explicit scale is supplied, the solver
uses the initial $\lVert y(X)=X\rVert_{K_{\rm ref}}$. The same fixed map and
scale are used at every load step; they are not rebuilt at the current
deformation.

The inverse-Riesz KSP is independent of the Newton-equation KSP and ignores
global PETSc overrides. Every dual-norm evaluation records its requested and
effective tolerances, convergence reason, recursive residual, independently
recomputed true residual, right-hand-side norm, and achieved true relative
residual. Exceeding `--riesz-true-residual-rtol` is a hard error. Each load-step
record contains the full metric provenance and certificate, while
`nonlinear_convergence.terminal` repeats the final step's stopping record.
The provenance includes material values, boundary treatment, element and
ordering choices, exact input hashes or rank-local payload hashes, and the
reference-state ownership ranges.

The direct solver selection is, for example:

```bash
mpiexec -n 2 ./.venv/bin/python \
  src/problems/hyperelasticity/jax_petsc/solve_HE_dof.py \
  --level 1 --steps 1 --total_steps 24 \
  --assembly_mode element --problem_build_mode rank_local \
  --mesh_source procedural --element_reorder_mode block_xyz \
  --local_hessian_mode element --distribution_strategy overlap_p2p \
  --assembly_backend coo_local \
  --convergence-metric reference_elastic_energy \
  --riesz-ksp-rtol 1e-10 --riesz-true-residual-rtol 1e-8
```

Focused one- and two-rank P1 smokes certify the operator and norm-solve
contract. They do not calibrate the nonlinear tolerance, establish
cross-mesh equivalence, or show that the default Jacobi norm preconditioner is
efficient. The small slender-beam smoke required 1,283 and 1,252 CG iterations
on one and two ranks, respectively, so the HyperElasticity-specific safety cap
is 5,000 rather than 1,000. That cap is not an endorsed performance policy.
Publication runs still require the calibration steps below. The FEniCS runner
does not yet expose this exact record contract and rejects this metric
selection.

## Plasticity3D Reference Elastic-Energy Metric

The maintained Plasticity3D DOF solver and backend-mix case runner expose two
explicit choices through `--convergence-metric`:

- `coefficient_l2` is the default and preserves the historical coefficient
  norm, raw-step correction, previous-state normalization, and default
  coefficient state scale `1.0`;
- `reference_elastic_energy` reuses the elastic tangent assembled at zero
  displacement on the reordered, constrained free space.

The publication-oriented path is intentionally restricted to the canonical
`glued_bottom` constraint. It verifies that the matrix dimensions equal the
number of free DOFs, checks numerical symmetry relative to the matrix infinity
norm, and requires a Cholesky/LDL-transpose inertia certificate with no
negative or zero pivots. The default factor backend is MUMPS; inability to
produce the inertia is a hard setup failure.

The backend-mix integration is available for the `local`,
`local_constitutiveAD`, and `local_sfd` assembly routes when the nonlinear
solver is driven by the repository-local Newton implementation. It captures
the elastic operator and its full HDF5/free-space provenance before the local
problem payload is released. Source nonlinear solves are rejected because
they do not expose this stopping contract. The backend-mix CLI keeps
`coefficient_l2` as its legacy default, so every publication campaign must
request `reference_elastic_energy` explicitly.

For this choice,

\[
  \lVert u\rVert_{K_{\rm el}} = (u^T K_{\rm el}u)^{1/2},
  \qquad
  \lVert g\rVert_{K_{\rm el}^{-1}}
  = (g^T K_{\rm el}^{-1}g)^{1/2}.
\]

Both absolute quantities have units equal to the square root of the discrete
reference elastic-energy unit. The initial-relative dual residual and relative
correction are dimensionless. If `--convergence-state-scale` is omitted, the
solver uses the initial nonlinear iterate's `K_el` primal norm, which has the
correct units. A zero initial norm is rejected; use a nonzero reference state
or provide a positive, physically justified scale explicitly.

The norm-solve controls are independent of the Newton-equation KSP controls:

```text
--riesz-ksp-type cg
--riesz-pc-type jacobi
--riesz-ksp-rtol 1e-10
--riesz-ksp-atol 1e-14
--riesz-ksp-max-it 1000
--riesz-true-residual-rtol 1e-8
--riesz-spd-factor-solver-type mumps
--riesz-symmetry-tol 1e-12
```

These CLI values cannot be silently replaced by global PETSc options. The JSON
field `nonlinear_convergence` records the operator provenance, input-dataset
and free-space/permutation hashes, tangent route, material ranges, free-space
and ordering contract, SPD certificate, state-scale source,
dimensionful absolute values, dimensionless relative values, requested Riesz
KSP controls, convergence reason, recursive residual, recomputed true
residual, right-hand-side norm, and achieved true relative residual.
Result-level residual and state fields are recomputed at the terminal state,
including maximum-iteration and time-capped exits; per-iteration history keeps
the residual evaluated at that iteration's pre-step state.

An illustrative small-run selection is:

```bash
mpiexec -n 1 ./.venv/bin/python \
  src/problems/slope_stability_3d/jax_petsc/solve_slope_stability_3d_dof.py \
  --mesh_name hetero_ssr_L1 --elem_degree 1 \
  --elastic_initial_guess \
  --convergence-metric reference_elastic_energy \
  --riesz-ksp-rtol 1e-10 --riesz-true-residual-rtol 1e-8
```

The equivalent backend-mix selection is:

```bash
mpiexec -n 2 ./.venv/bin/python \
  experiments/runners/run_plasticity3d_backend_mix_case.py \
  --assembly-backend local_constitutiveAD --solver-backend local \
  --out-dir artifacts/raw_results/example_runs/p3d_riesz \
  --output-json artifacts/raw_results/example_runs/p3d_riesz/output.json \
  --mesh-name hetero_ssr_L1 --elem-degree 1 \
  --quadrature-rule tetra_1point --constraint-variant glued_bottom \
  --convergence-mode all --grad-stop-tol 1e-3 --grad-stop-rtol 1e-3 \
  --convergence-metric reference_elastic_energy \
  --riesz-ksp-rtol 1e-10 --riesz-true-residual-rtol 1e-8
```

For this runner, `status=completed` under the reference metric requires both
the nonlinear solver's convergence message and a fresh terminal dual-residual
gate. `nonlinear_convergence.last_riesz_solve.rhs_norm` therefore corresponds
to the reported terminal coefficient-gradient diagnostic, including capped
or otherwise unsuccessful exits.

This implementation supplies the missing norm definition and audit trail. It
does not by itself calibrate nonlinear tolerances across `P1`, `P2`, and `P4`,
nor show that Jacobi is an efficient norm-solve preconditioner. Those choices
still require the stopping-sensitivity, cross-mesh, and MPI checks below.

## Problem-Specific Choices

The experiment protocol must construct and identify one map on the exact free
space used by the solve:

- scalar fields: the implemented P1 lumped-mass map, or a separately stated
  reference `H1` map;
- displacement mechanics: a reference elastic-energy map after essential
  constraints and rigid-mode removal;
- topology design variables: a design-space mass map in the chosen physical
  variable representation.

Record the assembly code, coefficients, boundary treatment, input hashes, and
whether the map is exact, lumped, or approximate. Do not mix latent-variable
and physical-density norms without the explicit design-map derivative.

## Solver Behavior

The selected dual norm controls the gradient/residual gate. The selected
primal norm controls the correction gate. Energy change and correction are
secondary gates; a publication result labeled successful also needs the
preregistered residual gate.

Newton and gradient descent recompute their result-level residual and state
norm at the terminal state, and retain the initial absolute dual residual used
for normalization. If an exceptional path has no valid initial evaluation,
the relative residual and relative target remain undefined rather than being
fabricated from the endpoint.

The current Newton trust-region and line-search implementations retain their
legacy Euclidean coefficient geometry for trial-step construction, trust
radius, and descent-direction normalization. This is deliberate: changing a
stopping metric must not silently change the algorithm being compared. The
history therefore reports both algorithmic coefficient-space step quantities
and the Riesz-scaled accepted correction. A study of a Riesz-metric trust
region is a different algorithm and needs a separate solver contract.

## Required Calibration

Before a metric is used for a headline run:

1. verify that `W` is defined on the constrained free space and has the same
   ordering and ownership as the state and residual vectors;
2. verify one-rank and distributed primal/dual norm parity at a stored state;
3. for an iterative inverse, make the true Riesz-solve residual small enough
   that tightening it cannot change the nonlinear stopping decision;
4. compare the chosen nonlinear tolerance with a tenfold-tighter solve on the
   same discretization;
5. record the metric name, provenance, state scale, and all norm-solve
   diagnostics in the versioned publication run record.

Passing these checks establishes an algebraic stopping contract. It does not
establish discretization convergence, global optimality, or physical validity.
