# EXP-DISC-001: Separated Plasticity3D Quadrature, Mesh, and Tolerance Checks

## Status and decision

**Prepared, not submitted.** The independent named-quadrature implementation
and streamed fixed-state evaluator are implemented and locally tested. A
dirty-worktree pilot now covers energy, residual, deterministic Hessian action,
and branch diagnostics; it is not publication evidence. This campaign remains
conditional: if high-order cases are censored or branch changes prevent an
error interpretation, the old nine-case trend is removed from the main paper
and the outcome is described only as endpoint sensitivity.

The managed local fixed-state quadrature diagnostics use the prescribed
normalized trigonometric displacement state with amplitude 0.02 and the
constrained lift applied. That analytic state is a derivative/evaluator test,
explicitly not a nonlinear-solve endpoint. The Karolina endpoint campaign
below is a separate evidence source and cannot be replaced by those local
diagnostics.

## Scientific question

For the retained `P4` Plasticity3D endpoint, are the observed energy and primary
observable changes caused by quadrature, mesh refinement, or algebraic stopping
accuracy? The campaign changes these factors in separate rows and evaluates all
successful saved states with common independently rebuilt quadrature operators.

## Frozen order and cases

Use the following five sequential release stages; do not authorize a later
stage until the preceding stage's run record and scientific gate have been
inspected. Stage 2 deliberately co-submits the two `P4(L1)` quadrature rows as
one paired scientific release unit. Their comparison requires both endpoints,
so neither row is released or interpreted independently. All other stages
contain one row. Every stage has a fresh prepared archive and a separate human
release authorization, and the manifest records its predecessor.

1. **Smoke stage:** `disc_p4l1_q24_smoke_np64`: one Newton step on `P4(L1)`, 24-point rule,
   64 ranks, 15-minute ceiling. This verifies HDF5 generation, rank-local
   independent quadrature, state export, and reference re-evaluation.
2. **Quadrature stage:** `disc_p4l1_q24_np64` and
   `disc_p4l1_q125_np64`: complete `P4(L1)` endpoints with the 24-point and
   positive 125-point Duffy rules, respectively, at 64 ranks. The ceilings are
   one and two hours.
3. **Mesh stage:** `disc_p4l2_q24_np128`: `P4(L2)`, 24-point rule, 128 ranks over two nodes,
   two-hour ceiling.
4. **Mesh-quadrature stage:** `disc_p4l2_q125_np128`: otherwise identical `P4(L2)` endpoint with the
   125-point rule, two-hour ceiling.
5. **Tolerance stage:** `disc_p4l1_q24_tight_np64`: `P4(L1)`, 24-point rule with one-order tighter
   KSP, correction, and gradient tolerances and a 120-iteration cap.

All rows use the glued-bottom problem at strength-reduction factor 1.55, constitutive AD,
rank-local assembly, the same fixed Hypre PMG policy, and the same Armijo/trust
globalization. Every successful state is re-evaluated with independently
constructed 24-point and 125-point geometry gradients, material fields, load,
and energy operators.

Every solve row uses the numerically checked reference-elastic-energy convergence metric
and metric-current-state correction normalization. Its independent norm solve
is GMRES/Hypre with relative tolerance $10^{-10}$, absolute tolerance
$10^{-14}$, cap 1000, and a recomputed $10^{-8}$ true-residual gate; MUMPS
inertia must numerically check positive definiteness of the glued free-space operator. The campaign
executor rejects coefficient stopping, stale endpoint Riesz evidence, or a
completed row that does not pass the residual gate.

## Stored quantities

- exact mesh, degree, named solve rule, rule point count, and generated HDF5
  checksum;
- clean commit, command, modules/environment, rank placement, and Slurm record;
- nonlinear status, every stopping metric, accepted/rejected steps, nonlinear
  and Krylov work, and timing decomposition;
- final coefficient state, energy, external work, maximum displacement, stress
  summaries when implemented, and material branch fractions;
- independently rebuilt full and free residual $\ell_2$/$\ell_\infty$ norms for
  every evaluation rule;
- a fixed, seeded, free-DOF coefficient-space unit direction; full and free
  Hessian-action norms; saved full action arrays; and file/content SHA-256
  hashes;
- common-reference internal energy, external work, total potential, maximum
  displacement, residual vectors, and Hessian actions for both evaluation
  rules;
- branch point counts, point and absolute-weight fractions, the minimum
  normalized active-branch margin, the number of samples below the
  $10^{-8}$ margin gate, minimum raw and normalized principal-value gaps, and
  the minimum normalized constitutive-denominator margin;
- rank-local element/overlap counts and peak rank/node memory.

For the managed local source campaign, every named rule writes the full
residual, deterministic Hessian action, and element-major branch map as a
separate non-object NPY array. The execution plan declares all 12 arrays per
degree before execution. Each nested JSON artifact descriptor records a
canonical staging-relative path, file SHA-256, array-content SHA-256, dtype,
and shape. The managed receipt, companion manifest, and finalization manifest
must bind the same complete path/hash set. Finalization and the independent
admission audit reload every array with pickling disabled, recompute both
hashes, and reject missing files, path traversal, symlink traversal, dtype or
shape drift, and any descriptor/receipt disagreement.

## Prespecified analysis

The fail-closed adjudicator is
`experiments/analysis/analyze_plasticity3d_discretization.py`. It requires all
six reviewed rows, clean common source provenance, successful own-rule stopping,
and complete common 24/125-point residual, action, branch-map, and endpoint
evidence before releasing any discretization interpretation.

1. **Quadrature:** each endpoint must first pass its own-rule free-residual
   stopping gate. On each mesh, compare the 24-point and 125-point solve states
   in the declared Riesz norm and compare both endpoints under the common
   125-point evaluator. The initial reporting gate is less than 0.1% relative
   change in reference total potential and every principal scalar observable.
   Residual-vector and Hessian-action effect sizes are always reported; a small
   energy difference alone is never accepted as evidence of quadrature
   adequacy.
2. **Mesh:** compare `P4(L1)` and `P4(L2)` only after each mesh has a verified
   quadrature policy. Project states to a declared common space before a state
   error is quoted.
3. **Tolerance:** the tight `P4(L1)` endpoint must differ from the production
   endpoint by materially less than the estimated quadrature/mesh effect. If it
   does not, no discretization trend is reported.
4. **Branches:** report branch fractions and branch-map differences only after
   evaluating both solve endpoints at the same 125-point locations. Fractions
   sampled by two different rules are diagnostic distributions, not pointwise
   branch-map differences. Any sample at or below the $10^{-8}$ normalized
   active-margin gate is flagged, and its Hessian is described only as
   selected-branch AD. If a mesh or quadrature change selects a different
   nonconvex branch, call the result an endpoint-sensitivity study, not
   convergence.
5. A timeout or memory failure is a censored observation. Do not silently fall
   back to a different rule, mesh, rank count, or solver policy.

The resource ceiling is 13.25 node-hours. Outputs are isolated below the common
Karolina revision root and include the solve state plus
`quadrature_reference.json` for every successful row.

## Terminal decisions

- **VERIFIED POLICY:** both enriched-rule endpoints pass their own residual
  gates, consecutive enriched rules pass the 0.1% common-evaluator gate, and
  the tolerance row is negligible relative to discretization effects. Energy,
  residual/action, and branch diagnostics are all disclosed.
- **ENDPOINT SENSITIVITY:** valid runs select different branches or do not show
  consistent reduction; report observations without convergence language.
- **REMOVE STUDY:** high-order rows are infeasible or evidence is insufficient;
  delete the historical confounded trend from the main manuscript.
- **INVALID:** factors changed simultaneously, common evaluation failed, or
  stopping/provenance differs.
