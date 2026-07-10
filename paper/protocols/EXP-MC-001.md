# EXP-MC-001: Plasticity3D Material-Point Branch Verification

## Research Question

Does the production scalar Plasticity3D Mohr--Coulomb endpoint potential
select and differentiate all five return branches consistently at controlled
material points, remain finite at repeated and nearly repeated principal
values under its documented eigenvalue tie break, and transform objectively
under rotations away from branch switches and spectral degeneracies?

This is a supplementary constitutive-implementation diagnostic. It is not a
validation of path-consistent incremental plasticity and it does not establish
generalized differentiability at branch interfaces.

## Objects Under Test

- production scalar:
  `src.problems.slope_stability_3d.jax.jax_energy_3d.mc_potential_density_3d`;
- production derivatives: `jax.grad` and `jax.hessian` of that scalar with
  respect to engineering strain `[xx, yy, zz, xy, yz, xz]`;
- separate diagnostic implementation:
  NumPy engineering-tensor conversion, symmetric eigendecomposition, branch
  predicates, candidate-energy formulas, tensor rotations, stress conversion,
  and tangent-action covariance.

The NumPy branch and candidate-energy formulas transcribe the same
Mohr--Coulomb algebra as the production implementation. They can expose
wiring, ordering, branch-selection, or rotation errors, but they are **not** an
independently derived constitutive reference.

## Frozen Dimensionless Material

| parameter | value |
| --- | ---: |
| cohesion term `c_bar` | `1` |
| `sin(phi)` | `0.5` |
| shear modulus | `10` |
| bulk modulus | `20` |
| Lamé parameter | `13.3333333333333` |
| eigenvalue tie-break scale | `1e-15` |

The normalization keeps branch coordinates and finite-difference scales
well-conditioned. This campaign tests constitutive algebra rather than a
particular physical unit system.

## Frozen Case Matrix

### Strict branch interiors

The principal-strain anchors below are ordered largest to smallest. They were
selected once by a seeded dimensionless search and then frozen in the runner.
Every invocation must independently reclassify each row and pass a normalized
active-branch-margin gate.

| expected branch | principal-strain anchor |
| --- | --- |
| elastic | `[-0.13631082, -0.16680545, -0.19711281]` |
| shear | `[0.05755777, -0.03796704, -0.06961258]` |
| left edge | `[0.06851800, 0.05688190, -0.06930851]` |
| right edge | `[0.39253952, -0.13488312, -0.19385268]` |
| apex | `[0.37517304, 0.31871759, 0.14611402]` |

At every interior the runner evaluates energy, gradient, Hessian, Hessian
symmetry, the NumPy-selected candidate energy, a centered energy directional
derivative, and a centered gradient/Hessian-vector product. The complete step
sequence is `3e-5, 1e-5, 3e-6, 1e-6, 3e-7`; the prespecified gate is `1e-6`.
Both centered states must retain the central branch at every recorded step.

### Nonsingular adjacent interfaces

The production partition has the following adjacency graph away from repeated
principal values:

1. elastic--shear;
2. shear--left edge;
3. shear--right edge;
4. left edge--apex;
5. right edge--apex.

For each edge, bisection locates the branch change on the straight segment
between the corresponding frozen interiors. Both sides are evaluated at
normalized segment offsets `1e-2`, `1e-4`, and `1e-6`. The intended labels,
finite energy/gradient/Hessian, symmetry defect, active predicate margin,
principal gap, denominator margin, and tie-break scale are retained.

These deliberately near-interface pairs are **not** subjected to centered
derivative gates. Their paired energy, gradient, and Hessian differences are
descriptive diagnostics only. No convergence of those differences is used to
claim classical or generalized differentiability at a switch.

### Repeated and nearly repeated principal values

Seven frozen cases cover exact hydrostatic spectra, exact double eigenvalues,
`1e-10` double-eigenvalue gaps, and a `1e-10` near-hydrostatic spacing. The
campaign requires finite energy, gradient, and Hessian and a symmetric Hessian
under the production `1e-15 * diag(0,1,2)` tie break.

Derivative-convergence and rotation-covariance gates are intentionally not
applied to these cases. The result demonstrates finite implementation output
for the frozen coordinate-dependent regularization; it does not prove a
coordinate-free derivative at an eigenvalue multiplicity.

### Random rotations

Three seeded proper orthogonal rotations are applied to every strict branch
interior. NumPy constructs `R E R^T` and the rotated strain direction. The
checks are:

- scalar energy invariance;
- branch-label invariance;
- stress covariance, with the engineering-coordinate energy gradient mapped
  to a symmetric stress tensor;
- tangent-action covariance for a seeded symmetric strain direction;
- rotation orthogonality and determinant `+1`.

Only distinct-principal-value interiors enter this gate. This separation is
required because the coordinate-axis tie break is not an objective
regularization at a repeated spectrum.

## Prespecified Gates

| metric | gate |
| --- | ---: |
| coverage | exactly one strict interior for each of five branches |
| normalized active branch margin | at least `1e-3` |
| NumPy selected-energy transcription error | at most `1e-12` relative |
| Hessian symmetry defect | at most `1e-10` |
| centered directional-energy scaled error at `h=1e-6` | at most `1e-7` |
| centered HVP scaled error at `h=1e-6` | at most `1e-7` |
| branch stability over every centered-FD step | required |
| two-sided interface coverage | all five adjacency edges, three offsets each |
| finite repeated-spectrum outputs | all seven cases |
| rotation energy/stress/tangent scaled error | at most `1e-9` |
| near-zero rotated tangent action | alternatively at most `1e-9` absolute |
| rotation orthogonality and determinant defect | at most `1e-12` |
| serialization | strict JSON; `NaN` and infinities forbidden |

The centered errors use the frozen dimensionless normalization with an
absolute scale floor of one. That convention is material for the apex branch:
its selected scalar is affine in the strain trace and its exact Hessian action
is zero, so a pure relative finite-difference error would be undefined and
would amplify roundoff divided by `h`.

## Reproduction Command

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
./.venv/bin/python \
  experiments/runners/run_plasticity3d_material_point_verification.py \
  --output artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-MC-001/material_point_verification.json \
  --report artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-MC-001/pilot_report.md \
  --run-record artifacts/reproduction/paper_revision_2026_07_10/pilots/EXP-MC-001/run_record.json \
  --run-kind pilot \
  --pilot-dirty-override \
  --pilot-override-reason "EXP-MC-001 implementation verification before clean revision commit"
```

This command is a small serial CPU calculation and does not import or invoke
DOLFINx, PETSc, MPI, or Slurm. It writes both the detailed
`plasticity3d_material_point_verification` schema and a terminal record that is
validated against `fenics-nonlinear-energies.publication-run-record` version
`1`; both serializers reject `NaN` and infinities.

For promotion after committing the revision, replace the three pilot/override
options with `--run-kind publication`. The publication preflight then requires
an empty `git status --porcelain` and rejects a dirty tree.

## Current Local Pilot

The 2026-07-10 dirty-worktree pilot passed every frozen gate:

| quantity | result |
| --- | ---: |
| strict branch interiors | `5/5` |
| two-sided interface pairs | `15/15` |
| seeded rotation checks | `15/15` |
| repeated/nearly repeated cases finite | `7/7` |
| minimum normalized active branch margin | `1.2545003889e-1` |
| maximum Hessian symmetry defect | `8.6789166847e-17` |
| maximum centered energy error at gate | `1.0591322264e-10` |
| maximum centered HVP error at gate | `1.5383701491e-9` |
| maximum NumPy energy-transcription error | `2.1795665085e-15` |
| maximum rotation energy-invariance error | `1.8193468188e-14` |
| maximum rotation stress-covariance error | `3.4067152706e-14` |
| maximum rotation tangent-action covariance error | `9.3328964512e-15` |

Because the repository was dirty, these values are pilot evidence rather than
publication-grade immutable evidence. A clean-commit rerun can promote the
record without changing the prespecified matrix or gates.

## Permitted And Forbidden Claims

Permitted after a clean rerun:

- the production scalar and its JAX derivatives are finite and internally
  consistent at the five tested strict branch interiors;
- all five nonsingular branch adjacencies were approached from both sides with
  the intended predicate labels;
- objectivity checks passed for the tested nondegenerate interiors;
- the explicit tie break produced finite derivatives for the seven tested
  repeated/nearly repeated spectra.

Forbidden:

- differentiability, semismoothness, or a generalized derivative at a branch
  switch;
- objectivity of the coordinate-dependent eigenvalue tie break at repeated
  spectra;
- equivalence to an independent return-mapping implementation;
- path consistency, incremental plasticity correctness, solver convergence,
  mesh convergence, or physical validation.
