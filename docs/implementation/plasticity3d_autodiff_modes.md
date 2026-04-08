# Plasticity3D Autodiff Modes

This note explains the two maintained autodiff tangent paths in
`src/problems/slope_stability_3d/` and how they differ from the earlier
"write one scalar energy on the whole element and let autodiff do the rest"
workflow.

The goal of the newer path is to keep autodiff and keep a scalar constitutive
energy as the source of truth, but avoid taking second derivatives with
respect to the full `P4` element DOF vector when that becomes too expensive.

## Why There Are Two Paths

The original 3D plasticity implementation followed the most direct pattern:

- write one scalar element energy `Pi_e(u_e)`
- use `jax.grad(Pi_e, u_e)` for the local residual
- use `jax.hessian(Pi_e, u_e)` for the local tangent

That is still the cleanest formulation and it remains available as the
`element` autodiff mode.

For higher-order `P4` plasticity, however, the tangent cost is dominated by
autodiff through a large element DOF vector. The maintained code now also keeps
an alternative `constitutive` autodiff mode that moves the differentiation
boundary down to the quadrature-point strain state.

Both modes are maintained. Neither uses hand-coded constitutive tangents.

## Mode 1: Element Autodiff

This is the original workflow.

### Input Model

You write the scalar element energy

$$
\Pi_e(u_e)
= \sum_q w_q\, \psi(\varepsilon_q(u_e), \text{material}_q).
$$

The code then asks JAX for:

- residual: `d Pi_e / d u_e`
- tangent: `d^2 Pi_e / d u_e^2`

### What This Preserves

- the element energy is the one and only source of truth
- residual and tangent are both derived automatically
- no special FEM assembly knowledge is needed beyond the element energy itself

### Cost Profile

This is the most automatic option, but it is also the most expensive second
derivative path for large elements because the Hessian is taken with respect to
all element DOFs.

## Mode 2: Constitutive Autodiff

This is the newer alternative.

### Input Model

You still write a scalar energy density at quadrature points:

$$
\psi(\varepsilon_q, \text{material}_q).
$$

The code then asks JAX for:

- stress: `sigma_q = d psi / d eps_q`
- constitutive tangent: `C_q = d^2 psi / d eps_q^2`

and assembles the element contributions with the usual strain operator:

$$
r_e = \sum_q w_q B_q^T \sigma_q,
\qquad
K_e = \sum_q w_q B_q^T C_q B_q.
$$

### What Changes Relative To The Old Way

The source of truth is still a scalar energy, but it is the quadrature-point
energy density instead of the whole element energy viewed as one giant AD
object.

So the mental model becomes:

- write `psi(eps_q, material_q)`
- let JAX differentiate `psi`
- let the FEM assembly build `B_q^T (.) B_q`

instead of:

- write `Pi_e(u_e)`
- let JAX differentiate the whole element energy with respect to the full DOF
  vector

### What This Preserves

- autodiff remains the derivative mechanism
- no hand-derived constitutive tangent is introduced
- the same scalar constitutive potential remains the source of truth

### Cost Profile

This is usually much cheaper for high-order plasticity because autodiff now
acts on the six engineering-strain components per quadrature point rather than
the full element displacement vector.

## What A New Constitutive Model Needs To Provide

For the maintained `constitutive` path, a new model should be supplied as a
scalar quadrature-point energy density:

- input:
  - local strain state `eps_q`
  - local material/state parameters at the quadrature point
- output:
  - one scalar energy density `psi_q`

From that, the maintained code obtains automatically:

- `sigma_q = grad(psi_q, eps_q)`
- `C_q = hessian(psi_q, eps_q)`

and reuses the same generic element assembly logic.

This means a new model author does not need to hand-code:

- element residuals
- element tangent matrices
- constitutive tangent matrices

The main requirement is that the model is expressible as a differentiable
scalar potential at the quadrature point.

## When The New Path Fits Well

The `constitutive` autodiff path is a good fit when:

- the constitutive law is potential-based
- the local state is naturally described at quadrature points
- second derivatives with respect to the full element DOF vector are too
  expensive

It is less natural for:

- non-potential constitutive updates
- strongly path-dependent return-map style models with separate history updates
- nonsmooth laws where second derivatives of the scalar potential are not a
  practical representation of the tangent

## Current Maintained Surface

The maintained JAX/PETSc solver exposes the choice through:

- CLI flag:
  `--autodiff_tangent_mode element|constitutive`

The current implementation lives mainly in:

- `src/problems/slope_stability_3d/jax/jax_energy_3d.py`
- `src/problems/slope_stability_3d/jax_petsc/reordered_element_assembler.py`
- `src/problems/slope_stability_3d/jax_petsc/solver.py`

## Practical Guidance

- prefer `element` when clarity and the most direct formulation matter more
  than tangent cost
- prefer `constitutive` for expensive `P4` plasticity runs when you want to
  keep autodiff but reduce Hessian assembly time

The problem card includes the benchmark context and a short summary of the two
paths here:

- [Plasticity3D problem card](../problems/Plasticity3D.md)
