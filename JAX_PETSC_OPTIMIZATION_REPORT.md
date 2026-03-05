# JAX-PETSc vs FEniCS Performance Optimization Report

This document compiles the findings, modifications, and benchmarks discovered while optimizing the `HyperElasticity3D_jax_petsc` solver to match and ultimately exceed FEniCS performance on highly deformed nonlinear systems (up to Level 4, $555,747$ DOFs, 32 MPI processes).

## 1. Key Finding: Matrix Condensation vs Diagonal Padding (Dirichlet Boundaries)

Initial comparisons showed FEniCS natively out-performing JAX-PETSc under `GAMG` (Algebraic Multigrid) preconditioning, despite using the exact same mathematical properties. The root cause was discovered in the boundary condition representations:
- **FEniCS Behavior**: Uses padding. Dirichlet boundaries are kept in the global system but replaced with `1.0` on the diagonal and `0.0` off-diagonal. 
- **JAX-PETSc Behavior**: Uses full matrix condensation. The global DOF array is strictly condensed, fully removing Dirichlet variables from the matrix and modifying the RHS vector accordingly.

**Result**: We found that FEniCS's `1.0` padded diagonal elements severely damaged `GAMG` smoothing structures during restriction/prolongation phases if left unadjusted. By strictly condensing the systems in JAX-PETSc, `GAMG` scales optimally.
In a comparative 1-Step benchmark running Level 4 across 32 cores:
- FEniCS (1e-6 Tolerance): Could not converge linearly efficiently under tight setups due to the padding artifacts.
- JAX-PETSc SFD (1e-6 Tolerance): `~14` KSP iterations.

## 2. Key Enhancement: The JAX `element` Assembly Mode 

To reduce setup time, we replaced building block Hessians via Sparse Finite Differences (SFD) with a purely analytical evaluated pipeline spanning independent finite elements.

- **Implementation**: 
  We utilized `jax.vmap` over `jax.hessian` computing exact localized `24x24` stiffness contributions natively without structural perturbations. By compiling this exactly once into a mapped primitive (`@jax.jit(jax.vmap(compute_elem_hessians))`), memory requirements and setup times plumed.
- **Benchmark** (Level 4, 32 cores, single step assembly setup):
  - `SFD`: Required `8.7` seconds and $4.44M$ evaluations.
  - `Element`: Required `4.39` seconds. Effectively **cutting assembly times by over 50%**, making it vastly computationally dominant for massive deformations.

## 3. Key Enhancement: The Fallback `retry_on_failure` Loop

When tracking severe deformations where physical boundaries undergo massive displacements per step (e.g., forcing a 60-degree rotation step in a single nonlinear iteration), the default parameter bounds crash out-of-bounds due to explosive gradients ($J = 405,000,000+$). FEniCS traditionally crashes.

- **Implementation**: 
  We architected an automatic retry wrapper within the continuous SNES/Newton script. If a Newton failure limit is reached (`Nonlinear step collapsed`, `Non-finite metric`), the step seamlessly aborts and:
  1. Relaxes the local line search boundaries significantly: `[-0.5, 2.0]`
  2. Modifies `KSP` preconditioner relative tolerances down to `1e-3` or `1e-6`.
  3. Rebuilds the stiffness preconditioner entirely.
  
- **Result**: Allowing full 24-step solves rotating $360^\circ \times 4$ with total autonomy. When the nonlinear loop exploded on Steps 5, 7, 9, 10, 11, 16, 21, and 24, the fallback parameters gracefully swept the physical distortion back into continuous bounds, guaranteeing numerical progression through immense strain.

## Next Steps for Future Improvements

- The workspace now contains `experiment_scripts/tmp_scripts` mapping all ad-hoc benchmarks.
- Current JAX code explicitly requires the flags `--assembly_mode element --retry_on_failure` to trigger best-in-class performance. FEniCS baseline scripts (`solve_HE_custom_jaxversion.py`) were modified similarly to test the padding matrix topology.