# Shared Scalar Checkpoint

Stage 2 shared-scalar extraction checkpoint for the p-Laplace and Ginzburg-Landau FEniCS custom and JAX+PETSc paths.

## Baseline parity

- `plaplace_fenics_custom_l5_np2`: message_match=True, iters_match=True, ksp_match=True, energy_rel_diff=0.000e+00
- `plaplace_jax_petsc_l5_np2`: message_match=True, iters_match=True, ksp_match=True, energy_rel_diff=0.000e+00
- `gl_fenics_custom_l5_np2`: message_match=True, iters_match=True, ksp_match=True, energy_rel_diff=0.000e+00
- `gl_jax_petsc_l5_np2`: message_match=True, iters_match=True, ksp_match=True, energy_rel_diff=0.000e+00

## Wrapper parity

- `plaplace_fenics_custom_wrapper_vs_canonical_l5_np2`: message_match=True, iters_match=True, ksp_match=True, energy_rel_diff=0.000e+00

## Outputs

- `comparison.json` contains the machine-readable summary.
- `json/` contains the canonical and wrapper rerun outputs used for this checkpoint.
