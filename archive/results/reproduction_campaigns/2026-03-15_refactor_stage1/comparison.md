# Baseline Comparison

Campaign: `2026-03-15_refactor_stage1`

This checkpoint establishes the post-structure, post-helper-extraction smoke baseline for the maintained solver paths.

| Case | Status | Message | Notes |
| --- | --- | --- | --- |
| `gl_fenics_custom_l5_np2` | pass | Converged (energy, step, gradient) | dofs=4225, iters=7, np=2 |
| `gl_jax_petsc_l5_np2` | pass | Converged (energy, step, gradient) | dofs=3969, iters=12, np=2 |
| `he_fenics_custom_l1_steps1_np2` | pass | Energy change converged | dofs=2187, np=2 |
| `he_jax_l1_steps1_np1` | pass | Converged (energy, step, gradient) | dofs=2133, np=1 |
| `he_jax_petsc_l1_steps1_np2` | pass | Converged (energy, step, gradient) | dofs=2133, iters=9, np=2 |
| `plaplace_fenics_custom_l5_np2` | pass | Converged (energy, step, gradient) | dofs=3201, iters=5, np=2 |
| `plaplace_jax_l5_np1` | pass | Stopping condition for f is satisfied | dofs=2945, iters=6, np=1 |
| `plaplace_jax_petsc_l5_np2` | pass | Converged (energy, step, gradient) | dofs=3201, iters=6, np=2 |
| `topology_parallel_nx24_ny12_np2` | pass | max_outer_iterations | outer=3, np=2 |
| `topology_serial_nx24_ny12_np1` | pass | max_outer_iterations | outer=3, np=1 |

Notes:
- The pure-JAX p-Laplace smoke passed after filtering unused mesh metadata in the shared `EnergyDerivator` parameter forwarding path.
- The serial topology smoke uses `--design_maxit 60` so the shortened 3-outer-iteration checkpoint remains a successful smoke rather than stopping on the design inner cap.
