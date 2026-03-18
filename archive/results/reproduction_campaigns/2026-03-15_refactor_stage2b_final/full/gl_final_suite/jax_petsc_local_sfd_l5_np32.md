# jax_petsc_local_sfd_l5_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `3969` |
| Setup time [s] | `0.975` |
| Total solve time [s] | `0.589` |
| Total Newton iterations | `8` |
| Total linear iterations | `38` |
| Total assembly time [s] | `0.192` |
| Total PC init time [s] | `0.185` |
| Total KSP solve time [s] | `0.112` |
| Total line-search time [s] | `0.094` |
| Final energy | `0.346231` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l5_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l5_np32.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `gmres` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.001` |
| `ksp_max_it` | `200` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.001` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `sfd_local` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.589 | 8 | 38 | 0.346231 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.589` |
| Newton iterations | `8` |
| Linear iterations | `38` |
| Energy | `0.346231` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.566910 | 0.098560 | 0.014806 | nan | -0.256431 | 11 | yes | 19 | 0.188 | 0.024 | 0.028 | 0.012 | 1.000000 | nan |
| 2 | 0.512218 | 0.054692 | 0.016156 | nan | 0.309746 | 6 | yes | 19 | 0.001 | 0.023 | 0.017 | 0.012 | 1.000000 | nan |
| 3 | 0.450718 | 0.061501 | 0.008997 | nan | -0.290453 | 5 | yes | 19 | 0.001 | 0.021 | 0.015 | 0.010 | 1.000000 | nan |
| 4 | 0.406775 | 0.043942 | 0.013905 | nan | 0.052132 | 6 | yes | 19 | 0.001 | 0.020 | 0.015 | 0.011 | 1.000000 | nan |
| 5 | 0.364007 | 0.042768 | 0.013842 | nan | 0.301982 | 4 | yes | 19 | 0.000 | 0.023 | 0.011 | 0.013 | 1.000000 | nan |
| 6 | 0.346551 | 0.017456 | 0.009534 | nan | 0.959763 | 2 | yes | 19 | 0.000 | 0.023 | 0.010 | 0.013 | 1.000000 | nan |
| 7 | 0.346232 | 0.000319 | 0.001153 | nan | 1.034005 | 2 | yes | 19 | 0.001 | 0.029 | 0.010 | 0.013 | 1.000000 | nan |
| 8 | 0.346231 | 0.000000 | 0.000017 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.001 | 0.024 | 0.007 | 0.011 | 1.000000 | nan |
