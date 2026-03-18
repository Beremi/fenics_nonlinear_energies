# jax_petsc_element_l5_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `2945` |
| Setup time [s] | `0.337` |
| Total solve time [s] | `0.239` |
| Total Newton iterations | `7` |
| Total linear iterations | `8` |
| Total assembly time [s] | `0.003` |
| Total PC init time [s] | `0.155` |
| Total KSP solve time [s] | `0.035` |
| Total line-search time [s] | `0.041` |
| Final energy | `-7.942969` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l5_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l5_np32.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `cg` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.1` |
| `ksp_max_it` | `30` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.1` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.239 | 7 | 8 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.239` |
| Newton iterations | `7` |
| Linear iterations | `8` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.132040 | 8661.946477 | 1796.149353 | nan | 1.956948 | 1 | yes | 9 | 0.001 | 0.022 | 0.005 | 0.006 | 1.000000 | nan |
| 2 | -5.633497 | 5.765537 | 1.470252 | nan | 0.219327 | 1 | yes | 9 | 0.000 | 0.023 | 0.005 | 0.006 | 1.000000 | nan |
| 3 | -7.664686 | 2.031189 | 4.935033 | nan | 0.551183 | 1 | yes | 9 | 0.000 | 0.024 | 0.005 | 0.006 | 1.000000 | nan |
| 4 | -7.930507 | 0.265821 | 2.340847 | nan | 0.948817 | 2 | yes | 9 | 0.000 | 0.021 | 0.006 | 0.006 | 1.000000 | nan |
| 5 | -7.942603 | 0.012096 | 0.429335 | nan | 1.055248 | 1 | yes | 9 | 0.000 | 0.021 | 0.004 | 0.007 | 1.000000 | nan |
| 6 | -7.942967 | 0.000364 | 0.055859 | nan | 1.141353 | 1 | yes | 9 | 0.001 | 0.021 | 0.004 | 0.005 | 1.000000 | nan |
| 7 | -7.942969 | 0.000001 | 0.003873 | 0.000063 | 1.002033 | 1 | yes | 9 | 0.000 | 0.022 | 0.004 | 0.006 | 1.000000 | nan |
