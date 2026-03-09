# jax_petsc_element_l7_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `48641` |
| Setup time [s] | `0.380` |
| Total solve time [s] | `0.233` |
| Total Newton iterations | `7` |
| Total linear iterations | `14` |
| Total assembly time [s] | `0.015` |
| Total PC init time [s] | `0.127` |
| Total KSP solve time [s] | `0.033` |
| Total line-search time [s] | `0.051` |
| Final energy | `-7.958292` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l7_np8.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l7_np8.log` |

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
| 1 | 0.233 | 7 | 14 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.233` |
| Newton iterations | `7` |
| Linear iterations | `14` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.379593 | 521138.927450 | 28062.385991 | nan | 1.956948 | 2 | yes | 9 | 0.002 | 0.020 | 0.005 | 0.007 | 1.000000 | nan |
| 2 | -3.926200 | 9.305793 | 13.013246 | nan | 0.862712 | 2 | yes | 9 | 0.002 | 0.018 | 0.005 | 0.007 | 1.000000 | nan |
| 3 | -7.344399 | 3.418199 | 9.386387 | nan | 0.809497 | 2 | yes | 9 | 0.002 | 0.019 | 0.005 | 0.007 | 1.000000 | nan |
| 4 | -7.927826 | 0.583428 | 3.187939 | nan | 1.002033 | 2 | yes | 9 | 0.002 | 0.018 | 0.005 | 0.007 | 1.000000 | nan |
| 5 | -7.957641 | 0.029814 | 0.654218 | nan | 1.055248 | 2 | yes | 9 | 0.002 | 0.017 | 0.004 | 0.007 | 1.000000 | nan |
| 6 | -7.958285 | 0.000644 | 0.083240 | nan | 1.088137 | 2 | yes | 9 | 0.002 | 0.018 | 0.005 | 0.007 | 1.000000 | nan |
| 7 | -7.958292 | 0.000008 | 0.007300 | 0.000344 | 1.055248 | 2 | yes | 9 | 0.002 | 0.017 | 0.004 | 0.007 | 1.000000 | nan |
