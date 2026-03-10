# fenics_custom_l8_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `262562` |
| Setup time [s] | `0.747` |
| Total solve time [s] | `1.991` |
| Total Newton iterations | `10` |
| Total linear iterations | `33` |
| Total assembly time [s] | `0.254` |
| Total PC init time [s] | `0.452` |
| Total KSP solve time [s] | `0.324` |
| Total line-search time [s] | `0.862` |
| Final energy | `0.345634` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l8_np4.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l8_np4.log` |

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
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 1.991 | 10 | 33 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.991` |
| Newton iterations | `10` |
| Linear iterations | `33` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.497423 | 0.167878 | 0.001852 | nan | -0.313745 | 7 | yes | 19 | 0.027 | 0.056 | 0.062 | 0.087 | 1.000000 | nan |
| 2 | 0.488732 | 0.008692 | 0.001794 | nan | -0.457164 | 5 | yes | 19 | 0.025 | 0.044 | 0.045 | 0.087 | 1.000000 | nan |
| 3 | 0.469363 | 0.019368 | 0.002582 | nan | 0.030673 | 5 | yes | 19 | 0.025 | 0.044 | 0.045 | 0.085 | 1.000000 | nan |
| 4 | 0.403294 | 0.066069 | 0.002398 | nan | 0.422809 | 3 | yes | 19 | 0.025 | 0.043 | 0.030 | 0.086 | 1.000000 | nan |
| 5 | 0.391633 | 0.011661 | 0.001304 | nan | -0.120777 | 2 | yes | 19 | 0.025 | 0.044 | 0.022 | 0.086 | 1.000000 | nan |
| 6 | 0.361279 | 0.030354 | 0.001511 | nan | 0.452732 | 3 | yes | 19 | 0.025 | 0.044 | 0.030 | 0.086 | 1.000000 | nan |
| 7 | 0.347113 | 0.014167 | 0.001172 | nan | 0.698051 | 2 | yes | 19 | 0.025 | 0.044 | 0.022 | 0.085 | 1.000000 | nan |
| 8 | 0.345637 | 0.001476 | 0.000352 | nan | 1.076491 | 2 | yes | 19 | 0.025 | 0.044 | 0.022 | 0.085 | 1.000000 | nan |
| 9 | 0.345634 | 0.000003 | 0.000017 | nan | 1.002949 | 2 | yes | 19 | 0.025 | 0.044 | 0.022 | 0.087 | 1.000000 | nan |
| 10 | 0.345634 | 0.000000 | 0.000000 | 0.000000 | 1.003649 | 2 | yes | 19 | 0.025 | 0.044 | 0.022 | 0.088 | 1.000000 | nan |
