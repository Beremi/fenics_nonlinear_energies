# fenics_custom_l8_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `261121` |
| Setup time [s] | `0.633` |
| Total solve time [s] | `4.335` |
| Total Newton iterations | `7` |
| Total linear iterations | `20` |
| Total assembly time [s] | `0.599` |
| Total PC init time [s] | `0.651` |
| Total KSP solve time [s] | `0.554` |
| Total line-search time [s] | `2.293` |
| Final energy | `0.345634` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l8_np1.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l8_np1.log` |

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
| 1 | 4.335 | 7 | 20 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `4.335` |
| Newton iterations | `7` |
| Linear iterations | `20` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.505217 | 0.160085 | 0.001852 | nan | -0.499650 | 3 | yes | 19 | 0.090 | 0.107 | 0.084 | 0.330 | 1.000000 | nan |
| 2 | 0.489236 | 0.015980 | 0.002069 | nan | 0.000050 | 6 | yes | 19 | 0.085 | 0.093 | 0.145 | 0.327 | 1.000000 | nan |
| 3 | 0.404011 | 0.085225 | 0.002051 | nan | 0.114245 | 3 | yes | 19 | 0.085 | 0.091 | 0.083 | 0.337 | 1.000000 | nan |
| 4 | 0.347794 | 0.056217 | 0.001771 | nan | 0.693685 | 2 | yes | 19 | 0.085 | 0.091 | 0.060 | 0.324 | 1.000000 | nan |
| 5 | 0.345639 | 0.002155 | 0.000440 | nan | 1.087654 | 2 | yes | 19 | 0.084 | 0.090 | 0.060 | 0.325 | 1.000000 | nan |
| 6 | 0.345634 | 0.000006 | 0.000026 | nan | 1.003382 | 2 | yes | 19 | 0.084 | 0.090 | 0.060 | 0.327 | 1.000000 | nan |
| 7 | 0.345634 | 0.000000 | 0.000000 | 0.000000 | 1.004515 | 2 | yes | 19 | 0.085 | 0.090 | 0.060 | 0.324 | 1.000000 | nan |
