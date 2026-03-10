# jax_petsc_element_l8_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `261121` |
| Setup time [s] | `1.023` |
| Total solve time [s] | `0.900` |
| Total Newton iterations | `6` |
| Total linear iterations | `20` |
| Total assembly time [s] | `0.129` |
| Total PC init time [s] | `0.204` |
| Total KSP solve time [s] | `0.131` |
| Total line-search time [s] | `0.402` |
| Final energy | `0.345634` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l8_np8.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l8_np8.log` |

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
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.900 | 6 | 20 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.900` |
| Newton iterations | `6` |
| Linear iterations | `20` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.469695 | 0.195606 | 0.001852 | nan | -0.499650 | 5 | yes | 19 | 0.022 | 0.040 | 0.030 | 0.069 | 1.000000 | nan |
| 2 | 0.414387 | 0.055309 | 0.001988 | nan | 0.048467 | 7 | yes | 19 | 0.022 | 0.034 | 0.042 | 0.068 | 1.000000 | nan |
| 3 | 0.348730 | 0.065657 | 0.001877 | nan | 0.609413 | 2 | yes | 19 | 0.022 | 0.034 | 0.015 | 0.067 | 1.000000 | nan |
| 4 | 0.345653 | 0.003077 | 0.000564 | nan | 1.049533 | 2 | yes | 19 | 0.021 | 0.032 | 0.015 | 0.067 | 1.000000 | nan |
| 5 | 0.345634 | 0.000019 | 0.000037 | nan | 1.011846 | 2 | yes | 19 | 0.021 | 0.032 | 0.015 | 0.067 | 1.000000 | nan |
| 6 | 0.345634 | 0.000000 | 0.000000 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.021 | 0.033 | 0.015 | 0.066 | 1.000000 | nan |
