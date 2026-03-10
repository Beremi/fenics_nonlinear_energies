# jax_petsc_element_l9_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `1.643` |
| Total solve time [s] | `1.937` |
| Total Newton iterations | `7` |
| Total linear iterations | `39` |
| Total assembly time [s] | `0.177` |
| Total PC init time [s] | `0.476` |
| Total KSP solve time [s] | `0.347` |
| Total line-search time [s] | `0.878` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l9_np32.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l9_np32.log` |

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
| 1 | 1.937 | 7 | 39 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.937` |
| Newton iterations | `7` |
| Linear iterations | `39` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.567294 | 0.098005 | 0.000926 | nan | -0.260530 | 12 | yes | 19 | 0.025 | 0.077 | 0.102 | 0.124 | 1.000000 | nan |
| 2 | 0.508740 | 0.058554 | 0.001016 | nan | 0.308881 | 6 | yes | 19 | 0.025 | 0.067 | 0.053 | 0.126 | 1.000000 | nan |
| 3 | 0.440645 | 0.068095 | 0.000564 | nan | -0.324908 | 7 | yes | 19 | 0.024 | 0.069 | 0.060 | 0.126 | 1.000000 | nan |
| 4 | 0.366617 | 0.074028 | 0.000870 | nan | 0.269093 | 5 | yes | 19 | 0.026 | 0.067 | 0.044 | 0.125 | 1.000000 | nan |
| 5 | 0.346065 | 0.020552 | 0.000658 | nan | 0.927141 | 3 | yes | 19 | 0.025 | 0.066 | 0.028 | 0.125 | 1.000000 | nan |
| 6 | 0.345627 | 0.000439 | 0.000088 | nan | 1.043602 | 3 | yes | 19 | 0.026 | 0.063 | 0.029 | 0.126 | 1.000000 | nan |
| 7 | 0.345626 | 0.000000 | 0.000002 | 0.000000 | 1.000416 | 3 | yes | 19 | 0.026 | 0.067 | 0.030 | 0.126 | 1.000000 | nan |
