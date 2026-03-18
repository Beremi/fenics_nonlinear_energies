# fenics_custom_l6_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16533` |
| Setup time [s] | `0.054` |
| Total solve time [s] | `0.144` |
| Total Newton iterations | `6` |
| Total linear iterations | `17` |
| Total assembly time [s] | `0.006` |
| Total PC init time [s] | `0.066` |
| Total KSP solve time [s] | `0.029` |
| Total line-search time [s] | `0.024` |
| Final energy | `0.345777` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l6_np8.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l6_np8.log` |

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
| 1 | 0.144 | 6 | 17 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.144` |
| Newton iterations | `6` |
| Linear iterations | `17` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.494058 | 0.171284 | 0.007408 | nan | -0.499650 | 5 | yes | 19 | 0.001 | 0.013 | 0.008 | 0.004 | 1.000000 | nan |
| 2 | 0.397131 | 0.096926 | 0.008193 | nan | 0.111979 | 4 | yes | 19 | 0.001 | 0.011 | 0.006 | 0.004 | 1.000000 | nan |
| 3 | 0.347553 | 0.049578 | 0.006964 | nan | 0.753532 | 2 | yes | 19 | 0.001 | 0.011 | 0.004 | 0.004 | 1.000000 | nan |
| 4 | 0.345782 | 0.001771 | 0.001522 | nan | 1.121943 | 2 | yes | 19 | 0.001 | 0.010 | 0.004 | 0.004 | 1.000000 | nan |
| 5 | 0.345777 | 0.000006 | 0.000109 | nan | 1.002249 | 2 | yes | 19 | 0.001 | 0.010 | 0.004 | 0.004 | 1.000000 | nan |
| 6 | 0.345777 | 0.000000 | 0.000000 | 0.000000 | 0.999283 | 2 | yes | 19 | 0.001 | 0.010 | 0.004 | 0.004 | 1.000000 | nan |
