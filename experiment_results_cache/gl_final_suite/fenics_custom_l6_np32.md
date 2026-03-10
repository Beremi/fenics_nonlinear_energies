# fenics_custom_l6_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16630` |
| Setup time [s] | `0.066` |
| Total solve time [s] | `0.314` |
| Total Newton iterations | `8` |
| Total linear iterations | `34` |
| Total assembly time [s] | `0.004` |
| Total PC init time [s] | `0.168` |
| Total KSP solve time [s] | `0.100` |
| Total line-search time [s] | `0.022` |
| Final energy | `0.345777` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l6_np32.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l6_np32.log` |

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
| 1 | 0.314 | 8 | 34 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.314` |
| Newton iterations | `8` |
| Linear iterations | `34` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.566888 | 0.098453 | 0.007408 | nan | -0.259830 | 11 | yes | 19 | 0.001 | 0.022 | 0.026 | 0.003 | 1.000000 | nan |
| 2 | 0.507054 | 0.059835 | 0.008124 | nan | 0.321443 | 5 | yes | 19 | 0.001 | 0.026 | 0.015 | 0.003 | 1.000000 | nan |
| 3 | 0.420086 | 0.086968 | 0.003977 | nan | -0.468594 | 3 | yes | 19 | 0.000 | 0.019 | 0.010 | 0.003 | 1.000000 | nan |
| 4 | 0.409691 | 0.010395 | 0.006419 | nan | -0.019577 | 5 | yes | 19 | 0.000 | 0.020 | 0.013 | 0.003 | 1.000000 | nan |
| 5 | 0.356704 | 0.052986 | 0.006361 | nan | 0.252700 | 4 | yes | 19 | 0.000 | 0.021 | 0.011 | 0.003 | 1.000000 | nan |
| 6 | 0.345886 | 0.010818 | 0.004311 | nan | 1.033573 | 2 | yes | 19 | 0.000 | 0.021 | 0.008 | 0.003 | 1.000000 | nan |
| 7 | 0.345777 | 0.000109 | 0.000339 | nan | 1.018477 | 2 | yes | 19 | 0.000 | 0.020 | 0.008 | 0.003 | 1.000000 | nan |
| 8 | 0.345777 | 0.000000 | 0.000003 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.000 | 0.019 | 0.008 | 0.003 | 1.000000 | nan |
