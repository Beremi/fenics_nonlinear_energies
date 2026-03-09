# fenics_custom_l6_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12429` |
| Setup time [s] | `0.059` |
| Total solve time [s] | `0.084` |
| Total Newton iterations | `5` |
| Total linear iterations | `9` |
| Total assembly time [s] | `0.006` |
| Total PC init time [s] | `0.042` |
| Total KSP solve time [s] | `0.012` |
| Total line-search time [s] | `0.019` |
| Final energy | `-7.954564` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l6_np4.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l6_np4.log` |

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
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.084 | 5 | 9 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.084` |
| Newton iterations | `5` |
| Linear iterations | `9` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.913967 | 6.128121 | 0.764575 | nan | 0.186438 | 1 | yes | 9 | 0.001 | 0.009 | 0.002 | 0.004 | 1.000000 | nan |
| 2 | -7.702910 | 1.788943 | 5.543528 | nan | 0.584072 | 2 | yes | 9 | 0.001 | 0.008 | 0.003 | 0.004 | 1.000000 | nan |
| 3 | -7.951029 | 0.248119 | 2.485123 | nan | 1.088137 | 2 | yes | 9 | 0.001 | 0.009 | 0.003 | 0.004 | 1.000000 | nan |
| 4 | -7.954542 | 0.003513 | 0.216883 | nan | 1.055248 | 2 | yes | 9 | 0.001 | 0.008 | 0.003 | 0.004 | 1.000000 | nan |
| 5 | -7.954564 | 0.000022 | 0.014429 | 0.000427 | 1.055248 | 2 | yes | 9 | 0.001 | 0.008 | 0.003 | 0.004 | 1.000000 | nan |
