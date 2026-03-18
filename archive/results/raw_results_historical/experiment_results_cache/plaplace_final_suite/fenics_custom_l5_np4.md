# fenics_custom_l5_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `3121` |
| Setup time [s] | `0.035` |
| Total solve time [s] | `0.040` |
| Total Newton iterations | `5` |
| Total linear iterations | `5` |
| Total assembly time [s] | `0.002` |
| Total PC init time [s] | `0.024` |
| Total KSP solve time [s] | `0.005` |
| Total line-search time [s] | `0.007` |
| Final energy | `-7.942969` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l5_np4.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l5_np4.log` |

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
| 1 | 0.040 | 5 | 5 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.040` |
| Newton iterations | `5` |
| Linear iterations | `5` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.987518 | 6.139202 | 0.564572 | nan | 0.100333 | 1 | yes | 9 | 0.001 | 0.005 | 0.001 | 0.001 | 1.000000 | nan |
| 2 | -7.716575 | 1.729057 | 5.977739 | nan | 0.637288 | 1 | yes | 9 | 0.000 | 0.005 | 0.001 | 0.001 | 1.000000 | nan |
| 3 | -7.940681 | 0.224106 | 2.302174 | nan | 1.141353 | 1 | yes | 9 | 0.000 | 0.005 | 0.001 | 0.001 | 1.000000 | nan |
| 4 | -7.942965 | 0.002284 | 0.187757 | nan | 1.002033 | 1 | yes | 9 | 0.000 | 0.005 | 0.001 | 0.001 | 1.000000 | nan |
| 5 | -7.942969 | 0.000004 | 0.003575 | 0.000233 | 1.002033 | 1 | yes | 9 | 0.000 | 0.005 | 0.001 | 0.001 | 1.000000 | nan |
