# fenics_custom_l7_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `49471` |
| Setup time [s] | `0.153` |
| Total solve time [s] | `0.224` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.015` |
| Total PC init time [s] | `0.119` |
| Total KSP solve time [s] | `0.035` |
| Total line-search time [s] | `0.046` |
| Final energy | `-7.958292` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l7_np8.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l7_np8.log` |

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
| 1 | 0.224 | 6 | 11 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.224` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.294315 | 5.966159 | 2.823305 | nan | 0.358647 | 1 | yes | 9 | 0.003 | 0.021 | 0.004 | 0.008 | 1.000000 | nan |
| 2 | -7.593532 | 2.299217 | 6.304315 | nan | 0.584072 | 2 | yes | 9 | 0.002 | 0.020 | 0.006 | 0.008 | 1.000000 | nan |
| 3 | -7.939266 | 0.345735 | 2.848327 | nan | 1.002033 | 2 | yes | 9 | 0.002 | 0.019 | 0.006 | 0.008 | 1.000000 | nan |
| 4 | -7.957990 | 0.018724 | 0.549093 | nan | 1.088137 | 2 | yes | 9 | 0.002 | 0.020 | 0.006 | 0.008 | 1.000000 | nan |
| 5 | -7.958289 | 0.000299 | 0.056949 | nan | 1.088137 | 2 | yes | 9 | 0.002 | 0.020 | 0.006 | 0.008 | 1.000000 | nan |
| 6 | -7.958292 | 0.000004 | 0.005016 | 0.000172 | 1.055248 | 2 | yes | 9 | 0.002 | 0.020 | 0.006 | 0.008 | 1.000000 | nan |
