# fenics_custom_l7_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `49172` |
| Setup time [s] | `0.168` |
| Total solve time [s] | `0.456` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.054` |
| Total PC init time [s] | `0.161` |
| Total KSP solve time [s] | `0.056` |
| Total line-search time [s] | `0.159` |
| Final energy | `-7.958292` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l7_np2.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l7_np2.log` |

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
| 1 | 0.456 | 6 | 11 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.456` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.395810 | 6.058701 | 2.785271 | nan | 0.358647 | 1 | yes | 9 | 0.010 | 0.032 | 0.006 | 0.027 | 1.000000 | nan |
| 2 | -7.636890 | 2.241080 | 6.072492 | nan | 0.584072 | 2 | yes | 9 | 0.009 | 0.028 | 0.010 | 0.026 | 1.000000 | nan |
| 3 | -7.941983 | 0.305094 | 2.647068 | nan | 0.948817 | 2 | yes | 9 | 0.009 | 0.026 | 0.010 | 0.026 | 1.000000 | nan |
| 4 | -7.958121 | 0.016137 | 0.520452 | nan | 1.088137 | 2 | yes | 9 | 0.009 | 0.026 | 0.010 | 0.027 | 1.000000 | nan |
| 5 | -7.958291 | 0.000170 | 0.042358 | nan | 1.088137 | 2 | yes | 9 | 0.009 | 0.024 | 0.010 | 0.026 | 1.000000 | nan |
| 6 | -7.958292 | 0.000002 | 0.003213 | 0.000125 | 1.055248 | 2 | yes | 9 | 0.009 | 0.025 | 0.010 | 0.026 | 1.000000 | nan |
