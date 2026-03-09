# fenics_custom_l8_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `197555` |
| Setup time [s] | `0.533` |
| Total solve time [s] | `0.368` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.022` |
| Total PC init time [s] | `0.212` |
| Total KSP solve time [s] | `0.068` |
| Total line-search time [s] | `0.054` |
| Final energy | `-7.959556` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l8_np32.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l8_np32.log` |

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
| 1 | 0.368 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.368` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -4.029154 | 8.370421 | 11.309276 | nan | 0.776608 | 2 | yes | 9 | 0.005 | 0.040 | 0.011 | 0.009 | 1.000000 | nan |
| 2 | -7.364184 | 3.335031 | 9.121457 | nan | 0.809497 | 2 | yes | 9 | 0.003 | 0.035 | 0.012 | 0.009 | 1.000000 | nan |
| 3 | -7.927495 | 0.563311 | 3.174114 | nan | 1.002033 | 2 | yes | 9 | 0.003 | 0.033 | 0.011 | 0.009 | 1.000000 | nan |
| 4 | -7.958799 | 0.031304 | 0.667438 | nan | 1.055248 | 2 | yes | 9 | 0.004 | 0.035 | 0.012 | 0.009 | 1.000000 | nan |
| 5 | -7.959537 | 0.000739 | 0.089631 | nan | 1.088137 | 2 | yes | 9 | 0.003 | 0.035 | 0.011 | 0.009 | 1.000000 | nan |
| 6 | -7.959556 | 0.000019 | 0.012198 | 0.000985 | 1.141353 | 2 | yes | 9 | 0.004 | 0.033 | 0.011 | 0.009 | 1.000000 | nan |
