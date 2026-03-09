# fenics_custom_l8_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `196645` |
| Setup time [s] | `0.637` |
| Total solve time [s] | `1.722` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.201` |
| Total PC init time [s] | `0.577` |
| Total KSP solve time [s] | `0.223` |
| Total line-search time [s] | `0.625` |
| Final energy | `-7.959556` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l8_np2.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l8_np2.log` |

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
| 1 | 1.722 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.722` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -4.001229 | 8.279039 | 11.187600 | nan | 0.776608 | 2 | yes | 9 | 0.036 | 0.116 | 0.039 | 0.105 | 1.000000 | nan |
| 2 | -7.370013 | 3.368784 | 9.266701 | nan | 0.809497 | 2 | yes | 9 | 0.032 | 0.111 | 0.039 | 0.104 | 1.000000 | nan |
| 3 | -7.928319 | 0.558306 | 3.164621 | nan | 1.002033 | 2 | yes | 9 | 0.032 | 0.094 | 0.036 | 0.104 | 1.000000 | nan |
| 4 | -7.958798 | 0.030479 | 0.665132 | nan | 1.055248 | 2 | yes | 9 | 0.034 | 0.087 | 0.036 | 0.103 | 1.000000 | nan |
| 5 | -7.959543 | 0.000745 | 0.089276 | nan | 1.088137 | 2 | yes | 9 | 0.034 | 0.085 | 0.036 | 0.104 | 1.000000 | nan |
| 6 | -7.959556 | 0.000013 | 0.009802 | 0.000639 | 1.088137 | 2 | yes | 9 | 0.032 | 0.086 | 0.037 | 0.104 | 1.000000 | nan |
