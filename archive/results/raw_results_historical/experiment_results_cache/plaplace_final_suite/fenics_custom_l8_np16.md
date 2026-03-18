# fenics_custom_l8_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `197492` |
| Setup time [s] | `0.508` |
| Total solve time [s] | `0.416` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.033` |
| Total PC init time [s] | `0.202` |
| Total KSP solve time [s] | `0.074` |
| Total line-search time [s] | `0.090` |
| Final energy | `-7.959556` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l8_np16.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l8_np16.log` |

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
| 1 | 0.416 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.416` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -3.977913 | 8.280404 | 11.252993 | nan | 0.776608 | 2 | yes | 9 | 0.006 | 0.037 | 0.013 | 0.015 | 1.000000 | nan |
| 2 | -7.350876 | 3.372963 | 9.276936 | nan | 0.809497 | 2 | yes | 9 | 0.005 | 0.034 | 0.012 | 0.015 | 1.000000 | nan |
| 3 | -7.927335 | 0.576459 | 3.224605 | nan | 1.002033 | 2 | yes | 9 | 0.005 | 0.032 | 0.012 | 0.015 | 1.000000 | nan |
| 4 | -7.958747 | 0.031412 | 0.684428 | nan | 1.055248 | 2 | yes | 9 | 0.005 | 0.033 | 0.012 | 0.015 | 1.000000 | nan |
| 5 | -7.959540 | 0.000793 | 0.099289 | nan | 1.088137 | 2 | yes | 9 | 0.005 | 0.033 | 0.012 | 0.015 | 1.000000 | nan |
| 6 | -7.959556 | 0.000017 | 0.011851 | 0.000689 | 1.088137 | 2 | yes | 9 | 0.005 | 0.034 | 0.013 | 0.015 | 1.000000 | nan |
