# jax_petsc_element_l8_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `197633` |
| Free DOFs | `195585` |
| Setup time [s] | `0.631` |
| Total solve time [s] | `0.601` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.083` |
| Total PC init time [s] | `0.272` |
| Total KSP solve time [s] | `0.092` |
| Total line-search time [s] | `0.134` |
| Final energy | `-7.959556` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l8_np8.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l8_np8.log` |

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
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.601 | 6 | 12 | -7.959556 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.601` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.959556` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.922941 | 4134621.047272 | 111906.518518 | nan | 1.956948 | 2 | yes | 9 | 0.015 | 0.050 | 0.015 | 0.023 | 1.000000 | nan |
| 2 | -5.529630 | 47.452571 | 51.891754 | nan | 1.817627 | 2 | yes | 9 | 0.014 | 0.045 | 0.015 | 0.022 | 1.000000 | nan |
| 3 | -7.583763 | 2.054133 | 7.992155 | nan | 0.723392 | 2 | yes | 9 | 0.014 | 0.048 | 0.016 | 0.022 | 1.000000 | nan |
| 4 | -7.956786 | 0.373023 | 2.995723 | nan | 1.174242 | 2 | yes | 9 | 0.014 | 0.043 | 0.015 | 0.022 | 1.000000 | nan |
| 5 | -7.959532 | 0.002746 | 0.229873 | nan | 1.055248 | 2 | yes | 9 | 0.014 | 0.043 | 0.015 | 0.022 | 1.000000 | nan |
| 6 | -7.959556 | 0.000024 | 0.015490 | 0.001101 | 1.088137 | 2 | yes | 9 | 0.013 | 0.042 | 0.015 | 0.022 | 1.000000 | nan |
