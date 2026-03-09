# jax_petsc_element_l5_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `2945` |
| Setup time [s] | `0.297` |
| Total solve time [s] | `0.061` |
| Total Newton iterations | `6` |
| Total linear iterations | `6` |
| Total assembly time [s] | `0.002` |
| Total PC init time [s] | `0.030` |
| Total KSP solve time [s] | `0.007` |
| Total line-search time [s] | `0.020` |
| Final energy | `-7.942969` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l5_np4.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l5_np4.log` |

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
| 1 | 0.061 | 6 | 6 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.061` |
| Newton iterations | `6` |
| Linear iterations | `6` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.096759 | 8661.981757 | 1796.149353 | nan | 1.956948 | 1 | yes | 9 | 0.000 | 0.006 | 0.001 | 0.003 | 1.000000 | nan |
| 2 | -5.834071 | 5.930831 | 1.185271 | nan | 0.219327 | 1 | yes | 9 | 0.000 | 0.005 | 0.001 | 0.003 | 1.000000 | nan |
| 3 | -7.716831 | 1.882760 | 4.555039 | nan | 0.551183 | 1 | yes | 9 | 0.000 | 0.005 | 0.001 | 0.003 | 1.000000 | nan |
| 4 | -7.938894 | 0.222063 | 2.030273 | nan | 1.055248 | 1 | yes | 9 | 0.000 | 0.005 | 0.001 | 0.003 | 1.000000 | nan |
| 5 | -7.942939 | 0.004045 | 0.214732 | nan | 1.088137 | 1 | yes | 9 | 0.000 | 0.005 | 0.001 | 0.003 | 1.000000 | nan |
| 6 | -7.942969 | 0.000030 | 0.016730 | 0.000837 | 1.055248 | 1 | yes | 9 | 0.000 | 0.005 | 0.001 | 0.003 | 1.000000 | nan |
