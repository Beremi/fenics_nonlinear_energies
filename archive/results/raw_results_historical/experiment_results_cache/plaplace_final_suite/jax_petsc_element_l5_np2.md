# jax_petsc_element_l5_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `3201` |
| Free DOFs | `2945` |
| Setup time [s] | `0.297` |
| Total solve time [s] | `0.049` |
| Total Newton iterations | `6` |
| Total linear iterations | `6` |
| Total assembly time [s] | `0.003` |
| Total PC init time [s] | `0.022` |
| Total KSP solve time [s] | `0.005` |
| Total line-search time [s] | `0.018` |
| Final energy | `-7.942969` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l5_np2.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/jax_petsc_element_l5_np2.log` |

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
| 1 | 0.049 | 6 | 6 | -7.942969 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.049` |
| Newton iterations | `6` |
| Linear iterations | `6` |
| Energy | `-7.942969` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.104966 | 8662.183483 | 1796.149353 | nan | 1.956948 | 1 | yes | 9 | 0.001 | 0.004 | 0.001 | 0.003 | 1.000000 | nan |
| 2 | -6.051242 | 5.946276 | 1.102847 | nan | 0.219327 | 1 | yes | 9 | 0.000 | 0.004 | 0.001 | 0.003 | 1.000000 | nan |
| 3 | -7.763671 | 1.712429 | 4.050638 | nan | 0.584072 | 1 | yes | 9 | 0.000 | 0.004 | 0.001 | 0.003 | 1.000000 | nan |
| 4 | -7.940583 | 0.176912 | 1.798694 | nan | 1.055248 | 1 | yes | 9 | 0.000 | 0.004 | 0.001 | 0.003 | 1.000000 | nan |
| 5 | -7.942963 | 0.002380 | 0.160770 | nan | 1.055248 | 1 | yes | 9 | 0.000 | 0.003 | 0.001 | 0.003 | 1.000000 | nan |
| 6 | -7.942969 | 0.000006 | 0.008625 | 0.000131 | 1.002033 | 1 | yes | 9 | 0.000 | 0.004 | 0.001 | 0.003 | 1.000000 | nan |
