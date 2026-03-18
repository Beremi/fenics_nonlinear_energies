# jax_petsc_local_sfd_l8_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `261121` |
| Setup time [s] | `2.491` |
| Total solve time [s] | `1.782` |
| Total Newton iterations | `7` |
| Total linear iterations | `30` |
| Total assembly time [s] | `0.425` |
| Total PC init time [s] | `0.440` |
| Total KSP solve time [s] | `0.343` |
| Total line-search time [s] | `0.528` |
| Final energy | `0.345634` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l8_np2.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l8_np2.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `gmres` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.001` |
| `ksp_max_it` | `200` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.001` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `sfd_local` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 1.782 | 7 | 30 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.782` |
| Newton iterations | `7` |
| Linear iterations | `30` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.548006 | 0.117296 | 0.001852 | nan | -0.285655 | 9 | yes | 19 | 0.060 | 0.071 | 0.095 | 0.075 | 1.000000 | nan |
| 2 | 0.509737 | 0.038269 | 0.001981 | nan | 0.349267 | 6 | yes | 19 | 0.061 | 0.061 | 0.065 | 0.076 | 1.000000 | nan |
| 3 | 0.442235 | 0.067502 | 0.001115 | nan | -0.344369 | 5 | yes | 19 | 0.061 | 0.062 | 0.056 | 0.075 | 1.000000 | nan |
| 4 | 0.365989 | 0.076246 | 0.001751 | nan | 0.280790 | 4 | yes | 19 | 0.061 | 0.061 | 0.046 | 0.075 | 1.000000 | nan |
| 5 | 0.346021 | 0.019968 | 0.001293 | nan | 0.935771 | 2 | yes | 19 | 0.061 | 0.062 | 0.027 | 0.077 | 1.000000 | nan |
| 6 | 0.345634 | 0.000387 | 0.000163 | nan | 1.041769 | 2 | yes | 19 | 0.061 | 0.061 | 0.027 | 0.075 | 1.000000 | nan |
| 7 | 0.345634 | 0.000000 | 0.000004 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.061 | 0.062 | 0.027 | 0.075 | 1.000000 | nan |
