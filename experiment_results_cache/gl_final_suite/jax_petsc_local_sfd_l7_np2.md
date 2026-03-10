# jax_petsc_local_sfd_l7_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65025` |
| Setup time [s] | `1.226` |
| Total solve time [s] | `0.328` |
| Total Newton iterations | `6` |
| Total linear iterations | `21` |
| Total assembly time [s] | `0.055` |
| Total PC init time [s] | `0.105` |
| Total KSP solve time [s] | `0.057` |
| Total line-search time [s] | `0.101` |
| Final energy | `0.345662` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l7_np2.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l7_np2.log` |

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
| 1 | 0.328 | 6 | 21 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.328` |
| Newton iterations | `6` |
| Linear iterations | `21` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.489881 | 0.175429 | 0.003705 | nan | -0.327008 | 7 | yes | 19 | 0.010 | 0.022 | 0.018 | 0.017 | 1.000000 | nan |
| 2 | 0.470053 | 0.019828 | 0.003384 | nan | -0.499650 | 5 | yes | 19 | 0.010 | 0.017 | 0.012 | 0.017 | 1.000000 | nan |
| 3 | 0.377964 | 0.092089 | 0.004974 | nan | 0.178457 | 3 | yes | 19 | 0.009 | 0.017 | 0.008 | 0.017 | 1.000000 | nan |
| 4 | 0.346432 | 0.031532 | 0.003748 | nan | 0.915712 | 2 | yes | 19 | 0.009 | 0.016 | 0.006 | 0.017 | 1.000000 | nan |
| 5 | 0.345663 | 0.000769 | 0.000506 | nan | 1.041337 | 2 | yes | 19 | 0.009 | 0.017 | 0.006 | 0.017 | 1.000000 | nan |
| 6 | 0.345662 | 0.000000 | 0.000012 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.009 | 0.017 | 0.006 | 0.017 | 1.000000 | nan |
