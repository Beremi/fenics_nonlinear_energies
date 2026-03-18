# jax_petsc_local_sfd_l9_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `2.865` |
| Total solve time [s] | `4.086` |
| Total Newton iterations | `8` |
| Total linear iterations | `38` |
| Total assembly time [s] | `1.197` |
| Total PC init time [s] | `0.663` |
| Total KSP solve time [s] | `0.697` |
| Total line-search time [s] | `1.415` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l9_np16.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l9_np16.log` |

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
| 1 | 4.086 | 8 | 38 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `4.086` |
| Newton iterations | `8` |
| Linear iterations | `38` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.565306 | 0.099994 | 0.000926 | nan | -0.265328 | 10 | yes | 19 | 0.156 | 0.124 | 0.175 | 0.180 | 1.000000 | nan |
| 2 | 0.508307 | 0.056998 | 0.001021 | nan | 0.320743 | 6 | yes | 19 | 0.149 | 0.077 | 0.105 | 0.178 | 1.000000 | nan |
| 3 | 0.441728 | 0.066579 | 0.000554 | nan | -0.303716 | 6 | yes | 19 | 0.146 | 0.078 | 0.105 | 0.176 | 1.000000 | nan |
| 4 | 0.375989 | 0.065738 | 0.000858 | nan | 0.223642 | 5 | yes | 19 | 0.149 | 0.076 | 0.090 | 0.176 | 1.000000 | nan |
| 5 | 0.348268 | 0.027721 | 0.000741 | nan | 0.718810 | 3 | yes | 19 | 0.148 | 0.077 | 0.060 | 0.176 | 1.000000 | nan |
| 6 | 0.345634 | 0.002634 | 0.000227 | nan | 1.079457 | 2 | yes | 19 | 0.148 | 0.077 | 0.044 | 0.176 | 1.000000 | nan |
| 7 | 0.345626 | 0.000008 | 0.000012 | nan | 1.004082 | 3 | yes | 19 | 0.151 | 0.079 | 0.059 | 0.176 | 1.000000 | nan |
| 8 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 0.999551 | 3 | yes | 19 | 0.148 | 0.076 | 0.059 | 0.176 | 1.000000 | nan |
