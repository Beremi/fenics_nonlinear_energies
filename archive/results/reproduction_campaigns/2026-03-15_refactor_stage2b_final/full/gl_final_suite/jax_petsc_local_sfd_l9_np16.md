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
| Setup time [s] | `4.571` |
| Total solve time [s] | `4.448` |
| Total Newton iterations | `8` |
| Total linear iterations | `38` |
| Total assembly time [s] | `1.446` |
| Total PC init time [s] | `0.738` |
| Total KSP solve time [s] | `0.712` |
| Total line-search time [s] | `1.437` |
| Final energy | `0.345626` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l9_np16.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l9_np16.log` |

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
| 1 | 4.448 | 8 | 38 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `4.448` |
| Newton iterations | `8` |
| Linear iterations | `38` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.565306 | 0.099994 | 0.000926 | nan | -0.265328 | 10 | yes | 19 | 0.363 | 0.105 | 0.175 | 0.180 | 1.000000 | nan |
| 2 | 0.508307 | 0.056998 | 0.001021 | nan | 0.320743 | 6 | yes | 19 | 0.152 | 0.091 | 0.108 | 0.178 | 1.000000 | nan |
| 3 | 0.441728 | 0.066579 | 0.000554 | nan | -0.303716 | 6 | yes | 19 | 0.157 | 0.091 | 0.108 | 0.178 | 1.000000 | nan |
| 4 | 0.375989 | 0.065738 | 0.000858 | nan | 0.223642 | 5 | yes | 19 | 0.155 | 0.090 | 0.092 | 0.179 | 1.000000 | nan |
| 5 | 0.348268 | 0.027721 | 0.000741 | nan | 0.718810 | 3 | yes | 19 | 0.156 | 0.090 | 0.061 | 0.180 | 1.000000 | nan |
| 6 | 0.345634 | 0.002634 | 0.000227 | nan | 1.079457 | 2 | yes | 19 | 0.154 | 0.091 | 0.045 | 0.181 | 1.000000 | nan |
| 7 | 0.345626 | 0.000008 | 0.000012 | nan | 1.004082 | 3 | yes | 19 | 0.154 | 0.091 | 0.062 | 0.181 | 1.000000 | nan |
| 8 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 0.999551 | 3 | yes | 19 | 0.156 | 0.091 | 0.061 | 0.181 | 1.000000 | nan |
