# jax_petsc_local_sfd_l9_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `7.952` |
| Total solve time [s] | `9.081` |
| Total Newton iterations | `7` |
| Total linear iterations | `29` |
| Total assembly time [s] | `2.623` |
| Total PC init time [s] | `2.179` |
| Total KSP solve time [s] | `1.574` |
| Total line-search time [s] | `2.478` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l9_np2.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l9_np2.log` |

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
| 1 | 9.081 | 7 | 29 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `9.081` |
| Newton iterations | `7` |
| Linear iterations | `29` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.536130 | 0.129170 | 0.000926 | nan | -0.296817 | 8 | yes | 19 | 0.393 | 0.350 | 0.397 | 0.353 | 1.000000 | nan |
| 2 | 0.509876 | 0.026255 | 0.000970 | nan | 0.355465 | 6 | yes | 19 | 0.354 | 0.311 | 0.305 | 0.354 | 1.000000 | nan |
| 3 | 0.439764 | 0.070112 | 0.000570 | nan | -0.426808 | 5 | yes | 19 | 0.372 | 0.304 | 0.267 | 0.354 | 1.000000 | nan |
| 4 | 0.365202 | 0.074562 | 0.000921 | nan | 0.268660 | 4 | yes | 19 | 0.387 | 0.303 | 0.216 | 0.353 | 1.000000 | nan |
| 5 | 0.345849 | 0.019352 | 0.000638 | nan | 0.956797 | 2 | yes | 19 | 0.391 | 0.305 | 0.131 | 0.355 | 1.000000 | nan |
| 6 | 0.345626 | 0.000223 | 0.000064 | nan | 1.016644 | 2 | yes | 19 | 0.356 | 0.302 | 0.129 | 0.354 | 1.000000 | nan |
| 7 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.371 | 0.303 | 0.129 | 0.355 | 1.000000 | nan |
