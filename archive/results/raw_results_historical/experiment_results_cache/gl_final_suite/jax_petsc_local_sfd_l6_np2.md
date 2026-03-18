# jax_petsc_local_sfd_l6_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16129` |
| Setup time [s] | `0.960` |
| Total solve time [s] | `0.142` |
| Total Newton iterations | `7` |
| Total linear iterations | `28` |
| Total assembly time [s] | `0.016` |
| Total PC init time [s] | `0.041` |
| Total KSP solve time [s] | `0.023` |
| Total line-search time [s] | `0.057` |
| Final energy | `0.345777` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l6_np2.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l6_np2.log` |

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
| 1 | 0.142 | 7 | 28 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.142` |
| Newton iterations | `7` |
| Linear iterations | `28` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.545669 | 0.119673 | 0.007408 | nan | -0.284255 | 8 | yes | 19 | 0.003 | 0.007 | 0.006 | 0.008 | 1.000000 | nan |
| 2 | 0.512179 | 0.033490 | 0.007911 | nan | 0.347434 | 6 | yes | 19 | 0.002 | 0.006 | 0.005 | 0.008 | 1.000000 | nan |
| 3 | 0.444413 | 0.067766 | 0.004562 | nan | -0.379524 | 5 | yes | 19 | 0.002 | 0.006 | 0.004 | 0.008 | 1.000000 | nan |
| 4 | 0.370136 | 0.074277 | 0.007235 | nan | 0.244936 | 3 | yes | 19 | 0.002 | 0.005 | 0.003 | 0.008 | 1.000000 | nan |
| 5 | 0.346584 | 0.023552 | 0.005501 | nan | 0.885088 | 2 | yes | 19 | 0.002 | 0.006 | 0.002 | 0.008 | 1.000000 | nan |
| 6 | 0.345778 | 0.000806 | 0.000944 | nan | 1.065762 | 2 | yes | 19 | 0.002 | 0.005 | 0.002 | 0.008 | 1.000000 | nan |
| 7 | 0.345777 | 0.000001 | 0.000033 | 0.000000 | 1.001816 | 2 | yes | 19 | 0.002 | 0.005 | 0.002 | 0.008 | 1.000000 | nan |
