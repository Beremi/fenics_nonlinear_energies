# jax_petsc_local_sfd_l6_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16129` |
| Setup time [s] | `1.034` |
| Total solve time [s] | `0.246` |
| Total Newton iterations | `10` |
| Total linear iterations | `36` |
| Total assembly time [s] | `0.043` |
| Total PC init time [s] | `0.061` |
| Total KSP solve time [s] | `0.038` |
| Total line-search time [s] | `0.095` |
| Final energy | `0.345777` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l6_np1.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l6_np1.log` |

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
| 1 | 0.246 | 10 | 36 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.246` |
| Newton iterations | `10` |
| Linear iterations | `36` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.497953 | 0.167388 | 0.007408 | nan | -0.330674 | 7 | yes | 19 | 0.005 | 0.009 | 0.007 | 0.010 | 1.000000 | nan |
| 2 | 0.490764 | 0.007189 | 0.007068 | nan | -0.408047 | 6 | yes | 19 | 0.005 | 0.006 | 0.006 | 0.010 | 1.000000 | nan |
| 3 | 0.483031 | 0.007733 | 0.009652 | nan | 0.020376 | 5 | yes | 19 | 0.004 | 0.006 | 0.005 | 0.010 | 1.000000 | nan |
| 4 | 0.420361 | 0.062670 | 0.008958 | nan | 0.382156 | 3 | yes | 19 | 0.004 | 0.006 | 0.003 | 0.009 | 1.000000 | nan |
| 5 | 0.409554 | 0.010807 | 0.005273 | nan | -0.042869 | 3 | yes | 19 | 0.004 | 0.006 | 0.003 | 0.009 | 1.000000 | nan |
| 6 | 0.377406 | 0.032148 | 0.005770 | nan | 0.539537 | 3 | yes | 19 | 0.004 | 0.006 | 0.003 | 0.010 | 1.000000 | nan |
| 7 | 0.354637 | 0.022769 | 0.004625 | nan | 0.124542 | 3 | yes | 19 | 0.005 | 0.006 | 0.003 | 0.009 | 1.000000 | nan |
| 8 | 0.345858 | 0.008779 | 0.003725 | nan | 1.085655 | 2 | yes | 19 | 0.004 | 0.006 | 0.002 | 0.009 | 1.000000 | nan |
| 9 | 0.345777 | 0.000081 | 0.000306 | nan | 1.022843 | 2 | yes | 19 | 0.004 | 0.006 | 0.002 | 0.010 | 1.000000 | nan |
| 10 | 0.345777 | 0.000000 | 0.000004 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.004 | 0.006 | 0.002 | 0.010 | 1.000000 | nan |
