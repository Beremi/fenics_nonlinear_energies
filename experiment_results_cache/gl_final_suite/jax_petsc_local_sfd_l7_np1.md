# jax_petsc_local_sfd_l7_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65025` |
| Setup time [s] | `1.545` |
| Total solve time [s] | `0.571` |
| Total Newton iterations | `7` |
| Total linear iterations | `28` |
| Total assembly time [s] | `0.123` |
| Total PC init time [s] | `0.159` |
| Total KSP solve time [s] | `0.114` |
| Total line-search time [s] | `0.159` |
| Final energy | `0.345662` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l7_np1.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_local_sfd_l7_np1.log` |

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
| 1 | 0.571 | 7 | 28 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.571` |
| Newton iterations | `7` |
| Linear iterations | `28` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.559896 | 0.105414 | 0.003705 | nan | -0.271259 | 9 | yes | 19 | 0.017 | 0.025 | 0.034 | 0.023 | 1.000000 | nan |
| 2 | 0.508794 | 0.051102 | 0.004045 | nan | 0.333306 | 6 | yes | 19 | 0.019 | 0.022 | 0.023 | 0.022 | 1.000000 | nan |
| 3 | 0.441805 | 0.066989 | 0.002202 | nan | -0.306681 | 4 | yes | 19 | 0.018 | 0.022 | 0.016 | 0.023 | 1.000000 | nan |
| 4 | 0.365226 | 0.076579 | 0.003423 | nan | 0.295351 | 3 | yes | 19 | 0.017 | 0.022 | 0.013 | 0.023 | 1.000000 | nan |
| 5 | 0.346021 | 0.019205 | 0.002540 | nan | 0.940137 | 2 | yes | 19 | 0.017 | 0.022 | 0.010 | 0.023 | 1.000000 | nan |
| 6 | 0.345662 | 0.000359 | 0.000312 | nan | 1.037671 | 2 | yes | 19 | 0.017 | 0.022 | 0.010 | 0.022 | 1.000000 | nan |
| 7 | 0.345662 | 0.000000 | 0.000006 | 0.000000 | 1.000684 | 2 | yes | 19 | 0.017 | 0.022 | 0.010 | 0.023 | 1.000000 | nan |
