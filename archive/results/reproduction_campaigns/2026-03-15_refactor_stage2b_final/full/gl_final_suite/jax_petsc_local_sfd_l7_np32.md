# jax_petsc_local_sfd_l7_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65025` |
| Setup time [s] | `1.027` |
| Total solve time [s] | `0.709` |
| Total Newton iterations | `7` |
| Total linear iterations | `32` |
| Total assembly time [s] | `0.207` |
| Total PC init time [s] | `0.213` |
| Total KSP solve time [s] | `0.120` |
| Total line-search time [s] | `0.158` |
| Final energy | `0.345662` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l7_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l7_np32.log` |

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
| 1 | 0.709 | 7 | 32 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.709` |
| Newton iterations | `7` |
| Linear iterations | `32` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.563255 | 0.102055 | 0.003705 | nan | -0.270559 | 9 | yes | 19 | 0.190 | 0.032 | 0.030 | 0.022 | 1.000000 | nan |
| 2 | 0.507930 | 0.055325 | 0.004127 | nan | 0.337837 | 6 | yes | 19 | 0.003 | 0.029 | 0.021 | 0.022 | 1.000000 | nan |
| 3 | 0.440288 | 0.067642 | 0.002196 | nan | -0.296817 | 6 | yes | 19 | 0.003 | 0.036 | 0.026 | 0.022 | 1.000000 | nan |
| 4 | 0.363176 | 0.077112 | 0.003412 | nan | 0.301550 | 4 | yes | 19 | 0.003 | 0.030 | 0.014 | 0.023 | 1.000000 | nan |
| 5 | 0.345881 | 0.017294 | 0.002462 | nan | 0.965262 | 3 | yes | 19 | 0.003 | 0.028 | 0.011 | 0.023 | 1.000000 | nan |
| 6 | 0.345662 | 0.000219 | 0.000243 | nan | 1.023976 | 2 | yes | 19 | 0.002 | 0.027 | 0.009 | 0.022 | 1.000000 | nan |
| 7 | 0.345662 | 0.000000 | 0.000003 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.002 | 0.029 | 0.010 | 0.023 | 1.000000 | nan |
