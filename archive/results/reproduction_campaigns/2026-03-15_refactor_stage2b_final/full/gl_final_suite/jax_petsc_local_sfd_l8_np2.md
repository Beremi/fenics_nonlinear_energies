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
| Setup time [s] | `2.834` |
| Total solve time [s] | `2.299` |
| Total Newton iterations | `7` |
| Total linear iterations | `30` |
| Total assembly time [s] | `0.708` |
| Total PC init time [s] | `0.556` |
| Total KSP solve time [s] | `0.403` |
| Total line-search time [s] | `0.582` |
| Final energy | `0.345634` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l8_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l8_np2.log` |

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
| 1 | 2.299 | 7 | 30 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `2.299` |
| Newton iterations | `7` |
| Linear iterations | `30` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.548006 | 0.117296 | 0.001852 | nan | -0.285655 | 9 | yes | 19 | 0.319 | 0.091 | 0.112 | 0.085 | 1.000000 | nan |
| 2 | 0.509737 | 0.038269 | 0.001981 | nan | 0.349267 | 6 | yes | 19 | 0.065 | 0.078 | 0.077 | 0.083 | 1.000000 | nan |
| 3 | 0.442235 | 0.067502 | 0.001115 | nan | -0.344369 | 5 | yes | 19 | 0.064 | 0.078 | 0.065 | 0.083 | 1.000000 | nan |
| 4 | 0.365989 | 0.076246 | 0.001751 | nan | 0.280790 | 4 | yes | 19 | 0.065 | 0.076 | 0.053 | 0.082 | 1.000000 | nan |
| 5 | 0.346021 | 0.019968 | 0.001293 | nan | 0.935771 | 2 | yes | 19 | 0.065 | 0.077 | 0.032 | 0.084 | 1.000000 | nan |
| 6 | 0.345634 | 0.000387 | 0.000163 | nan | 1.041769 | 2 | yes | 19 | 0.064 | 0.077 | 0.032 | 0.082 | 1.000000 | nan |
| 7 | 0.345634 | 0.000000 | 0.000004 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.065 | 0.077 | 0.032 | 0.083 | 1.000000 | nan |
