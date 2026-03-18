# jax_petsc_local_sfd_l9_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `16.981` |
| Total solve time [s] | `19.323` |
| Total Newton iterations | `9` |
| Total linear iterations | `33` |
| Total assembly time [s] | `5.951` |
| Total PC init time [s] | `4.506` |
| Total KSP solve time [s] | `3.434` |
| Total line-search time [s] | `4.957` |
| Final energy | `0.345626` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l9_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l9_np1.log` |

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
| 1 | 19.323 | 9 | 33 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `19.323` |
| Newton iterations | `9` |
| Linear iterations | `33` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.495938 | 0.169361 | 0.000926 | nan | -0.344102 | 7 | yes | 19 | 1.526 | 0.551 | 0.650 | 0.553 | 1.000000 | nan |
| 2 | 0.488637 | 0.007301 | 0.000906 | nan | -0.396884 | 5 | yes | 19 | 0.563 | 0.496 | 0.492 | 0.549 | 1.000000 | nan |
| 3 | 0.471141 | 0.017496 | 0.001247 | nan | -0.014346 | 5 | yes | 19 | 0.564 | 0.493 | 0.490 | 0.553 | 1.000000 | nan |
| 4 | 0.463630 | 0.007511 | 0.001241 | nan | -0.027774 | 4 | yes | 19 | 0.540 | 0.496 | 0.409 | 0.550 | 1.000000 | nan |
| 5 | 0.422880 | 0.040750 | 0.001250 | nan | -0.022975 | 4 | yes | 19 | 0.566 | 0.498 | 0.413 | 0.552 | 1.000000 | nan |
| 6 | 0.349514 | 0.073366 | 0.001239 | nan | 0.710346 | 2 | yes | 19 | 0.533 | 0.497 | 0.243 | 0.549 | 1.000000 | nan |
| 7 | 0.345633 | 0.003881 | 0.000336 | nan | 1.086088 | 2 | yes | 19 | 0.566 | 0.492 | 0.248 | 0.552 | 1.000000 | nan |
| 8 | 0.345626 | 0.000006 | 0.000018 | nan | 1.003382 | 2 | yes | 19 | 0.539 | 0.491 | 0.245 | 0.550 | 1.000000 | nan |
| 9 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.553 | 0.492 | 0.243 | 0.548 | 1.000000 | nan |
