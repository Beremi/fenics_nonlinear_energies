# jax_petsc_local_sfd_l7_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `48641` |
| Setup time [s] | `0.684` |
| Total solve time [s] | `0.434` |
| Total Newton iterations | `7` |
| Total linear iterations | `13` |
| Total assembly time [s] | `0.149` |
| Total PC init time [s] | `0.167` |
| Total KSP solve time [s] | `0.042` |
| Total line-search time [s] | `0.069` |
| Final energy | `-7.958292` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l7_np16.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_local_sfd_l7_np16.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `cg` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.1` |
| `ksp_max_it` | `30` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.1` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `sfd_local` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.434 | 7 | 13 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.434` |
| Newton iterations | `7` |
| Linear iterations | `13` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.354265 | 521138.952778 | 28062.385991 | nan | 1.956948 | 2 | yes | 9 | 0.140 | 0.025 | 0.006 | 0.010 | 1.000000 | nan |
| 2 | -3.968981 | 9.323245 | 13.014530 | nan | 0.862712 | 2 | yes | 9 | 0.002 | 0.024 | 0.006 | 0.010 | 1.000000 | nan |
| 3 | -7.352551 | 3.383570 | 9.194890 | nan | 0.809497 | 2 | yes | 9 | 0.001 | 0.024 | 0.006 | 0.010 | 1.000000 | nan |
| 4 | -7.928175 | 0.575624 | 3.123264 | nan | 1.002033 | 2 | yes | 9 | 0.002 | 0.025 | 0.007 | 0.010 | 1.000000 | nan |
| 5 | -7.957664 | 0.029489 | 0.641227 | nan | 1.055248 | 2 | yes | 9 | 0.001 | 0.024 | 0.006 | 0.010 | 1.000000 | nan |
| 6 | -7.958285 | 0.000620 | 0.081219 | nan | 1.088137 | 1 | yes | 9 | 0.001 | 0.022 | 0.004 | 0.010 | 1.000000 | nan |
| 7 | -7.958292 | 0.000008 | 0.007117 | 0.000369 | 1.055248 | 2 | yes | 9 | 0.001 | 0.024 | 0.006 | 0.010 | 1.000000 | nan |
