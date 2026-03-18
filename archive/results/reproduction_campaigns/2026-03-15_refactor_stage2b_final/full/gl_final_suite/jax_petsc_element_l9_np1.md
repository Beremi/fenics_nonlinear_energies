# jax_petsc_element_l9_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1046529` |
| Setup time [s] | `13.367` |
| Total solve time [s] | `15.610` |
| Total Newton iterations | `9` |
| Total linear iterations | `33` |
| Total assembly time [s] | `2.290` |
| Total PC init time [s] | `4.477` |
| Total KSP solve time [s] | `3.438` |
| Total line-search time [s] | `4.931` |
| Final energy | `0.345626` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_element_l9_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_element_l9_np1.log` |

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
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 15.610 | 9 | 33 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `15.610` |
| Newton iterations | `9` |
| Linear iterations | `33` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.495938 | 0.169361 | 0.000926 | nan | -0.344102 | 7 | yes | 19 | 0.255 | 0.555 | 0.651 | 0.548 | 1.000000 | nan |
| 2 | 0.488637 | 0.007301 | 0.000906 | nan | -0.396884 | 5 | yes | 19 | 0.256 | 0.492 | 0.490 | 0.549 | 1.000000 | nan |
| 3 | 0.471141 | 0.017496 | 0.001247 | nan | -0.014346 | 5 | yes | 19 | 0.256 | 0.496 | 0.498 | 0.549 | 1.000000 | nan |
| 4 | 0.463630 | 0.007511 | 0.001241 | nan | -0.027774 | 4 | yes | 19 | 0.253 | 0.486 | 0.408 | 0.548 | 1.000000 | nan |
| 5 | 0.422880 | 0.040750 | 0.001250 | nan | -0.022975 | 4 | yes | 19 | 0.253 | 0.485 | 0.410 | 0.547 | 1.000000 | nan |
| 6 | 0.349514 | 0.073366 | 0.001239 | nan | 0.710346 | 2 | yes | 19 | 0.256 | 0.490 | 0.245 | 0.548 | 1.000000 | nan |
| 7 | 0.345633 | 0.003881 | 0.000336 | nan | 1.086088 | 2 | yes | 19 | 0.252 | 0.488 | 0.244 | 0.546 | 1.000000 | nan |
| 8 | 0.345626 | 0.000006 | 0.000018 | nan | 1.003382 | 2 | yes | 19 | 0.256 | 0.488 | 0.245 | 0.548 | 1.000000 | nan |
| 9 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.253 | 0.498 | 0.248 | 0.548 | 1.000000 | nan |
