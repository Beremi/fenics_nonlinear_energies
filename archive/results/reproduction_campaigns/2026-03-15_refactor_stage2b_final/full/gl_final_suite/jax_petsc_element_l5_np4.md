# jax_petsc_element_l5_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `3969` |
| Setup time [s] | `0.510` |
| Total solve time [s] | `0.187` |
| Total Newton iterations | `11` |
| Total linear iterations | `41` |
| Total assembly time [s] | `0.005` |
| Total PC init time [s] | `0.054` |
| Total KSP solve time [s] | `0.027` |
| Total line-search time [s] | `0.096` |
| Final energy | `0.346231` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_element_l5_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_element_l5_np4.log` |

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
| 1 | 0.187 | 11 | 41 | 0.346231 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.187` |
| Newton iterations | `11` |
| Linear iterations | `41` |
| Energy | `0.346231` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.606747 | 0.058723 | 0.014806 | nan | -0.499650 | 1 | yes | 19 | 0.001 | 0.006 | 0.001 | 0.007 | 1.000000 | nan |
| 2 | 0.553274 | 0.053473 | 0.015373 | nan | -0.199983 | 9 | yes | 19 | 0.000 | 0.005 | 0.006 | 0.007 | 1.000000 | nan |
| 3 | 0.493498 | 0.059775 | 0.016883 | nan | 0.189887 | 6 | yes | 19 | 0.000 | 0.005 | 0.004 | 0.007 | 1.000000 | nan |
| 4 | 0.483835 | 0.009663 | 0.012014 | nan | -0.499650 | 5 | yes | 19 | 0.000 | 0.005 | 0.003 | 0.007 | 1.000000 | nan |
| 5 | 0.462616 | 0.021218 | 0.017445 | nan | -0.026641 | 5 | yes | 19 | 0.000 | 0.005 | 0.003 | 0.026 | 1.000000 | nan |
| 6 | 0.441996 | 0.020620 | 0.017772 | nan | 0.129773 | 4 | yes | 19 | 0.000 | 0.005 | 0.003 | 0.007 | 1.000000 | nan |
| 7 | 0.396030 | 0.045966 | 0.015825 | nan | 0.379190 | 3 | yes | 19 | 0.000 | 0.005 | 0.002 | 0.007 | 1.000000 | nan |
| 8 | 0.350562 | 0.045468 | 0.012705 | nan | 1.023708 | 2 | yes | 19 | 0.000 | 0.005 | 0.002 | 0.007 | 1.000000 | nan |
| 9 | 0.346318 | 0.004245 | 0.005261 | nan | 0.847668 | 2 | yes | 19 | 0.000 | 0.004 | 0.001 | 0.007 | 1.000000 | nan |
| 10 | 0.346231 | 0.000086 | 0.000652 | nan | 1.034705 | 2 | yes | 19 | 0.000 | 0.005 | 0.002 | 0.007 | 1.000000 | nan |
| 11 | 0.346231 | 0.000000 | 0.000011 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.000 | 0.005 | 0.002 | 0.007 | 1.000000 | nan |
