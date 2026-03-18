# fenics_custom_l5_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `3969` |
| Setup time [s] | `0.018` |
| Total solve time [s] | `0.108` |
| Total Newton iterations | `8` |
| Total linear iterations | `33` |
| Total assembly time [s] | `0.014` |
| Total PC init time [s] | `0.018` |
| Total KSP solve time [s] | `0.016` |
| Total line-search time [s] | `0.054` |
| Final energy | `0.346232` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l5_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l5_np1.log` |

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
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.108 | 8 | 33 | 0.346232 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.108` |
| Newton iterations | `8` |
| Linear iterations | `33` |
| Energy | `0.346232` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.562393 | 0.103077 | 0.014806 | nan | -0.261930 | 9 | yes | 19 | 0.002 | 0.003 | 0.004 | 0.007 | 1.000000 | nan |
| 2 | 0.512419 | 0.049974 | 0.016134 | nan | 0.327375 | 6 | yes | 19 | 0.002 | 0.002 | 0.003 | 0.007 | 1.000000 | nan |
| 3 | 0.446857 | 0.065562 | 0.008905 | nan | -0.315578 | 5 | yes | 19 | 0.002 | 0.002 | 0.002 | 0.007 | 1.000000 | nan |
| 4 | 0.402837 | 0.044020 | 0.013834 | nan | 0.104648 | 4 | yes | 19 | 0.002 | 0.002 | 0.002 | 0.007 | 1.000000 | nan |
| 5 | 0.357381 | 0.045456 | 0.013333 | nan | 0.357731 | 3 | yes | 19 | 0.002 | 0.002 | 0.001 | 0.007 | 1.000000 | nan |
| 6 | 0.346337 | 0.011044 | 0.007663 | nan | 1.013246 | 2 | yes | 19 | 0.002 | 0.002 | 0.001 | 0.007 | 1.000000 | nan |
| 7 | 0.346232 | 0.000105 | 0.000674 | nan | 1.013679 | 2 | yes | 19 | 0.002 | 0.002 | 0.001 | 0.007 | 1.000000 | nan |
| 8 | 0.346232 | 0.000000 | 0.000004 | 0.000000 | 1.000684 | 2 | yes | 19 | 0.002 | 0.002 | 0.001 | 0.007 | 1.000000 | nan |
