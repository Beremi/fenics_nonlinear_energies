# fenics_custom_l5_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `4189` |
| Setup time [s] | `0.026` |
| Total solve time [s] | `0.174` |
| Total Newton iterations | `9` |
| Total linear iterations | `39` |
| Total assembly time [s] | `0.004` |
| Total PC init time [s] | `0.087` |
| Total KSP solve time [s] | `0.058` |
| Total line-search time [s] | `0.022` |
| Final energy | `0.346232` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l5_np8.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l5_np8.log` |

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
| 1 | 0.174 | 9 | 39 | 0.346232 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.174` |
| Newton iterations | `9` |
| Linear iterations | `39` |
| Energy | `0.346232` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.513138 | 0.152332 | 0.014806 | nan | -0.310347 | 7 | yes | 19 | 0.001 | 0.010 | 0.009 | 0.002 | 1.000000 | nan |
| 2 | 0.509992 | 0.003146 | 0.015245 | nan | 0.252000 | 6 | yes | 19 | 0.000 | 0.010 | 0.008 | 0.002 | 1.000000 | nan |
| 3 | 0.463496 | 0.046496 | 0.011269 | nan | -0.499650 | 6 | yes | 19 | 0.000 | 0.010 | 0.009 | 0.002 | 1.000000 | nan |
| 4 | 0.447294 | 0.016202 | 0.016506 | nan | -0.003616 | 7 | yes | 19 | 0.000 | 0.010 | 0.010 | 0.003 | 1.000000 | nan |
| 5 | 0.412482 | 0.034811 | 0.016344 | nan | -0.068694 | 5 | yes | 19 | 0.000 | 0.009 | 0.007 | 0.002 | 1.000000 | nan |
| 6 | 0.348639 | 0.063843 | 0.016868 | nan | 0.724741 | 2 | yes | 19 | 0.000 | 0.010 | 0.004 | 0.002 | 1.000000 | nan |
| 7 | 0.346235 | 0.002404 | 0.004000 | nan | 1.066894 | 2 | yes | 19 | 0.000 | 0.009 | 0.004 | 0.003 | 1.000000 | nan |
| 8 | 0.346232 | 0.000003 | 0.000163 | nan | 1.001816 | 2 | yes | 19 | 0.000 | 0.009 | 0.004 | 0.002 | 1.000000 | nan |
| 9 | 0.346232 | 0.000000 | 0.000000 | 0.000000 | 1.002516 | 2 | yes | 19 | 0.000 | 0.009 | 0.004 | 0.002 | 1.000000 | nan |
