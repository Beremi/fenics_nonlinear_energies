# fenics_custom_l7_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `49665` |
| Free DOFs | `49651` |
| Setup time [s] | `0.182` |
| Total solve time [s] | `0.301` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.007` |
| Total PC init time [s] | `0.201` |
| Total KSP solve time [s] | `0.063` |
| Total line-search time [s] | `0.023` |
| Final energy | `-7.958292` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l7_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l7_np32.log` |

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
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.301 | 6 | 11 | -7.958292 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.301` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.958292` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.314237 | 6.002927 | 2.876352 | nan | 0.358647 | 1 | yes | 9 | 0.001 | 0.034 | 0.007 | 0.004 | 1.000000 | nan |
| 2 | -7.615234 | 2.300997 | 6.280122 | nan | 0.584072 | 2 | yes | 9 | 0.001 | 0.035 | 0.011 | 0.003 | 1.000000 | nan |
| 3 | -7.939439 | 0.324205 | 2.817100 | nan | 0.948817 | 2 | yes | 9 | 0.001 | 0.035 | 0.012 | 0.004 | 1.000000 | nan |
| 4 | -7.957995 | 0.018556 | 0.594718 | nan | 1.088137 | 2 | yes | 9 | 0.001 | 0.033 | 0.011 | 0.004 | 1.000000 | nan |
| 5 | -7.958291 | 0.000296 | 0.060596 | nan | 1.088137 | 2 | yes | 9 | 0.001 | 0.032 | 0.012 | 0.004 | 1.000000 | nan |
| 6 | -7.958292 | 0.000002 | 0.003860 | 0.000097 | 1.002033 | 2 | yes | 9 | 0.001 | 0.032 | 0.010 | 0.004 | 1.000000 | nan |
