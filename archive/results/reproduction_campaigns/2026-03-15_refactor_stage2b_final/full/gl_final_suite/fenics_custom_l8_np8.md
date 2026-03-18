# fenics_custom_l8_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `262977` |
| Setup time [s] | `0.828` |
| Total solve time [s] | `2.048` |
| Total Newton iterations | `13` |
| Total linear iterations | `48` |
| Total assembly time [s] | `0.223` |
| Total PC init time [s] | `0.586` |
| Total KSP solve time [s] | `0.434` |
| Total line-search time [s] | `0.730` |
| Final energy | `0.345634` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l8_np8.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l8_np8.log` |

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
| 1 | 2.048 | 13 | 48 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `2.048` |
| Newton iterations | `13` |
| Linear iterations | `48` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.500603 | 0.164699 | 0.001852 | nan | -0.368094 | 6 | yes | 19 | 0.018 | 0.054 | 0.050 | 0.056 | 1.000000 | nan |
| 2 | 0.496684 | 0.003919 | 0.002109 | nan | -0.266461 | 6 | yes | 19 | 0.017 | 0.045 | 0.051 | 0.056 | 1.000000 | nan |
| 3 | 0.495801 | 0.000884 | 0.002632 | nan | -0.017044 | 6 | yes | 19 | 0.016 | 0.044 | 0.050 | 0.056 | 1.000000 | nan |
| 4 | 0.479110 | 0.016690 | 0.002655 | nan | -0.012513 | 5 | yes | 19 | 0.016 | 0.045 | 0.043 | 0.057 | 1.000000 | nan |
| 5 | 0.460142 | 0.018968 | 0.002642 | nan | -0.028906 | 4 | yes | 19 | 0.018 | 0.046 | 0.036 | 0.056 | 1.000000 | nan |
| 6 | 0.422833 | 0.037309 | 0.002676 | nan | 0.370993 | 3 | yes | 19 | 0.018 | 0.044 | 0.029 | 0.056 | 1.000000 | nan |
| 7 | 0.421454 | 0.001379 | 0.001630 | nan | 0.214312 | 4 | yes | 19 | 0.018 | 0.043 | 0.035 | 0.056 | 1.000000 | nan |
| 8 | 0.413776 | 0.007678 | 0.001283 | nan | -0.126708 | 3 | yes | 19 | 0.018 | 0.045 | 0.028 | 0.056 | 1.000000 | nan |
| 9 | 0.383869 | 0.029907 | 0.001477 | nan | 0.692552 | 3 | yes | 19 | 0.016 | 0.045 | 0.028 | 0.056 | 1.000000 | nan |
| 10 | 0.357131 | 0.026739 | 0.000988 | nan | 0.558031 | 2 | yes | 19 | 0.016 | 0.044 | 0.021 | 0.056 | 1.000000 | nan |
| 11 | 0.346787 | 0.010344 | 0.000862 | nan | 0.575824 | 2 | yes | 19 | 0.016 | 0.044 | 0.021 | 0.057 | 1.000000 | nan |
| 12 | 0.345634 | 0.001152 | 0.000300 | nan | 1.094285 | 2 | yes | 19 | 0.017 | 0.044 | 0.021 | 0.056 | 1.000000 | nan |
| 13 | 0.345634 | 0.000001 | 0.000009 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.016 | 0.045 | 0.021 | 0.056 | 1.000000 | nan |
