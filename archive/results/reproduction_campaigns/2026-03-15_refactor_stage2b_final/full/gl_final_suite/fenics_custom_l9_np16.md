# fenics_custom_l9_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1050416` |
| Setup time [s] | `3.459` |
| Total solve time [s] | `3.266` |
| Total Newton iterations | `10` |
| Total linear iterations | `33` |
| Total assembly time [s] | `0.329` |
| Total PC init time [s] | `0.922` |
| Total KSP solve time [s] | `0.717` |
| Total line-search time [s] | `1.184` |
| Final energy | `0.345626` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l9_np16.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l9_np16.log` |

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
| 1 | 3.266 | 10 | 33 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `3.266` |
| Newton iterations | `10` |
| Linear iterations | `33` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.534889 | 0.130411 | 0.000926 | nan | -0.499650 | 5 | yes | 19 | 0.034 | 0.110 | 0.101 | 0.120 | 1.000000 | nan |
| 2 | 0.466206 | 0.068683 | 0.001094 | nan | 0.064262 | 4 | yes | 19 | 0.033 | 0.093 | 0.085 | 0.117 | 1.000000 | nan |
| 3 | 0.420853 | 0.045352 | 0.000976 | nan | 0.386254 | 4 | yes | 19 | 0.032 | 0.092 | 0.083 | 0.117 | 1.000000 | nan |
| 4 | 0.416180 | 0.004673 | 0.000571 | nan | 0.243535 | 3 | yes | 19 | 0.036 | 0.092 | 0.066 | 0.117 | 1.000000 | nan |
| 5 | 0.406188 | 0.009992 | 0.000446 | nan | 0.018543 | 5 | yes | 19 | 0.034 | 0.088 | 0.103 | 0.116 | 1.000000 | nan |
| 6 | 0.377529 | 0.028659 | 0.000500 | nan | 0.741670 | 3 | yes | 19 | 0.032 | 0.088 | 0.066 | 0.137 | 1.000000 | nan |
| 7 | 0.354794 | 0.022735 | 0.000529 | nan | 0.073159 | 3 | yes | 19 | 0.032 | 0.091 | 0.066 | 0.115 | 1.000000 | nan |
| 8 | 0.345927 | 0.008867 | 0.000492 | nan | 0.971193 | 2 | yes | 19 | 0.032 | 0.088 | 0.049 | 0.115 | 1.000000 | nan |
| 9 | 0.345627 | 0.000301 | 0.000074 | nan | 1.055032 | 2 | yes | 19 | 0.032 | 0.089 | 0.050 | 0.115 | 1.000000 | nan |
| 10 | 0.345626 | 0.000000 | 0.000001 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.032 | 0.090 | 0.049 | 0.115 | 1.000000 | nan |
