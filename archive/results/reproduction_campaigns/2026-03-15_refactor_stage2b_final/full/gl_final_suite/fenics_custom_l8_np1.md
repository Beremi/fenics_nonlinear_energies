# fenics_custom_l8_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `261121` |
| Setup time [s] | `0.780` |
| Total solve time [s] | `5.461` |
| Total Newton iterations | `7` |
| Total linear iterations | `20` |
| Total assembly time [s] | `0.759` |
| Total PC init time [s] | `0.815` |
| Total KSP solve time [s] | `0.680` |
| Total line-search time [s] | `2.917` |
| Final energy | `0.345634` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l8_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l8_np1.log` |

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
| 1 | 5.461 | 7 | 20 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `5.461` |
| Newton iterations | `7` |
| Linear iterations | `20` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.505217 | 0.160085 | 0.001852 | nan | -0.499650 | 3 | yes | 19 | 0.114 | 0.135 | 0.101 | 0.413 | 1.000000 | nan |
| 2 | 0.489236 | 0.015980 | 0.002069 | nan | 0.000050 | 6 | yes | 19 | 0.107 | 0.115 | 0.178 | 0.412 | 1.000000 | nan |
| 3 | 0.404011 | 0.085225 | 0.002051 | nan | 0.114245 | 3 | yes | 19 | 0.108 | 0.113 | 0.101 | 0.412 | 1.000000 | nan |
| 4 | 0.347794 | 0.056217 | 0.001771 | nan | 0.693685 | 2 | yes | 19 | 0.107 | 0.113 | 0.075 | 0.411 | 1.000000 | nan |
| 5 | 0.345639 | 0.002155 | 0.000440 | nan | 1.087654 | 2 | yes | 19 | 0.107 | 0.113 | 0.075 | 0.426 | 1.000000 | nan |
| 6 | 0.345634 | 0.000006 | 0.000026 | nan | 1.003382 | 2 | yes | 19 | 0.108 | 0.113 | 0.075 | 0.423 | 1.000000 | nan |
| 7 | 0.345634 | 0.000000 | 0.000000 | 0.000000 | 1.004515 | 2 | yes | 19 | 0.109 | 0.113 | 0.075 | 0.419 | 1.000000 | nan |
