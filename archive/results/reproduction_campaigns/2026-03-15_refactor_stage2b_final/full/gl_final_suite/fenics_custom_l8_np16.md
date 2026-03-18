# fenics_custom_l8_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `263084` |
| Setup time [s] | `0.774` |
| Total solve time [s] | `1.048` |
| Total Newton iterations | `10` |
| Total linear iterations | `35` |
| Total assembly time [s] | `0.091` |
| Total PC init time [s] | `0.388` |
| Total KSP solve time [s] | `0.245` |
| Total line-search time [s] | `0.292` |
| Final energy | `0.345634` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l8_np16.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l8_np16.log` |

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
| 1 | 1.048 | 10 | 35 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.048` |
| Newton iterations | `10` |
| Linear iterations | `35` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.495793 | 0.169509 | 0.001852 | nan | -0.340703 | 6 | yes | 19 | 0.011 | 0.044 | 0.039 | 0.029 | 1.000000 | nan |
| 2 | 0.488688 | 0.007104 | 0.001951 | nan | -0.371327 | 6 | yes | 19 | 0.009 | 0.039 | 0.039 | 0.029 | 1.000000 | nan |
| 3 | 0.465547 | 0.023141 | 0.002615 | nan | 0.038437 | 5 | yes | 19 | 0.009 | 0.039 | 0.033 | 0.029 | 1.000000 | nan |
| 4 | 0.400038 | 0.065510 | 0.002434 | nan | 0.491120 | 4 | yes | 19 | 0.009 | 0.038 | 0.027 | 0.029 | 1.000000 | nan |
| 5 | 0.386808 | 0.013229 | 0.001142 | nan | -0.060663 | 3 | yes | 19 | 0.009 | 0.038 | 0.022 | 0.029 | 1.000000 | nan |
| 6 | 0.359787 | 0.027021 | 0.001292 | nan | 0.490687 | 3 | yes | 19 | 0.009 | 0.039 | 0.022 | 0.029 | 1.000000 | nan |
| 7 | 0.347137 | 0.012650 | 0.001044 | nan | 0.600084 | 2 | yes | 19 | 0.009 | 0.037 | 0.015 | 0.029 | 1.000000 | nan |
| 8 | 0.345636 | 0.001501 | 0.000368 | nan | 1.078757 | 2 | yes | 19 | 0.009 | 0.040 | 0.016 | 0.029 | 1.000000 | nan |
| 9 | 0.345634 | 0.000003 | 0.000018 | nan | 1.001816 | 2 | yes | 19 | 0.009 | 0.038 | 0.016 | 0.029 | 1.000000 | nan |
| 10 | 0.345634 | 0.000000 | 0.000000 | 0.000000 | 1.005482 | 2 | yes | 19 | 0.009 | 0.037 | 0.016 | 0.029 | 1.000000 | nan |
