# fenics_custom_l7_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65529` |
| Setup time [s] | `0.265` |
| Total solve time [s] | `1.270` |
| Total Newton iterations | `11` |
| Total linear iterations | `42` |
| Total assembly time [s] | `0.177` |
| Total PC init time [s] | `0.247` |
| Total KSP solve time [s] | `0.206` |
| Total line-search time [s] | `0.581` |
| Final energy | `0.345662` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l7_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l7_np2.log` |

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
| 1 | 1.270 | 11 | 42 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.270` |
| Newton iterations | `11` |
| Linear iterations | `42` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.504032 | 0.161278 | 0.003705 | nan | -0.314446 | 8 | yes | 19 | 0.019 | 0.031 | 0.036 | 0.053 | 1.000000 | nan |
| 2 | 0.503922 | 0.000110 | 0.003704 | nan | 0.106748 | 5 | yes | 19 | 0.016 | 0.022 | 0.023 | 0.053 | 1.000000 | nan |
| 3 | 0.492240 | 0.011681 | 0.003288 | nan | -0.499650 | 5 | yes | 19 | 0.016 | 0.021 | 0.023 | 0.053 | 1.000000 | nan |
| 4 | 0.458331 | 0.033910 | 0.004860 | nan | -0.009980 | 7 | yes | 19 | 0.016 | 0.022 | 0.031 | 0.053 | 1.000000 | nan |
| 5 | 0.404326 | 0.054004 | 0.004790 | nan | 0.465295 | 3 | yes | 19 | 0.016 | 0.022 | 0.015 | 0.053 | 1.000000 | nan |
| 6 | 0.397756 | 0.006570 | 0.002398 | nan | -0.249368 | 3 | yes | 19 | 0.016 | 0.022 | 0.015 | 0.053 | 1.000000 | nan |
| 7 | 0.366880 | 0.030876 | 0.002989 | nan | 0.633838 | 3 | yes | 19 | 0.016 | 0.022 | 0.015 | 0.053 | 1.000000 | nan |
| 8 | 0.347929 | 0.018951 | 0.002187 | nan | 0.565362 | 2 | yes | 19 | 0.016 | 0.022 | 0.011 | 0.053 | 1.000000 | nan |
| 9 | 0.345668 | 0.002260 | 0.000908 | nan | 1.090619 | 2 | yes | 19 | 0.016 | 0.022 | 0.012 | 0.053 | 1.000000 | nan |
| 10 | 0.345662 | 0.000006 | 0.000046 | nan | 1.005482 | 2 | yes | 19 | 0.016 | 0.022 | 0.011 | 0.053 | 1.000000 | nan |
| 11 | 0.345662 | 0.000000 | 0.000000 | 0.000000 | 0.997018 | 2 | yes | 19 | 0.016 | 0.022 | 0.012 | 0.053 | 1.000000 | nan |
