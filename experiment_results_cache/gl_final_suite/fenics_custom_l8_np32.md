# fenics_custom_l8_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `8` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `263169` |
| Setup time [s] | `0.693` |
| Total solve time [s] | `0.664` |
| Total Newton iterations | `8` |
| Total linear iterations | `35` |
| Total assembly time [s] | `0.037` |
| Total PC init time [s] | `0.291` |
| Total KSP solve time [s] | `0.177` |
| Total line-search time [s] | `0.129` |
| Final energy | `0.345634` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l8_np32.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l8_np32.log` |

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
| 1 | 0.664 | 8 | 35 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.664` |
| Newton iterations | `8` |
| Linear iterations | `35` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.558338 | 0.106964 | 0.001852 | nan | -0.274925 | 9 | yes | 19 | 0.007 | 0.041 | 0.045 | 0.017 | 1.000000 | nan |
| 2 | 0.509282 | 0.049056 | 0.002033 | nan | 0.340370 | 6 | yes | 19 | 0.004 | 0.039 | 0.027 | 0.016 | 1.000000 | nan |
| 3 | 0.445022 | 0.064259 | 0.001114 | nan | -0.285655 | 5 | yes | 19 | 0.005 | 0.036 | 0.023 | 0.015 | 1.000000 | nan |
| 4 | 0.393947 | 0.051075 | 0.001741 | nan | 0.131606 | 6 | yes | 19 | 0.004 | 0.036 | 0.028 | 0.017 | 1.000000 | nan |
| 5 | 0.354081 | 0.039866 | 0.001674 | nan | 0.432673 | 3 | yes | 19 | 0.005 | 0.036 | 0.015 | 0.017 | 1.000000 | nan |
| 6 | 0.345702 | 0.008379 | 0.000858 | nan | 1.034705 | 2 | yes | 19 | 0.004 | 0.035 | 0.013 | 0.015 | 1.000000 | nan |
| 7 | 0.345634 | 0.000069 | 0.000068 | nan | 1.012546 | 2 | yes | 19 | 0.004 | 0.034 | 0.013 | 0.015 | 1.000000 | nan |
| 8 | 0.345634 | 0.000000 | 0.000000 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.004 | 0.034 | 0.013 | 0.016 | 1.000000 | nan |
