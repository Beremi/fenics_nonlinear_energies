# fenics_custom_l7_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65916` |
| Setup time [s] | `0.179` |
| Total solve time [s] | `0.366` |
| Total Newton iterations | `8` |
| Total linear iterations | `34` |
| Total assembly time [s] | `0.027` |
| Total PC init time [s] | `0.133` |
| Total KSP solve time [s] | `0.078` |
| Total line-search time [s] | `0.100` |
| Final energy | `0.345662` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l7_np8.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l7_np8.log` |

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
| 1 | 0.366 | 8 | 34 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.366` |
| Newton iterations | `8` |
| Linear iterations | `34` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.559007 | 0.106302 | 0.003705 | nan | -0.270559 | 9 | yes | 19 | 0.004 | 0.019 | 0.019 | 0.012 | 1.000000 | nan |
| 2 | 0.510198 | 0.048809 | 0.004045 | nan | 0.331473 | 6 | yes | 19 | 0.003 | 0.016 | 0.013 | 0.012 | 1.000000 | nan |
| 3 | 0.449679 | 0.060520 | 0.002244 | nan | -0.294819 | 5 | yes | 19 | 0.003 | 0.016 | 0.011 | 0.012 | 1.000000 | nan |
| 4 | 0.399470 | 0.050208 | 0.003519 | nan | 0.084589 | 5 | yes | 19 | 0.003 | 0.017 | 0.011 | 0.012 | 1.000000 | nan |
| 5 | 0.359348 | 0.040123 | 0.003469 | nan | 0.353365 | 3 | yes | 19 | 0.003 | 0.017 | 0.007 | 0.015 | 1.000000 | nan |
| 6 | 0.345837 | 0.013511 | 0.002147 | nan | 1.002249 | 2 | yes | 19 | 0.003 | 0.016 | 0.006 | 0.012 | 1.000000 | nan |
| 7 | 0.345662 | 0.000175 | 0.000216 | nan | 1.021443 | 2 | yes | 19 | 0.003 | 0.016 | 0.006 | 0.012 | 1.000000 | nan |
| 8 | 0.345662 | 0.000000 | 0.000002 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.003 | 0.016 | 0.006 | 0.012 | 1.000000 | nan |
