# fenics_custom_l7_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65025` |
| Setup time [s] | `0.166` |
| Total solve time [s] | `1.129` |
| Total Newton iterations | `7` |
| Total linear iterations | `28` |
| Total assembly time [s] | `0.154` |
| Total PC init time [s] | `0.162` |
| Total KSP solve time [s] | `0.178` |
| Total line-search time [s] | `0.569` |
| Final energy | `0.345662` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l7_np1.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l7_np1.log` |

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
| 1 | 1.129 | 7 | 28 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.129` |
| Newton iterations | `7` |
| Linear iterations | `28` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.562421 | 0.102888 | 0.003705 | nan | -0.267161 | 8 | yes | 19 | 0.023 | 0.030 | 0.047 | 0.082 | 1.000000 | nan |
| 2 | 0.508474 | 0.053947 | 0.004055 | nan | 0.327375 | 6 | yes | 19 | 0.022 | 0.022 | 0.035 | 0.081 | 1.000000 | nan |
| 3 | 0.441292 | 0.067182 | 0.002194 | nan | -0.308947 | 5 | yes | 19 | 0.022 | 0.022 | 0.030 | 0.081 | 1.000000 | nan |
| 4 | 0.365185 | 0.076108 | 0.003414 | nan | 0.292653 | 3 | yes | 19 | 0.022 | 0.022 | 0.021 | 0.081 | 1.000000 | nan |
| 5 | 0.346025 | 0.019160 | 0.002535 | nan | 0.937604 | 2 | yes | 19 | 0.022 | 0.022 | 0.015 | 0.082 | 1.000000 | nan |
| 6 | 0.345662 | 0.000363 | 0.000314 | nan | 1.038104 | 2 | yes | 19 | 0.023 | 0.022 | 0.015 | 0.080 | 1.000000 | nan |
| 7 | 0.345662 | 0.000000 | 0.000006 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.022 | 0.022 | 0.015 | 0.081 | 1.000000 | nan |
