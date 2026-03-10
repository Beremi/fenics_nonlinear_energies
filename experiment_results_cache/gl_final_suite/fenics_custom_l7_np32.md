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
| Total DOFs | `66049` |
| Free DOFs | `66049` |
| Setup time [s] | `0.185` |
| Total solve time [s] | `0.408` |
| Total Newton iterations | `8` |
| Total linear iterations | `35` |
| Total assembly time [s] | `0.012` |
| Total PC init time [s] | `0.204` |
| Total KSP solve time [s] | `0.122` |
| Total line-search time [s] | `0.046` |
| Final energy | `0.345662` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l7_np32.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l7_np32.log` |

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
| 1 | 0.408 | 8 | 35 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.408` |
| Newton iterations | `8` |
| Linear iterations | `35` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.567164 | 0.098146 | 0.003705 | nan | -0.260963 | 11 | yes | 19 | 0.002 | 0.025 | 0.033 | 0.006 | 1.000000 | nan |
| 2 | 0.508733 | 0.058431 | 0.004075 | nan | 0.309314 | 6 | yes | 19 | 0.002 | 0.026 | 0.021 | 0.006 | 1.000000 | nan |
| 3 | 0.442429 | 0.066304 | 0.002233 | nan | -0.297785 | 5 | yes | 19 | 0.001 | 0.027 | 0.017 | 0.006 | 1.000000 | nan |
| 4 | 0.372715 | 0.069714 | 0.003433 | nan | 0.242135 | 4 | yes | 19 | 0.001 | 0.026 | 0.014 | 0.006 | 1.000000 | nan |
| 5 | 0.347454 | 0.025261 | 0.002833 | nan | 0.773158 | 3 | yes | 19 | 0.001 | 0.024 | 0.012 | 0.006 | 1.000000 | nan |
| 6 | 0.345665 | 0.001789 | 0.000733 | nan | 1.065329 | 2 | yes | 19 | 0.002 | 0.024 | 0.008 | 0.006 | 1.000000 | nan |
| 7 | 0.345662 | 0.000003 | 0.000027 | nan | 1.001816 | 2 | yes | 19 | 0.001 | 0.026 | 0.008 | 0.005 | 1.000000 | nan |
| 8 | 0.345662 | 0.000000 | 0.000000 | 0.000000 | 0.999551 | 2 | yes | 19 | 0.002 | 0.026 | 0.009 | 0.005 | 1.000000 | nan |
