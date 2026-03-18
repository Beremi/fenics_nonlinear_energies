# fenics_custom_l5_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `5` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `4225` |
| Setup time [s] | `0.027` |
| Total solve time [s] | `0.173` |
| Total Newton iterations | `7` |
| Total linear iterations | `29` |
| Total assembly time [s] | `0.002` |
| Total PC init time [s] | `0.083` |
| Total KSP solve time [s] | `0.051` |
| Total line-search time [s] | `0.017` |
| Final energy | `0.346232` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l5_np16.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l5_np16.log` |

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
| 1 | 0.173 | 7 | 29 | 0.346232 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.173` |
| Newton iterations | `7` |
| Linear iterations | `29` |
| Energy | `0.346232` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.563043 | 0.102427 | 0.014806 | nan | -0.262795 | 9 | yes | 19 | 0.001 | 0.012 | 0.013 | 0.002 | 1.000000 | nan |
| 2 | 0.509472 | 0.053570 | 0.016138 | nan | 0.338104 | 5 | yes | 19 | 0.000 | 0.012 | 0.008 | 0.002 | 1.000000 | nan |
| 3 | 0.432276 | 0.077196 | 0.007885 | nan | -0.439103 | 5 | yes | 19 | 0.000 | 0.011 | 0.008 | 0.002 | 1.000000 | nan |
| 4 | 0.366546 | 0.065730 | 0.013340 | nan | 0.226175 | 4 | yes | 19 | 0.000 | 0.010 | 0.007 | 0.002 | 1.000000 | nan |
| 5 | 0.346799 | 0.019748 | 0.010248 | nan | 0.916844 | 2 | yes | 19 | 0.000 | 0.013 | 0.006 | 0.003 | 1.000000 | nan |
| 6 | 0.346233 | 0.000566 | 0.001555 | nan | 1.050934 | 2 | yes | 19 | 0.000 | 0.015 | 0.006 | 0.003 | 1.000000 | nan |
| 7 | 0.346232 | 0.000000 | 0.000033 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.000 | 0.011 | 0.004 | 0.002 | 1.000000 | nan |
