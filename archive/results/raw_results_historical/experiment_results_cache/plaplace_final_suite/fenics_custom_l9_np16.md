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
| Total DOFs | `788481` |
| Free DOFs | `788361` |
| Setup time [s] | `2.092` |
| Total solve time [s] | `1.313` |
| Total Newton iterations | `6` |
| Total linear iterations | `12` |
| Total assembly time [s] | `0.116` |
| Total PC init time [s] | `0.540` |
| Total KSP solve time [s] | `0.227` |
| Total line-search time [s] | `0.373` |
| Final energy | `-7.960006` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l9_np16.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l9_np16.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `cg` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.1` |
| `ksp_max_it` | `30` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.1` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 1.313 | 6 | 12 | -7.960006 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.313` |
| Newton iterations | `6` |
| Linear iterations | `12` |
| Energy | `-7.960006` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.168989 | 38.323447 | 44.819483 | nan | 1.678307 | 2 | yes | 9 | 0.020 | 0.098 | 0.038 | 0.060 | 1.000000 | nan |
| 2 | -7.551918 | 2.382929 | 8.546068 | nan | 0.723392 | 2 | yes | 9 | 0.019 | 0.092 | 0.038 | 0.067 | 1.000000 | nan |
| 3 | -7.953110 | 0.401192 | 3.106440 | nan | 1.141353 | 2 | yes | 9 | 0.019 | 0.080 | 0.036 | 0.067 | 1.000000 | nan |
| 4 | -7.959902 | 0.006792 | 0.345394 | nan | 1.141353 | 2 | yes | 9 | 0.020 | 0.090 | 0.039 | 0.060 | 1.000000 | nan |
| 5 | -7.960005 | 0.000102 | 0.035923 | nan | 1.088137 | 2 | yes | 9 | 0.019 | 0.091 | 0.039 | 0.060 | 1.000000 | nan |
| 6 | -7.960006 | 0.000001 | 0.003133 | 0.000172 | 1.055248 | 2 | yes | 9 | 0.019 | 0.089 | 0.038 | 0.060 | 1.000000 | nan |
