# fenics_custom_l9_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `784385` |
| Setup time [s] | `1.999` |
| Total solve time [s] | `10.833` |
| Total Newton iterations | `5` |
| Total linear iterations | `10` |
| Total assembly time [s] | `1.155` |
| Total PC init time [s] | `3.421` |
| Total KSP solve time [s] | `1.408` |
| Total line-search time [s] | `4.199` |
| Final energy | `-7.960005` |
| Raw JSON | `experiment_results_cache/plaplace_final_suite/fenics_custom_l9_np1.json` |
| Raw log | `experiment_results_cache/plaplace_final_suite/fenics_custom_l9_np1.log` |

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
| 1 | 10.833 | 5 | 10 | -7.960005 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `10.833` |
| Newton iterations | `5` |
| Linear iterations | `10` |
| Energy | `-7.960005` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.132170 | 38.283916 | 44.793319 | nan | 1.678307 | 2 | yes | 9 | 0.244 | 0.802 | 0.298 | 0.839 | 1.000000 | nan |
| 2 | -7.544088 | 2.411918 | 8.641301 | nan | 0.723392 | 2 | yes | 9 | 0.226 | 0.780 | 0.292 | 0.839 | 1.000000 | nan |
| 3 | -7.952877 | 0.408789 | 3.153469 | nan | 1.141353 | 2 | yes | 9 | 0.230 | 0.612 | 0.264 | 0.839 | 1.000000 | nan |
| 4 | -7.959905 | 0.007028 | 0.360585 | nan | 1.141353 | 2 | yes | 9 | 0.227 | 0.607 | 0.275 | 0.839 | 1.000000 | nan |
| 5 | -7.960005 | 0.000099 | 0.036497 | 0.002931 | 1.088137 | 2 | yes | 9 | 0.227 | 0.620 | 0.279 | 0.843 | 1.000000 | nan |
