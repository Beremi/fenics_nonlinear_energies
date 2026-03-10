# jax_petsc_element_l5_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `3969` |
| Setup time [s] | `0.576` |
| Total solve time [s] | `0.179` |
| Total Newton iterations | `8` |
| Total linear iterations | `38` |
| Total assembly time [s] | `0.002` |
| Total PC init time [s] | `0.069` |
| Total KSP solve time [s] | `0.045` |
| Total line-search time [s] | `0.060` |
| Final energy | `0.346231` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l5_np16.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l5_np16.log` |

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
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.179 | 8 | 38 | 0.346231 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.179` |
| Newton iterations | `8` |
| Linear iterations | `38` |
| Energy | `0.346231` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.563884 | 0.101586 | 0.014806 | nan | -0.261930 | 12 | yes | 19 | 0.000 | 0.011 | 0.014 | 0.008 | 1.000000 | nan |
| 2 | 0.511371 | 0.052513 | 0.016150 | nan | 0.318910 | 6 | yes | 19 | 0.000 | 0.009 | 0.007 | 0.007 | 1.000000 | nan |
| 3 | 0.449814 | 0.061557 | 0.008932 | nan | -0.276058 | 6 | yes | 19 | 0.000 | 0.008 | 0.006 | 0.007 | 1.000000 | nan |
| 4 | 0.403000 | 0.046814 | 0.013761 | nan | 0.089387 | 5 | yes | 19 | 0.000 | 0.008 | 0.006 | 0.007 | 1.000000 | nan |
| 5 | 0.359960 | 0.043040 | 0.013857 | nan | 0.347001 | 3 | yes | 19 | 0.000 | 0.008 | 0.004 | 0.007 | 1.000000 | nan |
| 6 | 0.346409 | 0.013552 | 0.008564 | nan | 1.007315 | 2 | yes | 19 | 0.000 | 0.008 | 0.003 | 0.007 | 1.000000 | nan |
| 7 | 0.346231 | 0.000177 | 0.000859 | nan | 1.021443 | 2 | yes | 19 | 0.000 | 0.008 | 0.003 | 0.007 | 1.000000 | nan |
| 8 | 0.346231 | 0.000000 | 0.000008 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.000 | 0.008 | 0.003 | 0.007 | 1.000000 | nan |
