# jax_petsc_element_l7_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65025` |
| Setup time [s] | `0.660` |
| Total solve time [s] | `0.307` |
| Total Newton iterations | `8` |
| Total linear iterations | `29` |
| Total assembly time [s] | `0.028` |
| Total PC init time [s] | `0.106` |
| Total KSP solve time [s] | `0.045` |
| Total line-search time [s] | `0.119` |
| Final energy | `0.345662` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l7_np8.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l7_np8.log` |

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
| 1 | 0.307 | 8 | 29 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.307` |
| Newton iterations | `8` |
| Linear iterations | `29` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.479181 | 0.186128 | 0.003705 | nan | -0.453066 | 6 | yes | 19 | 0.004 | 0.015 | 0.009 | 0.015 | 1.000000 | nan |
| 2 | 0.456505 | 0.022676 | 0.004176 | nan | -0.155932 | 6 | yes | 19 | 0.004 | 0.013 | 0.008 | 0.015 | 1.000000 | nan |
| 3 | 0.438405 | 0.018100 | 0.004714 | nan | -0.035105 | 4 | yes | 19 | 0.004 | 0.013 | 0.006 | 0.015 | 1.000000 | nan |
| 4 | 0.422210 | 0.016195 | 0.004789 | nan | -0.010413 | 4 | yes | 19 | 0.003 | 0.013 | 0.006 | 0.015 | 1.000000 | nan |
| 5 | 0.363063 | 0.059147 | 0.004762 | nan | 0.446101 | 3 | yes | 19 | 0.003 | 0.013 | 0.005 | 0.015 | 1.000000 | nan |
| 6 | 0.345940 | 0.017124 | 0.002651 | nan | 1.041769 | 2 | yes | 19 | 0.003 | 0.013 | 0.004 | 0.015 | 1.000000 | nan |
| 7 | 0.345662 | 0.000277 | 0.000274 | nan | 1.023276 | 2 | yes | 19 | 0.003 | 0.013 | 0.004 | 0.015 | 1.000000 | nan |
| 8 | 0.345662 | 0.000000 | 0.000003 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.003 | 0.013 | 0.004 | 0.015 | 1.000000 | nan |
