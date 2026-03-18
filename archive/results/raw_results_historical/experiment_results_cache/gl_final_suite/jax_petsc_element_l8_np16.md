# jax_petsc_element_l8_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `8` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `263169` |
| Free DOFs | `261121` |
| Setup time [s] | `0.839` |
| Total solve time [s] | `0.799` |
| Total Newton iterations | `7` |
| Total linear iterations | `30` |
| Total assembly time [s] | `0.073` |
| Total PC init time [s] | `0.190` |
| Total KSP solve time [s] | `0.109` |
| Total line-search time [s] | `0.399` |
| Final energy | `0.345634` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l8_np16.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l8_np16.log` |

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
| 1 | 0.799 | 7 | 30 | 0.345634 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.799` |
| Newton iterations | `7` |
| Linear iterations | `30` |
| Energy | `0.345634` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.548347 | 0.116954 | 0.001852 | nan | -0.283822 | 8 | yes | 19 | 0.010 | 0.030 | 0.027 | 0.058 | 1.000000 | nan |
| 2 | 0.509876 | 0.038472 | 0.002003 | nan | 0.357731 | 6 | yes | 19 | 0.011 | 0.028 | 0.021 | 0.057 | 1.000000 | nan |
| 3 | 0.441399 | 0.068477 | 0.001115 | nan | -0.352566 | 5 | yes | 19 | 0.010 | 0.027 | 0.018 | 0.057 | 1.000000 | nan |
| 4 | 0.364751 | 0.076648 | 0.001774 | nan | 0.274592 | 4 | yes | 19 | 0.010 | 0.026 | 0.014 | 0.057 | 1.000000 | nan |
| 5 | 0.345860 | 0.018891 | 0.001271 | nan | 0.965694 | 3 | yes | 19 | 0.011 | 0.027 | 0.012 | 0.058 | 1.000000 | nan |
| 6 | 0.345634 | 0.000227 | 0.000126 | nan | 1.019177 | 2 | yes | 19 | 0.011 | 0.026 | 0.009 | 0.057 | 1.000000 | nan |
| 7 | 0.345634 | 0.000000 | 0.000001 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.010 | 0.026 | 0.008 | 0.057 | 1.000000 | nan |
