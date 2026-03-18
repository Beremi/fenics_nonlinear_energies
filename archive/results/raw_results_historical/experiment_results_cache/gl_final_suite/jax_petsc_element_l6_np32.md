# jax_petsc_element_l6_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16129` |
| Setup time [s] | `0.647` |
| Total solve time [s] | `0.338` |
| Total Newton iterations | `8` |
| Total linear iterations | `35` |
| Total assembly time [s] | `0.005` |
| Total PC init time [s] | `0.152` |
| Total KSP solve time [s] | `0.083` |
| Total line-search time [s] | `0.092` |
| Final energy | `0.345777` |
| Raw JSON | `experiment_results_cache/gl_final_suite/jax_petsc_element_l6_np32.json` |
| Raw log | `experiment_results_cache/gl_final_suite/jax_petsc_element_l6_np32.log` |

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
| 1 | 0.338 | 8 | 35 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.338` |
| Newton iterations | `8` |
| Linear iterations | `35` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.564960 | 0.100381 | 0.007408 | nan | -0.266028 | 10 | yes | 19 | 0.001 | 0.023 | 0.021 | 0.013 | 1.000000 | nan |
| 2 | 0.510095 | 0.054865 | 0.008266 | nan | 0.329207 | 6 | yes | 19 | 0.001 | 0.020 | 0.014 | 0.012 | 1.000000 | nan |
| 3 | 0.448142 | 0.061953 | 0.004492 | nan | -0.257997 | 5 | yes | 19 | 0.001 | 0.019 | 0.013 | 0.011 | 1.000000 | nan |
| 4 | 0.379524 | 0.068618 | 0.006953 | nan | 0.176625 | 5 | yes | 19 | 0.001 | 0.020 | 0.011 | 0.013 | 1.000000 | nan |
| 5 | 0.352042 | 0.027482 | 0.006305 | nan | 0.543203 | 3 | yes | 19 | 0.001 | 0.017 | 0.008 | 0.011 | 1.000000 | nan |
| 6 | 0.345824 | 0.006218 | 0.002903 | nan | 1.047268 | 2 | yes | 19 | 0.000 | 0.019 | 0.006 | 0.011 | 1.000000 | nan |
| 7 | 0.345777 | 0.000048 | 0.000212 | nan | 1.016212 | 2 | yes | 19 | 0.001 | 0.017 | 0.005 | 0.011 | 1.000000 | nan |
| 8 | 0.345777 | 0.000000 | 0.000002 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.001 | 0.018 | 0.006 | 0.011 | 1.000000 | nan |
