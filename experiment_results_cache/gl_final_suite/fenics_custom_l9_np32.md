# fenics_custom_l9_np32

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `32` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1050285` |
| Setup time [s] | `2.855` |
| Total solve time [s] | `1.734` |
| Total Newton iterations | `8` |
| Total linear iterations | `37` |
| Total assembly time [s] | `0.136` |
| Total PC init time [s] | `0.487` |
| Total KSP solve time [s] | `0.517` |
| Total line-search time [s] | `0.487` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np32.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np32.log` |

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
| 1 | 1.734 | 8 | 37 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `1.734` |
| Newton iterations | `8` |
| Linear iterations | `37` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.566512 | 0.098788 | 0.000926 | nan | -0.262363 | 11 | yes | 19 | 0.018 | 0.074 | 0.148 | 0.063 | 1.000000 | nan |
| 2 | 0.506731 | 0.059780 | 0.001020 | nan | 0.332606 | 5 | yes | 19 | 0.017 | 0.057 | 0.059 | 0.063 | 1.000000 | nan |
| 3 | 0.426415 | 0.080316 | 0.000526 | nan | -0.499650 | 5 | yes | 19 | 0.017 | 0.062 | 0.057 | 0.062 | 1.000000 | nan |
| 4 | 0.384520 | 0.041895 | 0.000881 | nan | 0.091653 | 6 | yes | 19 | 0.017 | 0.058 | 0.090 | 0.063 | 1.000000 | nan |
| 5 | 0.352748 | 0.031772 | 0.000826 | nan | 0.424209 | 4 | yes | 19 | 0.017 | 0.058 | 0.053 | 0.056 | 1.000000 | nan |
| 6 | 0.345672 | 0.007076 | 0.000434 | nan | 1.027374 | 2 | yes | 19 | 0.017 | 0.059 | 0.029 | 0.058 | 1.000000 | nan |
| 7 | 0.345626 | 0.000046 | 0.000028 | nan | 1.010713 | 2 | yes | 19 | 0.017 | 0.060 | 0.032 | 0.062 | 1.000000 | nan |
| 8 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.017 | 0.060 | 0.047 | 0.061 | 1.000000 | nan |
