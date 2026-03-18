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
| Total DOFs | `1050625` |
| Free DOFs | `1050416` |
| Setup time [s] | `2.975` |
| Total solve time [s] | `2.860` |
| Total Newton iterations | `10` |
| Total linear iterations | `33` |
| Total assembly time [s] | `0.275` |
| Total PC init time [s] | `0.781` |
| Total KSP solve time [s] | `0.676` |
| Total line-search time [s] | `1.016` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np16.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np16.log` |

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
| 1 | 2.860 | 10 | 33 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `2.860` |
| Newton iterations | `10` |
| Linear iterations | `33` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.534889 | 0.130411 | 0.000926 | nan | -0.499650 | 5 | yes | 19 | 0.030 | 0.092 | 0.095 | 0.099 | 1.000000 | nan |
| 2 | 0.466206 | 0.068683 | 0.001094 | nan | 0.064262 | 4 | yes | 19 | 0.027 | 0.078 | 0.079 | 0.098 | 1.000000 | nan |
| 3 | 0.420853 | 0.045352 | 0.000976 | nan | 0.386254 | 4 | yes | 19 | 0.027 | 0.078 | 0.082 | 0.098 | 1.000000 | nan |
| 4 | 0.416180 | 0.004673 | 0.000571 | nan | 0.243535 | 3 | yes | 19 | 0.027 | 0.076 | 0.062 | 0.098 | 1.000000 | nan |
| 5 | 0.406188 | 0.009992 | 0.000446 | nan | 0.018543 | 5 | yes | 19 | 0.027 | 0.076 | 0.094 | 0.099 | 1.000000 | nan |
| 6 | 0.377529 | 0.028659 | 0.000500 | nan | 0.741670 | 3 | yes | 19 | 0.027 | 0.076 | 0.062 | 0.116 | 1.000000 | nan |
| 7 | 0.354794 | 0.022735 | 0.000529 | nan | 0.073159 | 3 | yes | 19 | 0.027 | 0.076 | 0.062 | 0.116 | 1.000000 | nan |
| 8 | 0.345927 | 0.008867 | 0.000492 | nan | 0.971193 | 2 | yes | 19 | 0.027 | 0.076 | 0.046 | 0.098 | 1.000000 | nan |
| 9 | 0.345627 | 0.000301 | 0.000074 | nan | 1.055032 | 2 | yes | 19 | 0.030 | 0.076 | 0.046 | 0.097 | 1.000000 | nan |
| 10 | 0.345626 | 0.000000 | 0.000001 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.027 | 0.077 | 0.046 | 0.098 | 1.000000 | nan |
