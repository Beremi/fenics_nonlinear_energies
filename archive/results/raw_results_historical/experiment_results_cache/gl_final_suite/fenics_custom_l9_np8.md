# fenics_custom_l9_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1050257` |
| Setup time [s] | `3.046` |
| Total solve time [s] | `4.674` |
| Total Newton iterations | `8` |
| Total linear iterations | `32` |
| Total assembly time [s] | `0.424` |
| Total PC init time [s] | `1.061` |
| Total KSP solve time [s] | `1.399` |
| Total line-search time [s] | `1.616` |
| Final energy | `0.345626` |
| Raw JSON | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np8.json` |
| Raw log | `experiment_results_cache/gl_final_suite/fenics_custom_l9_np8.log` |

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
| 1 | 4.674 | 8 | 32 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `4.674` |
| Newton iterations | `8` |
| Linear iterations | `32` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.506790 | 0.158509 | 0.000926 | nan | -0.304849 | 7 | yes | 19 | 0.056 | 0.157 | 0.284 | 0.203 | 1.000000 | nan |
| 2 | 0.506204 | 0.000586 | 0.000981 | nan | 0.166328 | 6 | yes | 19 | 0.053 | 0.129 | 0.248 | 0.202 | 1.000000 | nan |
| 3 | 0.479600 | 0.026604 | 0.000816 | nan | -0.499650 | 5 | yes | 19 | 0.053 | 0.130 | 0.210 | 0.200 | 1.000000 | nan |
| 4 | 0.430699 | 0.048901 | 0.001208 | nan | -0.076891 | 6 | yes | 19 | 0.052 | 0.129 | 0.246 | 0.199 | 1.000000 | nan |
| 5 | 0.350328 | 0.080371 | 0.001260 | nan | 0.643703 | 2 | yes | 19 | 0.052 | 0.129 | 0.103 | 0.203 | 1.000000 | nan |
| 6 | 0.345639 | 0.004689 | 0.000392 | nan | 1.084955 | 2 | yes | 19 | 0.053 | 0.128 | 0.103 | 0.203 | 1.000000 | nan |
| 7 | 0.345626 | 0.000013 | 0.000024 | nan | 1.004782 | 2 | yes | 19 | 0.052 | 0.129 | 0.103 | 0.203 | 1.000000 | nan |
| 8 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 0.999551 | 2 | yes | 19 | 0.052 | 0.129 | 0.103 | 0.203 | 1.000000 | nan |
