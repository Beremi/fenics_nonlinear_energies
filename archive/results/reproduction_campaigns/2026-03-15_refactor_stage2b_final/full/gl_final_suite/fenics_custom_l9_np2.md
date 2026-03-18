# fenics_custom_l9_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `9` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `1050625` |
| Free DOFs | `1048492` |
| Setup time [s] | `4.647` |
| Total solve time [s] | `19.900` |
| Total Newton iterations | `11` |
| Total linear iterations | `42` |
| Total assembly time [s] | `2.716` |
| Total PC init time [s] | `4.026` |
| Total KSP solve time [s] | `2.989` |
| Total line-search time [s] | `9.286` |
| Final energy | `0.345626` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l9_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l9_np2.log` |

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
| 1 | 19.900 | 11 | 42 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `19.900` |
| Newton iterations | `11` |
| Linear iterations | `42` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.495669 | 0.169631 | 0.000926 | nan | -0.329974 | 8 | yes | 19 | 0.252 | 0.413 | 0.521 | 0.856 | 1.000000 | nan |
| 2 | 0.487841 | 0.007827 | 0.000908 | nan | -0.416511 | 5 | yes | 19 | 0.235 | 0.360 | 0.339 | 0.843 | 1.000000 | nan |
| 3 | 0.468162 | 0.019679 | 0.001274 | nan | -0.002649 | 6 | yes | 19 | 0.236 | 0.362 | 0.397 | 0.841 | 1.000000 | nan |
| 4 | 0.445821 | 0.022341 | 0.001252 | nan | 0.064262 | 4 | yes | 19 | 0.233 | 0.359 | 0.279 | 0.835 | 1.000000 | nan |
| 5 | 0.416385 | 0.029436 | 0.001090 | nan | 0.335839 | 4 | yes | 19 | 0.236 | 0.360 | 0.281 | 0.838 | 1.000000 | nan |
| 6 | 0.397866 | 0.018519 | 0.000743 | nan | 0.222509 | 3 | yes | 19 | 0.233 | 0.358 | 0.223 | 0.836 | 1.000000 | nan |
| 7 | 0.365067 | 0.032799 | 0.000663 | nan | 0.834838 | 3 | yes | 19 | 0.271 | 0.364 | 0.227 | 0.843 | 1.000000 | nan |
| 8 | 0.348142 | 0.016925 | 0.000473 | nan | 0.351099 | 3 | yes | 19 | 0.235 | 0.358 | 0.222 | 0.851 | 1.000000 | nan |
| 9 | 0.345633 | 0.002509 | 0.000248 | nan | 1.118277 | 2 | yes | 19 | 0.244 | 0.365 | 0.167 | 0.852 | 1.000000 | nan |
| 10 | 0.345626 | 0.000006 | 0.000011 | nan | 1.007315 | 2 | yes | 19 | 0.270 | 0.363 | 0.167 | 0.850 | 1.000000 | nan |
| 11 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 0.994752 | 2 | yes | 19 | 0.270 | 0.364 | 0.167 | 0.840 | 1.000000 | nan |
