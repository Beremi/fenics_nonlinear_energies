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
| Setup time [s] | `3.287` |
| Total solve time [s] | `2.031` |
| Total Newton iterations | `8` |
| Total linear iterations | `37` |
| Total assembly time [s] | `0.187` |
| Total PC init time [s] | `0.595` |
| Total KSP solve time [s] | `0.542` |
| Total line-search time [s] | `0.571` |
| Final energy | `0.345626` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l9_np32.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l9_np32.log` |

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
| 1 | 2.031 | 8 | 37 | 0.345626 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `2.031` |
| Newton iterations | `8` |
| Linear iterations | `37` |
| Energy | `0.345626` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.566512 | 0.098788 | 0.000926 | nan | -0.262363 | 11 | yes | 19 | 0.026 | 0.084 | 0.115 | 0.060 | 1.000000 | nan |
| 2 | 0.506731 | 0.059780 | 0.001020 | nan | 0.332606 | 5 | yes | 19 | 0.020 | 0.073 | 0.058 | 0.060 | 1.000000 | nan |
| 3 | 0.426415 | 0.080316 | 0.000526 | nan | -0.499650 | 5 | yes | 19 | 0.020 | 0.063 | 0.054 | 0.059 | 1.000000 | nan |
| 4 | 0.384520 | 0.041895 | 0.000881 | nan | 0.091653 | 6 | yes | 19 | 0.018 | 0.073 | 0.115 | 0.079 | 1.000000 | nan |
| 5 | 0.352748 | 0.031772 | 0.000826 | nan | 0.424209 | 4 | yes | 19 | 0.028 | 0.075 | 0.079 | 0.081 | 1.000000 | nan |
| 6 | 0.345672 | 0.007076 | 0.000434 | nan | 1.027374 | 2 | yes | 19 | 0.023 | 0.078 | 0.053 | 0.082 | 1.000000 | nan |
| 7 | 0.345626 | 0.000046 | 0.000028 | nan | 1.010713 | 2 | yes | 19 | 0.027 | 0.079 | 0.035 | 0.076 | 1.000000 | nan |
| 8 | 0.345626 | 0.000000 | 0.000000 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.024 | 0.071 | 0.034 | 0.074 | 1.000000 | nan |
