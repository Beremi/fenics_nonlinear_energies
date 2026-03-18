# fenics_custom_l7_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `7` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65735` |
| Setup time [s] | `0.228` |
| Total solve time [s] | `0.606` |
| Total Newton iterations | `8` |
| Total linear iterations | `34` |
| Total assembly time [s] | `0.070` |
| Total PC init time [s] | `0.164` |
| Total KSP solve time [s] | `0.125` |
| Total line-search time [s] | `0.222` |
| Final energy | `0.345662` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l7_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l7_np4.log` |

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
| 1 | 0.606 | 8 | 34 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.606` |
| Newton iterations | `8` |
| Linear iterations | `34` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.543717 | 0.121592 | 0.003705 | nan | -0.286788 | 9 | yes | 19 | 0.010 | 0.025 | 0.031 | 0.028 | 1.000000 | nan |
| 2 | 0.510962 | 0.032755 | 0.003945 | nan | 0.350667 | 6 | yes | 19 | 0.009 | 0.020 | 0.021 | 0.028 | 1.000000 | nan |
| 3 | 0.444677 | 0.066285 | 0.002288 | nan | -0.368794 | 6 | yes | 19 | 0.009 | 0.020 | 0.021 | 0.028 | 1.000000 | nan |
| 4 | 0.380087 | 0.064590 | 0.003621 | nan | 0.193986 | 4 | yes | 19 | 0.009 | 0.020 | 0.015 | 0.028 | 1.000000 | nan |
| 5 | 0.349793 | 0.030294 | 0.003115 | nan | 0.649366 | 3 | yes | 19 | 0.009 | 0.020 | 0.012 | 0.028 | 1.000000 | nan |
| 6 | 0.345683 | 0.004110 | 0.001144 | nan | 1.069860 | 2 | yes | 19 | 0.009 | 0.020 | 0.009 | 0.028 | 1.000000 | nan |
| 7 | 0.345662 | 0.000021 | 0.000072 | nan | 1.009313 | 2 | yes | 19 | 0.009 | 0.020 | 0.009 | 0.028 | 1.000000 | nan |
| 8 | 0.345662 | 0.000000 | 0.000001 | 0.000000 | 0.998851 | 2 | yes | 19 | 0.009 | 0.020 | 0.009 | 0.028 | 1.000000 | nan |
