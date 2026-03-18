# fenics_custom_l6_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16563` |
| Setup time [s] | `0.064` |
| Total solve time [s] | `0.255` |
| Total Newton iterations | `8` |
| Total linear iterations | `31` |
| Total assembly time [s] | `0.006` |
| Total PC init time [s] | `0.137` |
| Total KSP solve time [s] | `0.079` |
| Total line-search time [s] | `0.029` |
| Final energy | `0.345777` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l6_np16.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l6_np16.log` |

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
| 1 | 0.255 | 8 | 31 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.255` |
| Newton iterations | `8` |
| Linear iterations | `31` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.504296 | 0.161046 | 0.007408 | nan | -0.325875 | 7 | yes | 19 | 0.001 | 0.019 | 0.017 | 0.004 | 1.000000 | nan |
| 2 | 0.504222 | 0.000074 | 0.007500 | nan | -0.133772 | 6 | yes | 19 | 0.001 | 0.017 | 0.014 | 0.004 | 1.000000 | nan |
| 3 | 0.499023 | 0.005199 | 0.008476 | nan | -0.210713 | 6 | yes | 19 | 0.001 | 0.017 | 0.015 | 0.004 | 1.000000 | nan |
| 4 | 0.419011 | 0.080012 | 0.010072 | nan | 0.056498 | 4 | yes | 19 | 0.001 | 0.017 | 0.010 | 0.004 | 1.000000 | nan |
| 5 | 0.351309 | 0.067702 | 0.009162 | nan | 0.643703 | 2 | yes | 19 | 0.001 | 0.017 | 0.006 | 0.004 | 1.000000 | nan |
| 6 | 0.345793 | 0.005516 | 0.003047 | nan | 1.076924 | 2 | yes | 19 | 0.001 | 0.018 | 0.006 | 0.004 | 1.000000 | nan |
| 7 | 0.345777 | 0.000016 | 0.000172 | nan | 1.005482 | 2 | yes | 19 | 0.001 | 0.015 | 0.005 | 0.004 | 1.000000 | nan |
| 8 | 0.345777 | 0.000000 | 0.000001 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.001 | 0.016 | 0.005 | 0.004 | 1.000000 | nan |
