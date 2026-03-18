# fenics_custom_l6_np1

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `1` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16129` |
| Setup time [s] | `0.052` |
| Total solve time [s] | `0.515` |
| Total Newton iterations | `10` |
| Total linear iterations | `33` |
| Total assembly time [s] | `0.079` |
| Total PC init time [s] | `0.075` |
| Total KSP solve time [s] | `0.068` |
| Total line-search time [s] | `0.265` |
| Final energy | `0.345777` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l6_np1.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l6_np1.log` |

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
| 1 | 0.515 | 10 | 33 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.515` |
| Newton iterations | `10` |
| Linear iterations | `33` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.495636 | 0.169706 | 0.007408 | nan | -0.332939 | 7 | yes | 19 | 0.008 | 0.011 | 0.013 | 0.027 | 1.000000 | nan |
| 2 | 0.486099 | 0.009537 | 0.007068 | nan | -0.434572 | 5 | yes | 19 | 0.008 | 0.007 | 0.010 | 0.026 | 1.000000 | nan |
| 3 | 0.441368 | 0.044731 | 0.009953 | nan | 0.116345 | 4 | yes | 19 | 0.008 | 0.007 | 0.008 | 0.026 | 1.000000 | nan |
| 4 | 0.419236 | 0.022132 | 0.008019 | nan | 0.483788 | 3 | yes | 19 | 0.008 | 0.007 | 0.006 | 0.026 | 1.000000 | nan |
| 5 | 0.401400 | 0.017836 | 0.004405 | nan | 0.484488 | 3 | yes | 19 | 0.008 | 0.007 | 0.006 | 0.026 | 1.000000 | nan |
| 6 | 0.377110 | 0.024290 | 0.003580 | nan | 0.962996 | 2 | yes | 19 | 0.008 | 0.007 | 0.005 | 0.027 | 1.000000 | nan |
| 7 | 0.352588 | 0.024522 | 0.003588 | nan | 0.046201 | 3 | yes | 19 | 0.008 | 0.007 | 0.006 | 0.027 | 1.000000 | nan |
| 8 | 0.346049 | 0.006539 | 0.003292 | nan | 0.966394 | 2 | yes | 19 | 0.008 | 0.007 | 0.005 | 0.027 | 1.000000 | nan |
| 9 | 0.345777 | 0.000273 | 0.000549 | nan | 1.056865 | 2 | yes | 19 | 0.008 | 0.007 | 0.005 | 0.027 | 1.000000 | nan |
| 10 | 0.345777 | 0.000000 | 0.000012 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.008 | 0.007 | 0.005 | 0.027 | 1.000000 | nan |
