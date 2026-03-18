# fenics_custom_l6_np4

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `4` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16527` |
| Setup time [s] | `0.064` |
| Total solve time [s] | `0.229` |
| Total Newton iterations | `9` |
| Total linear iterations | `31` |
| Total assembly time [s] | `0.018` |
| Total PC init time [s] | `0.088` |
| Total KSP solve time [s] | `0.045` |
| Total line-search time [s] | `0.068` |
| Final energy | `0.345777` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l6_np4.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l6_np4.log` |

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
| 1 | 0.229 | 9 | 31 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.229` |
| Newton iterations | `9` |
| Linear iterations | `31` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.501556 | 0.163786 | 0.007408 | nan | -0.318977 | 7 | yes | 19 | 0.002 | 0.011 | 0.010 | 0.008 | 1.000000 | nan |
| 2 | 0.496172 | 0.005384 | 0.007193 | nan | -0.413813 | 5 | yes | 19 | 0.002 | 0.010 | 0.007 | 0.008 | 1.000000 | nan |
| 3 | 0.433578 | 0.062594 | 0.010018 | nan | 0.046634 | 5 | yes | 19 | 0.002 | 0.009 | 0.007 | 0.008 | 1.000000 | nan |
| 4 | 0.397427 | 0.036150 | 0.009007 | nan | 0.541370 | 3 | yes | 19 | 0.002 | 0.010 | 0.005 | 0.008 | 1.000000 | nan |
| 5 | 0.383612 | 0.013815 | 0.004185 | nan | 0.056931 | 3 | yes | 19 | 0.002 | 0.010 | 0.004 | 0.008 | 1.000000 | nan |
| 6 | 0.355805 | 0.027807 | 0.004410 | nan | 0.751967 | 2 | yes | 19 | 0.002 | 0.010 | 0.003 | 0.008 | 1.000000 | nan |
| 7 | 0.346413 | 0.009392 | 0.003500 | nan | 0.677457 | 2 | yes | 19 | 0.002 | 0.010 | 0.003 | 0.008 | 1.000000 | nan |
| 8 | 0.345777 | 0.000636 | 0.000905 | nan | 1.079457 | 2 | yes | 19 | 0.002 | 0.010 | 0.003 | 0.008 | 1.000000 | nan |
| 9 | 0.345777 | 0.000000 | 0.000027 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.002 | 0.010 | 0.003 | 0.008 | 1.000000 | nan |
