# fenics_custom_l6_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16385` |
| Setup time [s] | `0.073` |
| Total solve time [s] | `0.245` |
| Total Newton iterations | `8` |
| Total linear iterations | `22` |
| Total assembly time [s] | `0.031` |
| Total PC init time [s] | `0.061` |
| Total KSP solve time [s] | `0.032` |
| Total line-search time [s] | `0.109` |
| Final energy | `0.345777` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l6_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/fenics_custom_l6_np2.log` |

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
| 1 | 0.245 | 8 | 22 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.245` |
| Newton iterations | `8` |
| Linear iterations | `22` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.507296 | 0.158046 | 0.007408 | nan | -0.499650 | 3 | yes | 19 | 0.004 | 0.010 | 0.004 | 0.014 | 1.000000 | nan |
| 2 | 0.490935 | 0.016361 | 0.008138 | nan | -0.047235 | 4 | yes | 19 | 0.004 | 0.007 | 0.005 | 0.014 | 1.000000 | nan |
| 3 | 0.476678 | 0.014257 | 0.008040 | nan | -0.086055 | 3 | yes | 19 | 0.004 | 0.008 | 0.004 | 0.013 | 1.000000 | nan |
| 4 | 0.432177 | 0.044501 | 0.008369 | nan | 0.046901 | 4 | yes | 19 | 0.004 | 0.008 | 0.005 | 0.014 | 1.000000 | nan |
| 5 | 0.349916 | 0.082261 | 0.008002 | nan | 0.551667 | 2 | yes | 19 | 0.004 | 0.007 | 0.003 | 0.014 | 1.000000 | nan |
| 6 | 0.345809 | 0.004107 | 0.002567 | nan | 1.037404 | 2 | yes | 19 | 0.004 | 0.007 | 0.003 | 0.014 | 1.000000 | nan |
| 7 | 0.345777 | 0.000032 | 0.000182 | nan | 1.014811 | 2 | yes | 19 | 0.004 | 0.007 | 0.003 | 0.014 | 1.000000 | nan |
| 8 | 0.345777 | 0.000000 | 0.000002 | 0.000000 | 1.000416 | 2 | yes | 19 | 0.004 | 0.007 | 0.003 | 0.014 | 1.000000 | nan |
