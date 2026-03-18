# fenics_custom_l6_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `fenics_custom` |
| Backend | `fenics` |
| Mesh level | `6` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `12545` |
| Free DOFs | `12513` |
| Setup time [s] | `0.051` |
| Total solve time [s] | `0.142` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.006` |
| Total PC init time [s] | `0.085` |
| Total KSP solve time [s] | `0.027` |
| Total line-search time [s] | `0.019` |
| Final energy | `-7.954564` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l6_np8.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/fenics_custom_l6_np8.log` |

## Frozen Settings

| Setting | Value |
|---|---|
| `ksp_type` | `cg` |
| `pc_type` | `hypre` |
| `ksp_rtol` | `0.1` |
| `ksp_max_it` | `30` |
| `use_trust_region` | `False` |
| `trust_subproblem_line_search` | `False` |
| `linesearch_interval` | `[-0.5, 2.0]` |
| `linesearch_tol` | `0.1` |
| `trust_radius_init` | `1.0` |
| `local_hessian_mode` | `-` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.142 | 6 | 11 | -7.954564 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.142` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.954564` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | -5.865915 | 6.079869 | 0.763320 | nan | 0.186438 | 1 | yes | 9 | 0.001 | 0.014 | 0.003 | 0.003 | 1.000000 | nan |
| 2 | -7.709561 | 1.843646 | 5.498669 | nan | 0.584072 | 2 | yes | 9 | 0.001 | 0.015 | 0.005 | 0.003 | 1.000000 | nan |
| 3 | -7.950743 | 0.241182 | 2.473069 | nan | 1.088137 | 2 | yes | 9 | 0.001 | 0.013 | 0.005 | 0.003 | 1.000000 | nan |
| 4 | -7.954519 | 0.003777 | 0.236369 | nan | 1.088137 | 2 | yes | 9 | 0.001 | 0.014 | 0.005 | 0.003 | 1.000000 | nan |
| 5 | -7.954563 | 0.000044 | 0.021852 | 0.001199 | 1.088137 | 2 | yes | 9 | 0.001 | 0.013 | 0.004 | 0.003 | 1.000000 | nan |
| 6 | -7.954564 | 0.000000 | 0.001199 | 0.000007 | 1.002033 | 2 | yes | 9 | 0.001 | 0.016 | 0.005 | 0.003 | 1.000000 | nan |
