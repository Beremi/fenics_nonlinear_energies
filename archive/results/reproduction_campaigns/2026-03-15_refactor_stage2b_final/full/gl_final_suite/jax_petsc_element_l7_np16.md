# jax_petsc_element_l7_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `7` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `66049` |
| Free DOFs | `65025` |
| Setup time [s] | `0.666` |
| Total solve time [s] | `0.287` |
| Total Newton iterations | `6` |
| Total linear iterations | `21` |
| Total assembly time [s] | `0.010` |
| Total PC init time [s] | `0.112` |
| Total KSP solve time [s] | `0.046` |
| Total line-search time [s] | `0.111` |
| Final energy | `0.345662` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_element_l7_np16.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_element_l7_np16.log` |

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
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.287 | 6 | 21 | 0.345662 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.287` |
| Newton iterations | `6` |
| Linear iterations | `21` |
| Energy | `0.345662` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.487182 | 0.178127 | 0.003705 | nan | -0.342969 | 6 | yes | 19 | 0.002 | 0.020 | 0.012 | 0.019 | 1.000000 | nan |
| 2 | 0.462568 | 0.024615 | 0.003496 | nan | -0.499650 | 6 | yes | 19 | 0.002 | 0.019 | 0.012 | 0.018 | 1.000000 | nan |
| 3 | 0.371421 | 0.091147 | 0.005069 | nan | 0.245368 | 3 | yes | 19 | 0.002 | 0.019 | 0.007 | 0.018 | 1.000000 | nan |
| 4 | 0.346184 | 0.025237 | 0.003498 | nan | 0.965262 | 2 | yes | 19 | 0.002 | 0.019 | 0.005 | 0.018 | 1.000000 | nan |
| 5 | 0.345662 | 0.000521 | 0.000389 | nan | 1.029907 | 2 | yes | 19 | 0.002 | 0.018 | 0.005 | 0.019 | 1.000000 | nan |
| 6 | 0.345662 | 0.000000 | 0.000005 | 0.000000 | 0.999983 | 2 | yes | 19 | 0.002 | 0.018 | 0.005 | 0.018 | 1.000000 | nan |
