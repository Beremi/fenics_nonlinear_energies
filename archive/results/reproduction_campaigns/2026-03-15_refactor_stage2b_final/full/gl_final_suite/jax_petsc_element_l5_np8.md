# jax_petsc_element_l5_np8

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `5` |
| MPI ranks | `8` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `4225` |
| Free DOFs | `3969` |
| Setup time [s] | `0.530` |
| Total solve time [s] | `0.185` |
| Total Newton iterations | `9` |
| Total linear iterations | `36` |
| Total assembly time [s] | `0.003` |
| Total PC init time [s] | `0.057` |
| Total KSP solve time [s] | `0.031` |
| Total line-search time [s] | `0.090` |
| Final energy | `0.346231` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_element_l5_np8.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_element_l5_np8.log` |

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
| 1 | 0.185 | 9 | 36 | 0.346231 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.185` |
| Newton iterations | `9` |
| Linear iterations | `36` |
| Energy | `0.346231` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.565910 | 0.099559 | 0.014806 | nan | -0.259130 | 9 | yes | 19 | 0.000 | 0.007 | 0.007 | 0.008 | 1.000000 | nan |
| 2 | 0.509431 | 0.056479 | 0.016211 | nan | 0.334439 | 5 | yes | 19 | 0.000 | 0.007 | 0.004 | 0.008 | 1.000000 | nan |
| 3 | 0.437258 | 0.072173 | 0.007781 | nan | -0.411980 | 5 | yes | 19 | 0.000 | 0.007 | 0.004 | 0.008 | 1.000000 | nan |
| 4 | 0.426004 | 0.011255 | 0.013478 | nan | -0.017311 | 5 | yes | 19 | 0.000 | 0.006 | 0.004 | 0.008 | 1.000000 | nan |
| 5 | 0.369341 | 0.056663 | 0.013583 | nan | 0.252432 | 4 | yes | 19 | 0.000 | 0.006 | 0.003 | 0.028 | 1.000000 | nan |
| 6 | 0.347733 | 0.021607 | 0.011010 | nan | 0.812246 | 2 | yes | 19 | 0.000 | 0.006 | 0.002 | 0.008 | 1.000000 | nan |
| 7 | 0.346235 | 0.001498 | 0.002673 | nan | 1.083123 | 2 | yes | 19 | 0.000 | 0.006 | 0.002 | 0.008 | 1.000000 | nan |
| 8 | 0.346231 | 0.000004 | 0.000127 | nan | 1.004082 | 2 | yes | 19 | 0.000 | 0.006 | 0.002 | 0.008 | 1.000000 | nan |
| 9 | 0.346231 | 0.000000 | 0.000001 | 0.000000 | 1.001116 | 2 | yes | 19 | 0.000 | 0.006 | 0.002 | 0.008 | 1.000000 | nan |
