# jax_petsc_element_l9_np2

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_element` |
| Backend | `element` |
| Mesh level | `9` |
| MPI ranks | `2` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `788481` |
| Free DOFs | `784385` |
| Setup time [s] | `5.743` |
| Total solve time [s] | `5.899` |
| Total Newton iterations | `6` |
| Total linear iterations | `11` |
| Total assembly time [s] | `0.650` |
| Total PC init time [s] | `3.142` |
| Total KSP solve time [s] | `0.910` |
| Total line-search time [s] | `1.054` |
| Final energy | `-7.960005` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l9_np2.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/plaplace_final_suite/jax_petsc_element_l9_np2.log` |

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
| `local_hessian_mode` | `element` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 5.899 | 6 | 11 | -7.960005 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `5.899` |
| Newton iterations | `6` |
| Linear iterations | `11` |
| Energy | `-7.960005` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 331.391595 | 33077101.490768 | 448769.786628 | nan | 1.956948 | 2 | yes | 9 | 0.108 | 0.613 | 0.166 | 0.178 | 1.000000 | nan |
| 2 | -4.961125 | 336.352721 | 208.093747 | nan | 1.956948 | 2 | yes | 9 | 0.110 | 0.581 | 0.166 | 0.174 | 1.000000 | nan |
| 3 | -7.524305 | 2.563180 | 2.298523 | nan | 0.411863 | 1 | yes | 9 | 0.107 | 0.579 | 0.113 | 0.173 | 1.000000 | nan |
| 4 | -7.951217 | 0.426912 | 2.326601 | nan | 0.948817 | 2 | yes | 9 | 0.108 | 0.493 | 0.158 | 0.174 | 1.000000 | nan |
| 5 | -7.959962 | 0.008745 | 0.317225 | nan | 1.088137 | 2 | yes | 9 | 0.108 | 0.437 | 0.149 | 0.179 | 1.000000 | nan |
| 6 | -7.960005 | 0.000042 | 0.018726 | 0.001933 | 1.055248 | 2 | yes | 9 | 0.108 | 0.440 | 0.157 | 0.176 | 1.000000 | nan |
