# jax_petsc_local_sfd_l6_np16

## Run Summary

| Field | Value |
|---|---|
| Solver | `jax_petsc_local_sfd` |
| Backend | `element` |
| Mesh level | `6` |
| MPI ranks | `16` |
| Result | `completed` |
| Failure mode | `-` |
| Total DOFs | `16641` |
| Free DOFs | `16129` |
| Setup time [s] | `0.864` |
| Total solve time [s] | `0.412` |
| Total Newton iterations | `8` |
| Total linear iterations | `33` |
| Total assembly time [s] | `0.159` |
| Total PC init time [s] | `0.106` |
| Total KSP solve time [s] | `0.056` |
| Total line-search time [s] | `0.085` |
| Final energy | `0.345777` |
| Raw JSON | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l6_np16.json` |
| Raw log | `artifacts/reproduction/2026-03-15_refactor_stage2b_final/full/gl_final_suite/jax_petsc_local_sfd_l6_np16.log` |

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
| `local_hessian_mode` | `sfd_local` |

## Step Summary

| Step | Time [s] | Newton | Linear | Energy | Message |
|---:|---:|---:|---:|---:|---|
| 1 | 0.412 | 8 | 33 | 0.345777 | Converged (energy, step, gradient) |

## Step 1

| Field | Value |
|---|---|
| Step time [s] | `0.412` |
| Newton iterations | `8` |
| Linear iterations | `33` |
| Energy | `0.345777` |
| Message | `Converged (energy, step, gradient)` |

| Newton | Energy | dE | Grad norm | Grad post | Alpha | KSP it | Accepted | LS evals | Assemble [s] | PC init [s] | Solve [s] | LS time [s] | Trust radius | Rho |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.564346 | 0.100996 | 0.007408 | nan | -0.268294 | 10 | yes | 19 | 0.153 | 0.015 | 0.015 | 0.011 | 1.000000 | nan |
| 2 | 0.509142 | 0.055204 | 0.008273 | nan | 0.335571 | 6 | yes | 19 | 0.001 | 0.014 | 0.010 | 0.011 | 1.000000 | nan |
| 3 | 0.442990 | 0.066153 | 0.004418 | nan | -0.276491 | 5 | yes | 19 | 0.001 | 0.014 | 0.009 | 0.011 | 1.000000 | nan |
| 4 | 0.370115 | 0.072875 | 0.006814 | nan | 0.275292 | 4 | yes | 19 | 0.001 | 0.013 | 0.007 | 0.011 | 1.000000 | nan |
| 5 | 0.346939 | 0.023176 | 0.005533 | nan | 0.845568 | 2 | yes | 19 | 0.001 | 0.012 | 0.004 | 0.011 | 1.000000 | nan |
| 6 | 0.345778 | 0.001161 | 0.001160 | nan | 1.071693 | 2 | yes | 19 | 0.001 | 0.013 | 0.004 | 0.011 | 1.000000 | nan |
| 7 | 0.345777 | 0.000001 | 0.000038 | nan | 1.001116 | 2 | yes | 19 | 0.001 | 0.012 | 0.004 | 0.011 | 1.000000 | nan |
| 8 | 0.345777 | 0.000000 | 0.000000 | 0.000000 | 1.002249 | 2 | yes | 19 | 0.001 | 0.012 | 0.004 | 0.011 | 1.000000 | nan |
