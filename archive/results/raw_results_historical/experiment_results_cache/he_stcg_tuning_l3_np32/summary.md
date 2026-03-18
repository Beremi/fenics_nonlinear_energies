# HE STCG Trust-Region Sweep

Level `3`, `32` MPI ranks, full `24/24` trajectory.

Shared settings:
- `ksp_type=stcg`, `pc_type=gamg`
- `ksp_rtol=1e-1`, `ksp_max_it=100`
- rebuild PC every Newton iteration (`pc_setup_on_ksp_cap=False`)
- trust region on, `trust_radius_init=2.0`
- line search interval `[-0.5, 2.0]`
- `maxit=100`

| Backend | Variant | Post LS | LS tol | All 24 converged | Total [s] | Newton | Linear | Final energy | Max step [s] | Max KSP it | KSP cap hits | Newton maxit steps | Used max it | Final message | JSON |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| fenics_custom | stcg_only | no | - | yes | 41.883 | 743 | 17235 | 93.705064 | 2.481 | 79 | 0 | - | no | Gradient norm converged | `experiment_results_cache/he_stcg_tuning_l3_np32/fenics_custom_stcg_only.json` |
| fenics_custom | stcg_postls_tol1e-1 | yes | 1e-01 | yes | 37.526 | 645 | 20716 | 93.704397 | 2.212 | 100 | 3 | - | yes | Gradient norm converged | `experiment_results_cache/he_stcg_tuning_l3_np32/fenics_custom_stcg_postls_tol1e-1.json` |
| fenics_custom | stcg_postls_tol1e-3 | yes | 1e-03 | yes | 40.019 | 653 | 21173 | 93.704601 | 2.030 | 100 | 4 | - | yes | Gradient norm converged | `experiment_results_cache/he_stcg_tuning_l3_np32/fenics_custom_stcg_postls_tol1e-3.json` |
| fenics_custom | stcg_postls_tol1e-6 | yes | 1e-06 | yes | 45.946 | 656 | 21263 | 93.704637 | 2.189 | 100 | 7 | - | yes | Gradient norm converged | `experiment_results_cache/he_stcg_tuning_l3_np32/fenics_custom_stcg_postls_tol1e-6.json` |
| jax_petsc_element | stcg_only | no | - | no | 57.329 | 843 | 21029 | 93.705020 | 3.779 | 99 | 0 | - | no | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_tuning_l3_np32/jax_petsc_element_stcg_only.json` |
| jax_petsc_element | stcg_postls_tol1e-1 | yes | 1e-01 | yes | 54.887 | 711 | 25364 | 93.704737 | 2.626 | 100 | 24 | - | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_tuning_l3_np32/jax_petsc_element_stcg_postls_tol1e-1.json` |
| jax_petsc_element | stcg_postls_tol1e-3 | yes | 1e-03 | yes | 62.076 | 711 | 24951 | 93.704693 | 3.498 | 100 | 18 | - | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_tuning_l3_np32/jax_petsc_element_stcg_postls_tol1e-3.json` |
| jax_petsc_element | stcg_postls_tol1e-6 | yes | 1e-06 | yes | 69.343 | 713 | 25032 | 93.704234 | 3.969 | 100 | 17 | - | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_tuning_l3_np32/jax_petsc_element_stcg_postls_tol1e-6.json` |
