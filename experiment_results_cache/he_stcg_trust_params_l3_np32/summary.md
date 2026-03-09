# HE STCG Trust-Parameter Sweep

Level `3`, `32` MPI ranks, full `24/24` trajectory.

Shared settings:
- `ksp_type=stcg`, `pc_type=gamg`
- `ksp_rtol=1e-1`, `ksp_max_it=100`
- rebuild PC every Newton iteration (`pc_setup_on_ksp_cap=False`)
- trust-region post line search on, `linesearch_tol=1e-1`
- line-search interval `[-0.5, 2.0]`
- `maxit=100`

| Backend | Stage | Label | All 24 converged | Total [s] | Newton | Linear | Radius | Shrink | Expand | Eta shrink | Eta expand | Max KSP it | KSP cap hits | Used max it | Final message | JSON |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| fenics_custom | radius | radius_0_5 | yes | 40.188 | 668 | 20154 | 0.5 | 0.5 | 1.5 | 0.05 | 0.75 | 93 | 0 | no | Gradient norm converged | `experiment_results_cache/he_stcg_trust_params_l3_np32/fenics_custom_radius_0_5.json` |
| fenics_custom | radius | radius_1_0 | yes | 39.069 | 655 | 20397 | 1 | 0.5 | 1.5 | 0.05 | 0.75 | 100 | 5 | yes | Gradient norm converged | `experiment_results_cache/he_stcg_trust_params_l3_np32/fenics_custom_radius_1_0.json` |
| fenics_custom | radius | radius_2_0 | yes | 41.348 | 654 | 21568 | 2 | 0.5 | 1.5 | 0.05 | 0.75 | 100 | 4 | yes | Gradient norm converged | `experiment_results_cache/he_stcg_trust_params_l3_np32/fenics_custom_radius_2_0.json` |
| fenics_custom | radius | radius_4_0 | yes | 41.075 | 643 | 21448 | 4 | 0.5 | 1.5 | 0.05 | 0.75 | 100 | 7 | yes | Energy change converged | `experiment_results_cache/he_stcg_trust_params_l3_np32/fenics_custom_radius_4_0.json` |
| jax_petsc_element | radius | radius_0_5 | yes | 54.769 | 720 | 23509 | 0.5 | 0.5 | 1.5 | 0.05 | 0.75 | 100 | 14 | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_trust_params_l3_np32/jax_petsc_element_radius_0_5.json` |
| jax_petsc_element | radius | radius_1_0 | yes | 57.259 | 716 | 23553 | 1 | 0.5 | 1.5 | 0.05 | 0.75 | 100 | 14 | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_trust_params_l3_np32/jax_petsc_element_radius_1_0.json` |
| jax_petsc_element | radius | radius_2_0 | yes | 57.957 | 712 | 25470 | 2 | 0.5 | 1.5 | 0.05 | 0.75 | 100 | 23 | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_trust_params_l3_np32/jax_petsc_element_radius_2_0.json` |
| jax_petsc_element | radius | radius_4_0 | yes | 59.322 | 711 | 26877 | 4 | 0.5 | 1.5 | 0.05 | 0.75 | 100 | 23 | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_trust_params_l3_np32/jax_petsc_element_radius_4_0.json` |
| fenics_custom | update | update_base_r1_0 | yes | 39.745 | 660 | 20922 | 1 | 0.5 | 1.5 | 0.05 | 0.75 | 100 | 3 | yes | Gradient norm converged | `experiment_results_cache/he_stcg_trust_params_l3_np32/fenics_custom_update_base_r1_0.json` |
| fenics_custom | update | update_expand2_r1_0 | yes | 42.091 | 647 | 20216 | 1 | 0.5 | 2 | 0.05 | 0.75 | 100 | 2 | yes | Gradient norm converged | `experiment_results_cache/he_stcg_trust_params_l3_np32/fenics_custom_update_expand2_r1_0.json` |
| fenics_custom | update | update_stricter_r1_0 | yes | 42.430 | 659 | 20771 | 1 | 0.5 | 1.5 | 0.1 | 0.9 | 100 | 1 | yes | Gradient norm converged | `experiment_results_cache/he_stcg_trust_params_l3_np32/fenics_custom_update_stricter_r1_0.json` |
| fenics_custom | update | update_strong_shrink_r1_0 | yes | 42.907 | 659 | 20542 | 1 | 0.25 | 1.5 | 0.1 | 0.75 | 100 | 3 | yes | Gradient norm converged | `experiment_results_cache/he_stcg_trust_params_l3_np32/fenics_custom_update_strong_shrink_r1_0.json` |
| jax_petsc_element | update | update_base_r0_5 | yes | 58.091 | 707 | 22880 | 0.5 | 0.5 | 1.5 | 0.05 | 0.75 | 100 | 11 | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_trust_params_l3_np32/jax_petsc_element_update_base_r0_5.json` |
| jax_petsc_element | update | update_expand2_r0_5 | yes | 60.405 | 725 | 23832 | 0.5 | 0.5 | 2 | 0.05 | 0.75 | 100 | 15 | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_trust_params_l3_np32/jax_petsc_element_update_expand2_r0_5.json` |
| jax_petsc_element | update | update_stricter_r0_5 | yes | 59.243 | 718 | 23236 | 0.5 | 0.5 | 1.5 | 0.1 | 0.9 | 100 | 13 | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_trust_params_l3_np32/jax_petsc_element_update_stricter_r0_5.json` |
| jax_petsc_element | update | update_strong_shrink_r0_5 | yes | 59.508 | 722 | 23917 | 0.5 | 0.25 | 1.5 | 0.1 | 0.75 | 100 | 18 | yes | Converged (energy, step, gradient) | `experiment_results_cache/he_stcg_trust_params_l3_np32/jax_petsc_element_update_strong_shrink_r0_5.json` |
